import os
import numpy as np
import torch
import copy
import torch.distributed
import torchvision.models as models
import torchvision
from torchvision.transforms import functional as t_F
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *
from torch.utils.data import RandomSampler

class EpochBatchSampler(RandomSampler):
    """
    Sampler that can:
    1. Sample fixed per-epoch indices (by epoch ID)
    2. Sample specific batch index (batch_idx)
    3. Sample specific batch list (batch_list)
    """
    def __init__(self, data_source, generator=None, initial_epoch=0):
        super().__init__(data_source, generator=generator)
        self.data_source = data_source
        self.epoch = initial_epoch
        self.indices_epoch = {}     # {epoch_id: [idx1, idx2, ...]}
        self.indices_batch = {}     # {batch_id: [idx1, ..., idxB]}
        self.batch_size = None

        self.batch_idx = None       # single batch sampling
        self.batch_list = None      # multiple batches sampling

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.batch_idx = None
        self.batch_list = None

    def use_batch(self, batch_size):
        self.batch_size = batch_size
        self._create_batch_indices()

    def set_batch(self, batch_idx):
        self.batch_idx = batch_idx
        self.batch_list = None

    def set_batch_list(self, batch_list):
        self.batch_list = batch_list
        self.batch_idx = None

    def _create_batch_indices(self):
        assert self.batch_size is not None
        self.indices_batch.clear()
        batch_id = 0
        for ep, indices in self.indices_epoch.items():
            for i in range(0, len(indices), self.batch_size):
                self.indices_batch[batch_id] = indices[i:i + self.batch_size]
                batch_id += 1

    def get_batch_list_img_mapping(self):
        # Returns {first_image_id_in_batch: batch_id}
        return {self.indices_batch[bid][0]: bid for bid in self.batch_list}

    def __iter__(self):
        # batch list priority > single batch > per-epoch sampling
        if self.batch_list is not None:
            all_indices = [self.indices_batch[bid] for bid in self.batch_list]
            indices = [i for sub in all_indices for i in sub]
        elif self.batch_idx is not None:
            indices = self.indices_batch[self.batch_idx]
        else:
            if self.epoch not in self.indices_epoch:
                self.indices_epoch[self.epoch] = list(super().__iter__())
                # optional: deterministic sort → sorted(self.indices_epoch[self.epoch])
            indices = self.indices_epoch[self.epoch]
        return iter(indices)

    def __len__(self):
        # You can customize this based on your use case
        if self.batch_idx is not None:
            return len(self.indices_batch[self.batch_idx])
        elif self.batch_list is not None:
            return sum(len(self.indices_batch[bid]) for bid in self.batch_list)
        elif self.epoch in self.indices_epoch:
            return len(self.indices_epoch[self.epoch])
        return len(self.data_source)


def float_to_int_n_torch(tensor: torch.Tensor, n: int) -> torch.Tensor:
    levels = (1 << n) - 1
    return torch.clamp((tensor * levels).round(), 0, levels).to(torch.int32)

def int_n_to_float_torch(tensor: torch.Tensor, n: int) -> torch.Tensor:
    levels = (1 << n) - 1
    return (tensor.to(torch.float32) / levels).to(torch.float16)

def pack_tensor(tensor: torch.Tensor, n: int) -> torch.Tensor:
    """
    Fully vectorized packing: (N,) int32 -> (compressed,) uint8
    """
    tensor = tensor.flatten()
    device = tensor.device

    total_bits = tensor.numel() * n
    num_bytes = (total_bits + 7) // 8

    expanded = torch.zeros(total_bits, dtype=torch.uint8, device=device)

    for bit in range(n):
        expanded[bit::n] = ((tensor >> bit) & 1).to(torch.uint8)

    padded = torch.cat([expanded, torch.zeros(num_bytes * 8 - total_bits, dtype=torch.uint8, device=device)])
    packed = padded.view(-1, 8).flip(dims=[1])
    packed = (packed * (2 ** torch.arange(8, device=device))).sum(dim=1).to(torch.uint8)
    return packed


def unpack_tensor(packed: torch.Tensor, shape, n: int) -> torch.Tensor:
    """
    Fully vectorized unpacking: (compressed,) uint8 -> (N,) int32
    """
    device = packed.device

    unpacked_bits = ((packed.unsqueeze(1) >> torch.arange(7, -1, -1, device=device)) & 1).flatten()
    flat_len = torch.prod(torch.tensor(shape)).item()
    total_bits = flat_len * n
    unpacked_bits = unpacked_bits[:total_bits]

    bits = unpacked_bits.view(-1, n)
    powers = (2 ** torch.arange(n, device=device)).unsqueeze(0)
    values = (bits * powers).sum(dim=1).to(torch.int32)
    return values.view(shape)


def encode_soft_labels_with_sign(values: torch.Tensor, indices: torch.Tensor, num_classes: int, nbits: int = 9):
    """
    使用 float16 的 sign bit + mantissa 低位编码 index。
    sign = 0 → index
    sign = 1 → num_classes - index
    """
    max_representation = (1 << nbits)*2
    # DEBUG mode
    # print(f"max_representation: {max_representation}")  
    assert values.shape == indices.shape
    assert values.dtype == torch.float16
    assert indices.max() < max_representation
    assert nbits <= 10

    values_bits = values.view(torch.uint16).to(torch.int32)

    # 判断哪些 index 要反转（sign = 1）
    sign_mask = (indices >= (1 << (nbits))).to(torch.int32)  # 简单地把后一半类编号映射到 sign=1
    logical_index = torch.where(sign_mask == 0, indices, num_classes - indices)

    # 清除 mantissa 低位
    mantissa_mask = ~((1 << nbits) - 1)
    values_masked = values_bits & mantissa_mask

    # 嵌入逻辑 index 到 mantissa 低位
    embedded_bits = values_masked | (logical_index & ((1 << nbits) - 1))

    # 修改 sign bit
    embedded_bits = torch.where(
        sign_mask.bool(),
        embedded_bits | (1 << 15),  # set sign bit
        embedded_bits & ~(1 << 15)  # clear sign bit
    )

    return embedded_bits.to(torch.uint16).view(torch.float16)


def decode_soft_labels_with_sign(values: torch.Tensor, num_classes: int, nbits: int = 9):
    """
    解码 sign + mantissa 得到原始 index 和 soft label 的近似值
    """
    assert values.dtype == torch.float16

    values_bits = values.view(torch.uint16).to(torch.int32)

    # 取 sign bit
    sign_bit = (values_bits >> 15) & 1

    # 取 mantissa 低位（存储的是 index 或反向 index）
    logical_index = values_bits & ((1 << nbits) - 1)

    # 还原 index
    index = torch.where(sign_bit == 0, logical_index, num_classes - logical_index)

    # 清除 mantissa 低位，恢复近似 soft label
    value_bits = values_bits & ~((1 << nbits) - 1)
    value_bits = value_bits & ~(1 << 15)  # 清除 sign bit，恢复正数
    approx_values = value_bits.to(torch.uint16).view(torch.float16)

    return index.to(torch.long), approx_values


def retrieve_index_prior(args, target, mix_index):
    '''Unvectorized version of the function'''
    # reserved_number = int((args.ratio/100)*args.ncls)
    # reserved_index = []

    # for i in range(target.shape[0]):
    #     curr_target = target[i]
    #     curr_mix_index = mix_index[i].unsqueeze(0)
    #     curr_relevant = args.prior_info[curr_target].clone()  # ⭐️ clone防止污染原prior_info
    #     curr_relevant[curr_mix_index] = 0
    #     selected_wo_mix_index = torch.topk(curr_relevant, reserved_number-1)[1].cuda()
    #     top_k_relevance = torch.cat([selected_wo_mix_index, curr_mix_index], dim=0)
    #     top_k_relevance = torch.sort(top_k_relevance)[0]  # 排序
    #     reserved_index.append(top_k_relevance)

    # reserved_index = torch.stack(reserved_index, dim=0).cuda()
    
    '''Vectorized version of the function'''
    batch_size = target.shape[0]
    reserved_number = int((args.ratio/100)*args.ncls)

    curr_relevant = args.prior_info[target].clone().cuda(args.gpu) if hasattr(args, 'gpu') else args.prior_info[target].clone().cuda()# [batchsize, ncls]
    row_indices = torch.arange(batch_size, device=curr_relevant.device)
    
    # DEBUG mode
    # print(f"reserved_number: {reserved_number}")
    # print(f"args.ncls: {args.ncls}")
    # print(f"curr_relevant.shape: {curr_relevant.shape}")
    # print(f"mix_index.shape: {mix_index.shape}")
    # print(f"target.shape: {target.shape}")
    target = target.cuda(args.gpu) if hasattr(args, 'gpu') else target.cuda()# 确保 target 在正确的设备上
    mixed_class = target[mix_index]
    
    curr_relevant[row_indices, mixed_class] = float('-inf')

    topk_values, topk_indices = torch.topk(curr_relevant, k=reserved_number-1, dim=1)
    mixed_class = mixed_class.unsqueeze(1)  # [batchsize, 1]
    reserved_index = torch.cat([topk_indices, mixed_class], dim=1)  # [batchsize, reserved_number]
    reserved_index = torch.sort(reserved_index, dim=1)[0]  # sort按列排，取values

    return reserved_index

class RandomResizedCropWithCoords(torchvision.transforms.RandomResizedCrop):
    def __init__(self, **kwargs):
        super(RandomResizedCropWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords):
        try:
            reference = (coords.any())
        except:
            reference = False
        if not reference:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            coords = (i / img.size[1],
                      j / img.size[0],
                      h / img.size[1],
                      w / img.size[0])
            coords = torch.FloatTensor(coords)
        else:
            i = coords[0].item() * img.size[1]
            j = coords[1].item() * img.size[0]
            h = coords[2].item() * img.size[1]
            w = coords[3].item() * img.size[0]
        return t_F.resized_crop(img, i, j, h, w, self.size,
                                 self.interpolation), coords


class ComposeWithCoords(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(ComposeWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomHorizontalFlipWithRes':
                img, status = t(img, status)
            else:
                img = t(img)
        return img, status, coords


class RandomHorizontalFlipWithRes(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, status):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        if status is not None:
            if status == True:
                return t_F.hflip(img), status
            else:
                return img, status
        else:
            status = False
            if torch.rand(1) < self.p:
                status = True
                return t_F.hflip(img), status
            return img, status


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

def get_FKD_info(fkd_path):
    def custom_sort_key(s):
        # Extract numeric part from the string using regular expression
        numeric_part = int(s.split('_')[1].split('.tar')[0])
        return numeric_part
    
    max_epoch = len([f for f in os.listdir(fkd_path) if f.endswith(".tar")])
    epoch_file = torch.load(os.path.join(fkd_path, 'epoch_0.tar'))
    batch_size = epoch_file[0][0].shape[0]
    last_batch_size = epoch_file[-1][0].shape[0]
    len_batch_list = len(epoch_file)
    print('last batch size: {}'.format(last_batch_size))    
    num_img = batch_size * (len_batch_list - 1) + last_batch_size

    print('======= FKD: dataset info ======')
    print('path: {}'.format(fkd_path))
    print('num img: {}'.format(num_img))
    print('batch size: {}'.format(batch_size))
    print('max epoch: {}'.format(max_epoch))
    print('================================')
    return max_epoch, batch_size, num_img

   
class MultiDatasetImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, mode, dataset='imagenet1k', **kwargs):
        """
        The root should contain the train and val folder
        example: xxx/tiny-imagenet-200/train
        """
        super(MultiDatasetImageFolder, self).__init__(root, **kwargs)

        if mode != 'val':
            return  # only use as val dataset

        if dataset in ['imagenet1k', 'imagenet21k']:
            pass    # keep the original image folder

        elif dataset == 'tiny':
            base_dir = os.path.dirname(root)
            _, self.class_to_idx = MultiDatasetImageFolder.find_tiny_classes(os.path.join(base_dir, 'wnids.txt'))
            self.samples = MultiDatasetImageFolder.make_tiny_dataset(root, self.class_to_idx)
            self.targets = [s[1] for s in self.samples]
            assert len(self.samples) == len(self.targets), "samples and targets should have same length"
            assert len(set(self.targets)) == 200, "tiny imagenet should have 200 classes"
    
    @staticmethod
    def find_tiny_classes(class_file):
        # https://github.com/zeyuanyin/tiny-imagenet/blob/main/classification/tiny_imagenet_dataset.py
        with open(class_file) as r:
            classes = list(map(lambda s: s.strip(), r.readlines()))

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    @staticmethod
    def make_tiny_dataset(root, class_to_idx):
        # https://github.com/zeyuanyin/tiny-imagenet/blob/main/classification/tiny_imagenet_dataset.py
        images = []
        dir_path = root

        dirname = dir_path.split('/')[-1]
        if dirname == 'train':
            for fname in sorted(os.listdir(dir_path)):
                cls_fpath = os.path.join(dir_path, fname)
                if os.path.isdir(cls_fpath):
                    cls_imgs_path = os.path.join(cls_fpath, 'images')
                    for imgname in sorted(os.listdir(cls_imgs_path)):
                        path = os.path.join(cls_imgs_path, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)
        else:
            imgs_path = os.path.join(dir_path, 'images')
            imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

            with open(imgs_annotations) as r:
                data_info = map(lambda s: s.split('\t'), r.readlines())

            cls_map = {line_data[0]: line_data[1] for line_data in data_info}

            for imgname in sorted(os.listdir(imgs_path)):
                path = os.path.join(imgs_path, imgname)
                item = (path, class_to_idx[cls_map[imgname]])
                images.append(item)

        return images

class ImageFolder_FKD_MIX(MultiDatasetImageFolder):
    def __init__(self, fkd_path, mode, args_epoch=None, args_bs=None, args_use_batch=False, **kwargs):
        self.fkd_path = fkd_path
        self.mode = mode
        self.use_batch = args_use_batch  # modified to use batch
        super(ImageFolder_FKD_MIX, self).__init__(mode='train', **kwargs)
        self.batch_config = None  # [list(coords), list(flip_status)]
        self.batch_config_idx = 0  # index of processing image in this batch
        self.dataset = "" if 'dataset' not in kwargs else kwargs['dataset'] # use for imagenet21k
        if self.mode == 'fkd_load':
            max_epoch, batch_size, num_img = get_FKD_info(self.fkd_path)
            # if args_epoch > max_epoch:
            #     raise ValueError(f'`--epochs` should be no more than max epoch.')
            if args_bs != batch_size:
                if batch_size == 1000: # special case for ImageNet-1K, IPC=1
                    self.args_bs = batch_size
                else:
                    raise ValueError(f'`--batch-size` should be same in both saving and loading phase. ({args_bs} != {batch_size}) Please use `--gradient-accumulation-steps` to control batch size in model forward phase.')
            # self.img2batch_idx_list = torch.load('/path/to/img2batch_idx_list.tar')
            self.img2batch_idx_list = get_img2batch_idx_list(num_img=num_img, batch_size=batch_size, epochs=max_epoch)
            self.batch_num_per_epoch = len(self.img2batch_idx_list[0])
            self.epoch = None
            self.batch_idx_across_all_epochs = None
            self.batch_list = None
            self.batch_mapping = None
            self.all_epoch_configs = None   # ⭐️ 新增：预加载所有 epoch 的 config
            
    def set_all_epoch_configs(self, all_cfgs):
        """
        all_cfgs: list，长度 = max_epoch，
        all_cfgs[e] = torch.load('epoch_e.tar') 得到的 list
        """
        self.all_epoch_configs = all_cfgs
        
    # def __getitem__(self, index):
    #     path, target = self.samples[index]

    #     if self.mode == 'fkd_save':
    #         coords_ = None
    #         flip_ = None
    #         coords_cutout_ = None # for ImageNet-21K-P
    #     elif self.mode == 'fkd_load':
    #         if self.batch_config == None:
    #             raise ValueError('config is not loaded')
    #         assert self.batch_config_idx <= len(self.batch_config[0]), "batch config index should be less than length of batch config"

    #         coords_ = self.batch_config[0][self.batch_config_idx]

    #         if self.dataset == 'imagenet21k':
    #             coords_cutout_ = self.batch_config[1][self.batch_config_idx]
    #         else:
    #             flip_ = self.batch_config[1][self.batch_config_idx]

    #         self.batch_config_idx += 1
    #     else:
    #         raise ValueError('mode should be fkd_save or fkd_load')

    #     sample = self.loader(path)

    #     if self.transform is not None:
    #         if self.dataset == 'imagenet21k':
    #             sample_new, coords_status, coords_cutout = self.transform(sample, coords_, coords_cutout_)
    #             flip_status = None
    #         else:
    #             sample_new, flip_status, coords_status = self.transform(sample, coords_, flip_)
    #     else:
    #         flip_status = None
    #         coords_status = None

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     if self.dataset == 'imagenet21k': # also for fkd_load to prevent None type flip_status for _MapDatasetFetcher
    #         return sample_new, target, coords_status, coords_cutout, index

    #     # modifed to return index
    #     return sample_new, target, flip_status, coords_status, index
    
    
    def __getitem__(self, index):
        """
        带详细计时的 __getitem__
        用来分析 Dataset 中每一步的耗时：
        - cfg_time:       FKD batch_config 取数据
        - load_img_time:  读图 + 解码
        - transform_time: transform(...)
        """

        import time
        t0 = time.time()

        # ------------------------------------
        # 0. 取样本路径与类别
        # ------------------------------------
        path, target = self.samples[index]

        # ------------------------------------
        # 1. FKD 配置（coords_ / flip_ / mix params）
        # ------------------------------------
        if self.mode == 'fkd_save':
            coords_ = None
            flip_ = None
            coords_cutout_ = None  # for imagenet21k
        elif self.mode == 'fkd_load':
            if self.batch_config is None:
                raise ValueError("config is not loaded")
            assert self.batch_config_idx <= len(self.batch_config[0]), \
                f"batch_config_idx {self.batch_config_idx} >= len(batch_config[0])"

            coords_ = self.batch_config[0][self.batch_config_idx]

            if self.dataset == 'imagenet21k':
                coords_cutout_ = self.batch_config[1][self.batch_config_idx]
                flip_ = None
            else:
                flip_ = self.batch_config[1][self.batch_config_idx]
                coords_cutout_ = None

            self.batch_config_idx += 1
        else:
            raise ValueError("mode should be fkd_save or fkd_load")

        t1 = time.time()

        # ------------------------------------
        # 2. 读图 (PIL loader)
        # ------------------------------------
        sample = self.loader(path)
        t2 = time.time()

        # ------------------------------------
        # 3. 应用 transform（含 coords / flip / crop）
        # ------------------------------------
        if self.transform is not None:
            if self.dataset == 'imagenet21k':
                sample_new, coords_status, coords_cutout = self.transform(
                    sample, coords_, coords_cutout_
                )
                flip_status = None
            else:
                sample_new, flip_status, coords_status = self.transform(
                    sample, coords_, flip_
                )
        else:
            sample_new = sample
            flip_status = None
            coords_status = None

        t3 = time.time()

        # ------------------------------------
        # 4. target transform（通常没啥）
        # ------------------------------------
        if self.target_transform is not None:
            target = self.target_transform(target)

        # ------------------------------------
        # 5. 随机打印计时（避免刷屏）
        # ------------------------------------
        # if np.random.rand() < 0.001:
            # print(
            #     f"[getitem idx={index}] "
            #     f"cfg={t1 - t0:.4f}s, "
            #     f"load_img={t2 - t1:.4f}s, "
            #     f"transform={t3 - t2:.4f}s, "
            #     f"total={t3 - t0:.4f}s"
            # )

        # ------------------------------------
        # 6. 返回格式保持不变
        # ------------------------------------
        if self.dataset == 'imagenet21k':
            return sample_new, target, coords_status, coords_cutout, index

        return sample_new, target, flip_status, coords_status, index



    def load_batch_config(self, img_idx):
        """
        用预加载到内存的 all_epoch_configs 来取 batch config
        """
        if self.batch_list is not None:
            assert self.batch_mapping is not None
            current_batch_idx = self.batch_mapping[img_idx]
        else:
            assert self.batch_idx_across_all_epochs is not None
            current_batch_idx = self.batch_idx_across_all_epochs

        # 动态计算 epoch & 相对 batch_id
        epoch = current_batch_idx // self.batch_num_per_epoch
        batch_idx = current_batch_idx % self.batch_num_per_epoch

        # ⭐️ 必须已经预先 set_all_epoch_configs
        assert self.all_epoch_configs is not None, "all_epoch_configs is not set. Call set_all_epoch_configs() first."

        # 直接从内存 list 里取，不再 torch.load
        config = self.all_epoch_configs[epoch][batch_idx]

        self.batch_config_idx = 0
        self.batch_config = config[:2]  # [coords, flip_status]
        return config[2:]               # [mix_index, mix_lam, mix_bbox, soft_label]

        
    def set_batch(self, batch_idx):
        self.batch_idx_across_all_epochs = batch_idx

    def set_batch_list(self, batch_list, mapping):
        self.batch_list = batch_list
        self.batch_mapping = mapping

    def set_epoch(self, epoch):
        self.epoch = epoch

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, args, rand_index=None, lam=None, bbox=None):
    if args.mode == 'fkd_save':
        rand_index = torch.randperm(images.size()[0]).cuda(images.device)
        lam = np.random.beta(args.cutmix, args.cutmix)
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    elif args.mode == 'fkd_load':
        assert rand_index is not None and lam is not None and bbox is not None
        rand_index = rand_index.cuda(images.device)
        lam = lam
        bbx1, bby1, bbx2, bby2 = bbox
    else:
        raise ValueError('mode should be fkd_save or fkd_load')

    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images, args, rand_index=None, lam=None):
    if args.mode == 'fkd_save':
        rand_index = torch.randperm(images.size()[0]).cuda(images.device)
        lam = np.random.beta(args.mixup, args.mixup)
    elif args.mode == 'fkd_load':
        assert rand_index is not None and lam is not None
        rand_index = rand_index.cuda(images.device)
        lam = lam
    else:
        raise ValueError('mode should be fkd_save or fkd_load')

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None


def mix_aug(images, args, rand_index=None, lam=None, bbox=None):
    if args.mix_type == 'mixup':
        return mixup(images, args, rand_index, lam)
    elif args.mix_type == 'cutmix':
        return cutmix(images, args, rand_index, lam, bbox)
    else:
        return images, None, None, None


def get_img2batch_idx_list(num_img = 50000, batch_size = 1024, seed=42, epochs=300):
    train_dataset = torch.utils.data.TensorDataset(torch.arange(num_img))
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    img2batch_idx_list = []
    for epoch in range(epochs):
        img2batch_idx = {}
        for batch_idx, img_indices in enumerate(batch_sampler):
            img2batch_idx[img_indices[0]] = batch_idx

        img2batch_idx_list.append(img2batch_idx)
    return img2batch_idx_list

 

def cosine_temperature_schedule(t, n, epoch):
    return 1 + 0.5 * (t - 1) * (1 + math.cos(math.pi * epoch / n))

def exp_temperature_schedule(t_start, t_end, epoch, max_epoch):
    decay_rate = math.log(t_start / t_end) / max_epoch
    return max(t_end, t_start * math.exp(-decay_rate * epoch))

def power_temperature_schedule(t_start, t_end, epoch, max_epoch, power=2.0):
    ratio = 1 - (epoch / max_epoch)
    return t_end + (t_start - t_end) * (ratio ** power)

def linear_temperature_schedule(t_start, t_end, epoch, max_epoch, delay_ratio=0.6):
    delay_epoch = int(max_epoch * delay_ratio)
    if epoch < delay_epoch:
        return t_start
    decay_epoch = epoch - delay_epoch
    decay_total = max_epoch - delay_epoch
    return t_start - (t_start - t_end) * (decay_epoch / decay_total)
 
def load_model(args, model_name):
    orig_name = copy.deepcopy(model_name)
    prefix= model_name.split('_')[0]
    if args.dataset_name == 'imagewoof' or args.dataset_name == 'imagenet-nette':
        if prefix == 'ResNet18':
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, args.ncls)
        elif prefix == 'ResNet50':
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, args.ncls)
        elif prefix == 'Densenet121':
            model = models.densenet121(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, args.ncls)
        elif prefix == 'ShuffleNetV2':
            model = models.shufflenet_v2_x1_0(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, args.ncls)
        elif prefix == 'MobileNetV2':
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, args.ncls)
        elif prefix == 'AlexNet':
            model = models.alexnet(pretrained=False)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, args.ncls)
        else:
            raise ValueError('model_name should be one of ResNet18, ResNet50, Densenet121, ShuffleNetV2, MobileNetV2')
    elif args.dataset_name == 'imagenet1k':
        if prefix == 'ResNet18':
            model = models.resnet18(weights=None)
        elif prefix == 'ResNet50':
            model = models.resnet50(weights=None)
        elif prefix == 'ResNet101':
            model = models.resnet101(weights=None)
        elif prefix == 'Densenet121':
            model = models.densenet121(weights=None)
        elif prefix == 'Densenet169':
            model = models.densenet169(weights=None)
        elif prefix == 'Densenet201':
            model = models.densenet201(weights=None)
        elif prefix == 'Densenet161':
            model = models.densenet161(weights=None)
        elif prefix == 'MobileNetV2':
            model = models.mobilenet_v2(weights=None)
        elif prefix == 'ShuffleNetV2':
            model = models.shufflenet_v2_x0_5(weights=None)
        elif prefix == 'EfficientNet':
            model = models.efficientnet_b0(weights=None)
        elif prefix == 'AlexNet':
            model = models.alexnet(weights=None)
        else:
            raise ValueError('model_name should be one of ResNet18, ResNet50, ResNet101, Densenet121, Densenet169, Densenet201, Densenet161, MobileNetV2')
    else:
        if prefix == 'ResNet18':
            model = ResNet18(args.ncls)
        elif prefix == 'ResNet50':
            model = ResNet50(args.ncls)
        elif prefix == 'ResNet101':
            model = ResNet101(args.ncls)
        elif prefix == 'Densenet121':
            model = DenseNet121(args.ncls)
        elif prefix == 'Densenet169':
            model = DenseNet169(args.ncls)
        elif prefix == 'Densenet201':
            model = DenseNet201(args.ncls)
        elif prefix == 'Densenet161':
            model = DenseNet161(args.ncls)
        elif prefix == 'MobileNetV2':
            model = MobileNetV2(args.ncls)
        elif prefix == 'ShuffleNetV2':
            model = ShuffleNetV2(net_size=0.5, ncls=args.ncls)
        elif prefix == 'ConvNetW128':
            model = conv.ConvNet(channel=3, num_classes=args.ncls, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(args.input_size,args.input_size)) 
        else:
            raise ValueError('model_name should be one of ResNet18, ResNet50, ResNet101, Densenet121, Densenet169, Densenet201, Densenet161, MobileNetV2')
        
    model_weight_path = os.path.join(args.model_pool_dir, orig_name + '.pth')

    def pruning_classifier(model=None, classes=[]):
        try:
            model_named_parameters = [name for name, x in model.named_parameters()]
            for name, x in model.named_parameters():
                if (
                    name == model_named_parameters[-1]
                    or name == model_named_parameters[-2]
                ):
                    x.data = x[classes]
        except:
            print("ERROR in changing the number of classes.")
        return model
    
    if 'conv' in model_name:
        model = pruning_classifier(model, range(args.ncls))
        checkpoint = torch.load(
                model_weight_path, map_location="cpu",weights_only=True
            )
        model.load_state_dict(checkpoint["model"])
    else:
        state_dict = torch.load(model_weight_path, weights_only=True)
        model.load_state_dict(state_dict)
    
    return model


def count_img_files(directory):
    img_count = 0
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        # 统计所有 .jpg 和 .JPG 文件
        img_count += len([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))])

    return img_count


def load_online_model(model_name, args):
    if args.dataset_name == 'imagenet1k':
        if model_name == 'MobileNetV2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif model_name == 'ResNet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'ResNet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'Densenet121':
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        elif model_name == 'EfficientNet':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == 'ShuffleNetV2':
            model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        elif model_name == 'AlexNet':
            model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Model {model_name} is not supported")
    else:
        raise NotImplementedError(f"Online model loading for {args.dataset_name} is not supported yet")

    return model