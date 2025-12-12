import os
import numpy as np
import torch
import torchvision
import copy
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from torchvision.transforms import functional as t_F
# 获取当前脚本的目录
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *
from torch.utils.data import RandomSampler


from torch.utils.data import RandomSampler


class EpochBatchSampler(RandomSampler):
    """
    只负责：
    - indices_epoch[epoch] = 该 epoch 的样本 index 顺序（用 sampler_indices_dict 填）
    - indices_batch[global_batch_id] = 对应的 index 列表
    - batch_list: 本轮训练实际要跑的 global_batch_id 列表（step_list）
    """
    def __init__(self, data_source, generator=None, initial_epoch=0):
        super().__init__(data_source, generator=generator)
        self.data_source = data_source
        self.epoch = initial_epoch

        self.indices_epoch = {}   # {epoch_id: [idx1, idx2, ...]}
        self.indices_batch = {}   # {global_batch_id: [idx1, ..., idxB]}
        self.batch_size = None
        self.batch_list = None    # [global_batch_id0, global_batch_id1, ...]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def use_batch(self, batch_size):
        self.batch_size = batch_size
        self._create_batch_indices()

    def _create_batch_indices(self):
        assert self.batch_size is not None
        self.indices_batch.clear()

        global_batch_id = 0
        # 注意：这里遍历顺序要和当时保存 FKD 时一样
        for ep in sorted(self.indices_epoch.keys()):
            indices = self.indices_epoch[ep]
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                self.indices_batch[global_batch_id] = batch_indices
                global_batch_id += 1

        # 一共多少个 batch，可记在 dataset 里用
        self.batch_num_per_epoch = global_batch_id // len(self.indices_epoch)

    def set_batch_list(self, batch_list):
        self.batch_list = batch_list

    def __iter__(self):
        if self.batch_list is not None:
            # 把选中的所有 batch 拼起来
            all_indices = []
            for global_batch_id in self.batch_list:
                all_indices.extend(self.indices_batch[global_batch_id])
            return iter(all_indices)

        # 默认整 epoch 顺序
        if self.epoch not in self.indices_epoch:
            self.indices_epoch[self.epoch] = list(super().__iter__())
        return iter(self.indices_epoch[self.epoch])

    def __len__(self):
        if self.batch_list is not None:
            return sum(len(self.indices_batch[bid]) for bid in self.batch_list)
        if self.epoch in self.indices_epoch:
            return len(self.indices_epoch[self.epoch])
        return len(self.data_source)



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


class FKDConfigManager:
    """
    管理所有 epoch_x.tar 的 FKD config
    每个 cfg[epoch][local_batch_id] = [coords_list, flip_list 或 coords_cutout_list, mix_index, mix_lam, mix_bbox, soft_label]
    """
    def __init__(self, fkd_path):
        self.fkd_path = fkd_path
        self.max_epoch, self.batch_size, self.num_img = get_FKD_info(fkd_path)

        self.all_epoch_configs = []
        for e in range(self.max_epoch):
            path = os.path.join(self.fkd_path, f'epoch_{e}.tar')
            cfg = torch.load(path, map_location='cpu')  # list of batches
            self.all_epoch_configs.append(cfg)

        self.batch_num_per_epoch = len(self.all_epoch_configs[0])  # 假定每 epoch batch 数一致

    def global_to_local(self, global_batch_id):
        epoch_id = global_batch_id // self.batch_num_per_epoch
        local_batch_id = global_batch_id % self.batch_num_per_epoch
        return epoch_id, local_batch_id

    def get_batch_config(self, global_batch_id):
        epoch_id, local_batch_id = self.global_to_local(global_batch_id)
        cfg = self.all_epoch_configs[epoch_id][local_batch_id]
        # cfg: [coords_list, flip_or_cutout_list, mix_index, mix_lam, mix_bbox, soft_label]
        return cfg


from torchvision import datasets

class ImageFolderTensor(datasets.ImageFolder):
    """
    返回 Tensor 图片，而不是 PIL
    transform: 只做 ToTensor+Normalize（不能做随机增强）
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)     # PIL

        # 只做 ToTensor（Normalize 也可以，但最好放 soft_train 里）
        img = transforms.functional.to_tensor(img)  # PIL -> Tensor [C,H,W]

        return img, target, index



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