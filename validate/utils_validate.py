import numpy as np
import torch
import torchvision.transforms as transforms
import os
import sys
import torchvision
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *
from torch.utils.data import RandomSampler


# class EpochBatchSampler(RandomSampler):
#     """
#     Sampler that can:
#     1. Sample fixed per-epoch indices (by epoch ID)
#     2. Sample specific batch index (batch_idx)
#     3. Sample specific batch list (batch_list)
#     """
#     def __init__(self, data_source, generator=None, initial_epoch=0):
#         super().__init__(data_source, generator=generator)
#         self.data_source = data_source
#         self.epoch = initial_epoch
#         self.indices_epoch = {}     # {epoch_id: [idx1, idx2, ...]}
#         self.indices_batch = {}     # {batch_id: [idx1, ..., idxB]}
#         self.batch_size = None

#         self.batch_idx = None       # single batch sampling
#         self.batch_list = None      # multiple batches sampling

#     def set_epoch(self, epoch):
#         self.epoch = epoch
#         self.batch_idx = None
#         self.batch_list = None

#     def use_batch(self, batch_size):
#         self.batch_size = batch_size
#         self._create_batch_indices()

#     def set_batch(self, batch_idx):
#         self.batch_idx = batch_idx
#         self.batch_list = None

#     def set_batch_list(self, batch_list):
#         self.batch_list = batch_list
#         self.batch_idx = None

#     def _create_batch_indices(self):
#         assert self.batch_size is not None
#         self.indices_batch.clear()
#         batch_id = 0
#         for ep, indices in self.indices_epoch.items():
#             for i in range(0, len(indices), self.batch_size):
#                 self.indices_batch[batch_id] = indices[i:i + self.batch_size]
#                 batch_id += 1

#     def get_batch_list_img_mapping(self):
#         # Returns {first_image_id_in_batch: batch_id}
#         return {self.indices_batch[bid][0]: bid for bid in self.batch_list}

#     def __iter__(self):
#         # batch list priority > single batch > per-epoch sampling
#         if self.batch_list is not None:
#             all_indices = [self.indices_batch[bid] for bid in self.batch_list]
#             indices = [i for sub in all_indices for i in sub]
#         elif self.batch_idx is not None:
#             indices = self.indices_batch[self.batch_idx]
#         else:
#             if self.epoch not in self.indices_epoch:
#                 self.indices_epoch[self.epoch] = list(super().__iter__())
#                 # optional: deterministic sort → sorted(self.indices_epoch[self.epoch])
#             indices = self.indices_epoch[self.epoch]
#         return iter(indices)

#     def __len__(self):
#         # You can customize this based on your use case
#         if self.batch_idx is not None:
#             return len(self.indices_batch[self.batch_idx])
#         elif self.batch_list is not None:
#             return sum(len(self.indices_batch[bid]) for bid in self.batch_list)
#         elif self.epoch in self.indices_epoch:
#             return len(self.indices_epoch[self.epoch])
#         return len(self.data_source)


class ConfidenceFilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, pretrained_model, threshold=0.9, max_attempts=10, device='cuda'):
        self.dataset = dataset  # e.g., ImageFolder
        self.model = pretrained_model.eval().to(device)
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.device = device
        self.fallback_count = 0  # ✅ 新增：记录 fallback 的次数

    def __len__(self):
        return len(self.dataset)
    
    def reset_fallback_counter(self):
        self.fallback_count = 0

    def __getitem__(self, idx):
        for _ in range(self.max_attempts):
            image, label = self.dataset[idx]  # This will call transform (e.g., RandomResizedCrop)

            with torch.no_grad():
                img_input = image.unsqueeze(0).to(self.device)  # [1,C,H,W]
                output = self.model(img_input)
                prob = torch.softmax(output, dim=1)
                max_conf = torch.max(prob).item()

                if max_conf > self.threshold:
                    return image, label  # Accepted crop
                
        # ✅ fallback：记录一次 fallback
        self.fallback_count += 1
        return image, label


def get_bn_affine_params(model):
    """收集所有 BatchNorm 层中可训练的 weight 和 bias 参数"""
    bn_params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
            bn_params.append(m.weight)
            bn_params.append(m.bias)
    return bn_params

def test_time_training_bn_only(model, test_loader, ttt_steps=10, lr=1e-3, device='cuda'):
    model = model.to(device)
    model.eval()

    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        inputs.requires_grad = True

        model.train()  # 为了让 BN 层统计更新（或 affine 可调）

        # 只优化 BN 的 scale 和 bias 参数
        bn_params = get_bn_affine_params(model)
        optimizer = torch.optim.SGD(bn_params, lr=lr)

        for _ in range(ttt_steps):
            optimizer.zero_grad()
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            entropy.backward()
            optimizer.step()

def heuristic_avg(soft_labels):
    mask = (soft_labels == 0)
    missing_number = torch.sum(mask, dim=1)[0]
    # print(f"missing_number: {missing_number}")
    if missing_number == 0:
        # print("No missing labels, return original soft_labels")
        return soft_labels
    assumed_value = (1-torch.sum(soft_labels, dim=1))/(missing_number)
    # finds the indices of the elements that are equal to 0
    assumed_value_expanded = assumed_value.unsqueeze(1).expand_as(soft_labels)  # [batch_size, class_num]
    # fill the spot with 0 using assumed_value
    soft_labels[mask] = assumed_value_expanded[mask]
    return soft_labels


def heuristic_random(soft_labels):
    new_soft_labels = soft_labels.clone().cpu()
    
    mask = (new_soft_labels == 0)
    missing_number = torch.sum(mask, dim=1)  # 每个样本缺失的类别数
    missing_value = (1 - torch.sum(new_soft_labels, dim=1))  # 每个样本缺失的概率总和


    for i in range(new_soft_labels.shape[0]):
        num_missing = missing_number[i].item()
        if num_missing == 0:
            continue
        total_missing_val = missing_value[i].item()

        # 生成一组随机值并归一化
        rand_vals = torch.rand(num_missing)
        rand_vals /= rand_vals.sum()  # 归一化
        rand_vals *= total_missing_val  # 使总和为缺失值

        # 找到该样本中缺失的位置
        missing_indices = torch.where(mask[i])[0]
        new_soft_labels[i, missing_indices] = rand_vals

    return new_soft_labels.cuda()



def heuristic_prior(soft_labels, args, targets):
    soft_labels = soft_labels.clone()
    
    # 获取当前 batch 的 prior 信息：[B, C]
    batch_prior = args.prior_info[targets]  # [B, C]
    
    # 找出 soft_labels 中为 0 的位置
    mask = (soft_labels == 0)  # [B, C]
    
    # 每个样本的 missing_value = 1 - sum(non-zero部分)
    current_sum = torch.sum(soft_labels, dim=1, keepdim=True)  # [B, 1]
    missing_value = 1.0 - current_sum  # [B, 1]

    # 对应的 prior 部分，乘上 mask
    picked_prior = batch_prior * mask  # [B, C]
    prior_sum = torch.sum(picked_prior, dim=1, keepdim=True)  # [B, 1]
    
    # 避免除以0
    ratio = torch.zeros_like(prior_sum)
    ratio[prior_sum > 0] = missing_value[prior_sum > 0] / prior_sum[prior_sum > 0]
    
    # 构造填补后的值
    filled = picked_prior * ratio  # [B, C]
    
    # 把为0的位置填上新值
    soft_labels[mask] = filled[mask]

    return soft_labels




# keep top k largest values, and smooth others
def keep_top_k(p,k,n_classes=1000): # p is the softmax on label output
    if k == n_classes:
        return p

    values, indices = p.topk(k, dim=1)

    mask_topk = torch.zeros_like(p)
    mask_topk.scatter_(-1, indices, 1.0)
    top_p = mask_topk * p

    minor_value = (1 - torch.sum(values, dim=1)) / (n_classes-k)
    minor_value = minor_value.unsqueeze(1).expand(p.shape)
    mask_smooth = torch.ones_like(p)
    mask_smooth.scatter_(-1, indices, 0)
    smooth_p = mask_smooth * minor_value

    topk_smooth_p = top_p + smooth_p
    assert np.isclose(topk_smooth_p.sum().item(), p.shape[0]), f'{topk_smooth_p.sum().item()} not close to {p.shape[0]}'
    return topk_smooth_p


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups

def load_small_dataset_model(model, args):
    if model == 'ResNet18':
        net = ResNet18(args.ncls)
    elif model == 'ResNet50':
        net = ResNet50(args.ncls)
    elif model == 'ResNet101':
        net = ResNet101(args.ncls)
    return net

def load_val_loader(args):
    if args.dataset_name == "cifar100" or args.dataset_name == "cifar10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
        ])
    elif args.dataset_name == "imagenet1k" or args.dataset_name == "imagewoof" or args.dataset_name=='imagenet-nette':
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
        ])
    elif args.dataset_name == "tiny_imagenet":
         transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
        ])
    else:
        raise NotImplementedError(f"dataset {args.dataset_name} not implemented")
    
    test_set = torchvision.datasets.ImageFolder(root=args.val_dir, transform=transform_test)

    # load dataset for CIFAR-100 
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=16,pin_memory=True)
    return testloader
