import argparse
import math
import os
import shutil
import sys
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import InterpolationMode
from utils_validate import *
from timm.data import Mixup
import hashlib
# It is imported for you to access and modify the PyTorch source code (via Ctrl+Click), more details in README.md
from torch.utils.data._utils.fetch import _MapDatasetFetcher
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *
from relabel.utils_fkd_reconstruct import *
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)
def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)
from torch.utils.data import RandomSampler
import random
import yaml


def worker_init_fn(worker_id):
    # 不同 worker 的 seed 保证不同，但全局可复现
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def seed_everything(seed: int = 42):
    import os, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def load_config_from_yaml(args):
    config_path = args.config_path
    with open(config_path, 'r') as f:
        all_config = yaml.safe_load(f)

    dataset = args.dataset_name
    model = args.model
    ipc = str(args.ipc)

    if dataset not in all_config:
        raise ValueError(f"Dataset {dataset} not found in config")

    cfg = all_config[dataset]

    for key, value in cfg.items():
        if key != "hyperparams":
            setattr(args, key, value)

    try:
        args.adamw_lr, args.eta = cfg['hyperparams'][model][ipc]
    except KeyError:
        raise ValueError(f"No hyperparams found for dataset={dataset}, model={model}, ipc={ipc}")


def get_args():
    parser = argparse.ArgumentParser("FKD Training on Cifar-100")
    parser.add_argument('--exp-name', type=str,
                        default="", help='the name of the run')
    parser.add_argument('--original-data-path', required='True', type=str,
                        help='name of the original data')
    parser.add_argument('--simple', default=False,action='store_true',)
    parser.add_argument('--fkd-path', required='True', type=str,
                        help='path to the fkd labels')
    parser.add_argument('--output-dir', required='True', type=str,
                        help='output directory')
    parser.add_argument('--dataset-name',default='cifar100',type=str,
                        help='dataset name')
    parser.add_argument('--min-scale', type=float, default=0.08, )
    parser.add_argument('--batch-size', type=int,
                        default=16, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('-j', '--workers', default=2, type=int,
                        help='number of data loading workers')
    parser.add_argument('--ipc',type=int,help='number of images per class')
    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--sgd-lr', type=float,
                        default=0.01, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,
                        default=0.5, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,
                        default=1e-4, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')
    parser.add_argument('--model', type=str,
                        default='ResNet18', help='student model name')
    parser.add_argument('--keep-topk', type=int, default=1000,
                        help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    parser.add_argument('--wandb-project', type=str,
                        default='RankDD', help='wandb project name')
    parser.add_argument('--wandb-api-key', type=str,
                        default=None, help='wandb api key')
    parser.add_argument('--mix-type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,
                        help='seed for batch loading sampler')
    parser.add_argument('--val-dir', required=True, type=str,
                        help="path to the validation data")

    # hard soft labels training configurations
    parser.add_argument('--setting', type=str, default='S6',)
    parser.add_argument('--hard-epochs', type=int, default=200,)
    parser.add_argument('--soft-epochs', type=int, default=100,)
    parser.add_argument('--hard-batch-size', type=int, default=16)
    parser.add_argument('--n', type=int)
    parser.add_argument('--label-smoothing', type=float, default=0.8,)

    parser.add_argument('--config-path', type=str, required=True,)
    args = parser.parse_args()

    args.mode = 'fkd_load'

    load_config_from_yaml(args)

    args.epochs = args.hard_epochs + args.soft_epochs
    
    if args.setting == "S1":
        args.epochs_weight = [args.hard_epochs, args.soft_epochs]
        args.training_type = ['hard', 'soft']
    elif args.setting == "S2":
        args.epochs_weight, args.training_type = generate_epoch_plan(args.n, args.epochs, start_pattern='hard')
    elif args.setting == "S3":
        args.epochs_weight, args.training_type = generate_epoch_plan(2*args.n, args.epochs, start_pattern='hard')
    elif args.setting == "S4":
        args.epochs_weight, args.training_type = generate_epoch_plan(args.n, args.epochs, start_pattern='soft')
    elif args.setting == "S5":
        args.epochs_weight, args.training_type = generate_epoch_plan(2*args.n, args.epochs, start_pattern='soft')
    elif args.setting == "S6":
        args.epochs_weight = [args.soft_epochs//2, args.hard_epochs,args.soft_epochs-args.soft_epochs//2]
        args.training_type = ['soft', 'hard', 'soft']
    elif args.setting == "S7":
        args.epochs_weight = [args.hard_epochs//2, args.soft_epochs,args.hard_epochs-args.hard_epochs//2]
        args.training_type = ['hard', 'soft', 'hard']
    elif args.setting == "S8":
        args.epochs_weight = [args.soft_epochs, args.hard_epochs]
        args.training_type = ['soft', 'hard']
    else:
        raise ValueError("setting not supported")
    print(args)

    args.output_dir = os.path.join(args.output_dir, args.dataset_name, args.exp_name)
    # args.prior_info = torch.load(args.prior_dir, weights_only=True).cuda()
    # print(args.prior_info[0])
    # whether you want to test if the pregenerated soft labels are aligned with the cropped regions
    args.test_model = load_online_model('ResNet18', args).cuda()
    # args.test_model.train()
    
    return args


def generate_epoch_plan(n, total_epoch, start_pattern='hard'):
    """
    生成一个训练计划：
    - length_list: 包含每段的 epoch 数，例如 [n, n, ..., remaining]
    - pattern_list: 对应每段的模式，例如 ['hard', 'soft', ...]

    参数：
    - n: 每段的长度（除最后一段）
    - total_epoch: 总训练轮数
    - start_pattern: 'hard' 或 'soft'，决定 pattern 的起始模式

    返回：
    - length_list, pattern_list
    """
    assert start_pattern in ['hard', 'soft'], "start_pattern 必须是 'hard' 或 'soft'"

    # 构造长度列表
    num_full_blocks = total_epoch // n
    remaining = total_epoch % n

    epoch_weight = [n] * num_full_blocks
    if remaining > 0:
        epoch_weight.append(remaining)

    # 构造 pattern 列表
    training_type = []
    current = start_pattern
    for _ in range(len(epoch_weight)):
        training_type.append(current)
        current = 'soft' if current == 'hard' else 'hard'
        
    assert len(epoch_weight) == len(training_type), "should be same length"

    return epoch_weight, training_type


def is_special_epoch(epoch, total_epochs):
    in_last_80_percent = epoch >= int(total_epochs * 0.8)
    ends_with_9_or_last = (epoch % 10 == 9) or (epoch == total_epochs - 1)
    return in_last_80_percent and ends_with_9_or_last

def sim(images,model):
    images = images.cuda()
    output = model(images)
    
    output = F.softmax(output/20, dim=1)
    return output


def build_student_model(args):

    model_dict = {
        'ResNet18': (ResNet18, models.resnet18),
        'ResNet50': (ResNet50, models.resnet50),
        'ResNet101': (ResNet101, models.resnet101),
        'MobileNetV2': (MobileNet_V2, models.mobilenet_v2),
        'Densenet121': (DenseNet121, models.densenet121),
    }

    if args.model not in model_dict:
        raise ValueError(f"Unsupported model: {args.model}")

    small_res_model, imagenet_model_fn = model_dict[args.model]

    if args.input_size <= 64:
        # resnet for small resolutions
        model = small_res_model(args.ncls)
    else:
        # resolutions for 224*224
        model = imagenet_model_fn(pretrained=False)
        if args.ncls != 1000:
            model.fc = nn.Linear(model.fc.in_features, args.ncls)

    return model


import torchvision.transforms.functional as F_vision

import torch
import torchvision.transforms.functional as F_vision
from torchvision.transforms import InterpolationMode


def apply_fkd_transform_batch_tensor(
    images_tensor,    # [B, C, H, W]，可以在 CPU 或 GPU
    coords_list,      # list / numpy / Tensor，形状[B, 4]，每行为(i_ratio, j_ratio, h_ratio, w_ratio)
    flip_list,        # list[bool] 或 bool Tensor
    input_size,       # 裁剪后目标边长
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    """
    更高效版本：
      - coords & flip 在 CPU 上处理
      - images_tensor 保持原 device（CPU / GPU 均可）
      - 无 GPU .item() 同步
    """
    # ====== 1. 预处理 coords / flip 到 CPU Tensor ======
    if isinstance(coords_list, torch.Tensor):
        coords = coords_list.detach().to(dtype=torch.float32, device='cpu')
    else:
        coords = torch.as_tensor(coords_list, dtype=torch.float32, device='cpu')

    if isinstance(flip_list, torch.Tensor):
        flips = flip_list.detach().to(dtype=torch.bool, device='cpu')
    else:
        flips = torch.as_tensor(flip_list, dtype=torch.bool, device='cpu')

    B, C, H, W = images_tensor.shape
    out_imgs = []

    # ====== 2. 按图像循环做裁剪 + 翻转 ======
    # 注意：这里只在 CPU 的 coords 上 .item()，不会触发 GPU 同步
    for idx in range(B):
        img = images_tensor[idx]  # 保持原 device

        i_ratio, j_ratio, h_ratio, w_ratio = coords[idx]

        i = int(i_ratio.item() * H)
        j = int(j_ratio.item() * W)
        h = int(h_ratio.item() * H)
        w = int(w_ratio.item() * W)

        # 在 img 所在的 device 上做裁剪 + resize
        img = F_vision.resized_crop(
            img,
            top=i,
            left=j,
            height=h,
            width=w,
            size=[input_size, input_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        if flips[idx]:
            img = F_vision.hflip(img)

        out_imgs.append(img)

    imgs = torch.stack(out_imgs, dim=0)  # [B, C, input_size, input_size]

    # ====== 3. 批量 Normalize ======
    mean_t = torch.as_tensor(mean, dtype=imgs.dtype, device=imgs.device).view(1, C, 1, 1)
    std_t  = torch.as_tensor(std,  dtype=imgs.dtype, device=imgs.device).view(1, C, 1, 1)
    imgs = (imgs - mean_t) / std_t

    return imgs



def main():
    args = get_args()
    
    # set up wandb
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project, entity="Soft_label_prunning", dir="./")
    wandb.run.name = args.exp_name

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    print(args.original_data_path)
    assert os.path.exists(args.original_data_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # soft label related Data loading
    sampler_indices_dict_dir = os.path.join(args.fkd_path, 'sampler_indices_dict.pt')
    sampler_indices_dict = torch.load(sampler_indices_dict_dir, weights_only=True)


    # soft label related Data loading
    soft_train_dataset = ImageFolderTensor(
        root=args.original_data_path,
        transform=None,
        target_transform=None,
    )

    real_teacher = models.resnet18(pretrained=True).cuda()
    real_teacher.train()
    
    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = EpochBatchSampler(soft_train_dataset, generator=generator)

    sampler_indices_dict_dir = os.path.join(args.fkd_path, 'sampler_indices_dict.pt')
    sampler_indices_dict = torch.load(sampler_indices_dict_dir, weights_only=True)
    sampler.indices_epoch = {int(k): v for k, v in sampler_indices_dict.items()}

    sampler.use_batch(args.batch_size)

    soft_train_loader = torch.utils.data.DataLoader(
        soft_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        generator=generator,
        worker_init_fn=worker_init_fn,
        num_workers=args.workers,
        persistent_workers=True,
        pin_memory=True,
    )
    args.soft_train_loader = soft_train_loader
    
    fkd_manager = FKDConfigManager(args.fkd_path)
    args.fkd_manager = fkd_manager

    # 你之前用的 normalize
    normalize = transforms.Normalize(mean=args.mean_norm, std=args.std_norm)

    # 这是你原来的增广流程，只是从 Dataset 拿出来放这里
    args.fkd_transform = ComposeWithCoords(transforms=[
        RandomResizedCropWithCoords(
            size=args.input_size,
            scale=(args.min_scale, 1.0),
            interpolation=InterpolationMode.BILINEAR
        ),
        RandomHorizontalFlipWithRes(),
        transforms.ToTensor(),
        normalize,
    ])

    # hard label related Data loading
    hard_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(args.mean_norm, args.std_norm)
    ])

    base_dataset = datasets.ImageFolder(root=args.original_data_path, transform=hard_transform)

    hard_train_loader = torch.utils.data.DataLoader(
        base_dataset,
        batch_size=args.hard_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # load validation data
    val_loader = load_val_loader(args)

    # load student model
    print("=> loading student model '{}'".format(args.model))
    model = build_student_model(args).cuda()
    model.train()
        
    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.sgd_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (1. + math.cos(math.pi * step / args.epochs / args.eta)) if step <= args.epochs else 0, last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0-step/args.epochs) if step <= args.epochs else 0, last_epoch=-1)

 
    args.best_acc1=0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.soft_train_loader = soft_train_loader
    args.hard_train_loader = hard_train_loader
    args.mixup_fn = Mixup(
                        mixup_alpha=0.0,
                        cutmix_alpha=1.0,
                        label_smoothing=args.label_smoothing,  # 适配 soft labels
                        num_classes=args.ncls
                    )
    args.val_loader = val_loader
    epoch = 0
    times = 0
    for cur_type, cur_epoch_length in zip(args.training_type, args.epochs_weight):
        cur_epoch_length = int(cur_epoch_length)
        pool = []
        
        for _ in range(cur_epoch_length):
            global wandb_metrics
            wandb_metrics = {}
            if not pool:
                total_steps = args.n * args.fkd_manager.batch_num_per_epoch
                g = torch.Generator()
                g.manual_seed(args.fkd_seed + times)
                pool = torch.randperm(total_steps, generator=g).tolist()
                times += 1
            batch_size = args.fkd_manager.batch_num_per_epoch

            step_list = pool[:batch_size]   # 取前 batch_size 个
            pool = pool[batch_size:]        # 剩余的继续用

            run_one_epoch(cur_type, model, args, epoch, optimizer, scheduler, step_list, real_teacher, sampler=sampler)
            epoch += 1
   
            
import time

def run_one_epoch(train_type, model, args, epoch, optimizer, scheduler,  step_list, real_teacher=None, sampler=None):
    print(f"\nEpoch: {epoch}")
    t0 = time.time()

    if train_type == 'soft':
        soft_train(model, args, step_list, epoch,real_teacher=real_teacher)

    elif train_type == 'hard':
        hard_train(model, args, epoch)
    else:
        raise ValueError("Training type must be 'hard' or 'soft'")

    # 2️⃣ 验证
    should_validate = (epoch % 10 == 0 or epoch == args.epochs - 1) if not args.simple else is_special_epoch(epoch, args.epochs)
    if should_validate:
        top1 = validate(model, args, epoch)
    else:
        top1 = 0

    wandb.log(wandb_metrics)

    scheduler.step()
    
    is_best = top1 > args.best_acc1
    args.best_acc1 = max(top1, args.best_acc1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc1': args.best_acc1,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, is_best, output_dir=args.output_dir, epoch=epoch + 1)



def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def is_topk_order_fully_equal(t1: torch.Tensor, t2: torch.Tensor, topk: int = 500) -> bool:
    t1_topk_idx = torch.topk(t1, topk, dim=1).indices
    t2_topk_idx = torch.topk(t2, topk, dim=1).indices

    t1_topk = torch.gather(t1, dim=1, index=t1_topk_idx)
    t2_topk = torch.gather(t2, dim=1, index=t2_topk_idx)

    t1_rank = torch.argsort(torch.argsort(t1_topk, dim=1), dim=1)
    t2_rank = torch.argsort(torch.argsort(t2_topk, dim=1), dim=1)

    return torch.equal(t1_rank, t2_rank)


def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def hard_train(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # args.hard_train_loader.dataset.reset_fallback_counter()

    optimizer = args.optimizer
    scheduler = args.scheduler
    criterion = soft_cross_entropy  # 适配 CutMix
    model.train()
    t1 = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(args.hard_train_loader):
        if len(inputs) % 2 != 0:  # 如果 batch_size 是奇数，丢掉最后一个样本
            inputs = inputs[:-1]
            labels = labels[:-1]
            
        inputs, labels_mixed = inputs.to('cuda'), labels.to('cuda')
        inputs, labels_mixed = args.mixup_fn(inputs, labels)  # 应用 CutMix
        labels_mixed = labels_mixed.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_mixed)
        loss.backward()
        n = inputs.size(0)
        prec1, prec5 = accuracy(outputs, labels.cuda(), topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        optimizer.step()
        
    metrics = {
    "train/loss": objs.avg,
    "train/Top1": top1.avg,
    "train/Top5": top5.avg,
    "train/lr": scheduler.get_last_lr()[0],
    "train/epoch": epoch,}
    wandb_metrics.update(metrics)


    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    print(printInfo)
    t1 = time.time()
        
def benchmark_soft_dataloader(args):
    import time
    
    step_list = []
    total_steps = args.n * args.soft_train_loader.dataset.batch_num_per_epoch
    pool = list(range(total_steps))
    for i in range(args.soft_train_loader.dataset.batch_num_per_epoch):
        random_step = torch.randint(0, len(pool), (1,)).item()
        step_list.append(pool[random_step])

    args.soft_train_loader.sampler.set_batch_list(step_list)
    mappings = args.soft_train_loader.sampler.get_batch_list_img_mapping()
    args.soft_train_loader.dataset.set_batch_list(step_list, mappings)
    t0 = time.time()
    n_batch = 0
    for batch_data in args.soft_train_loader:
        # 不做任何运算，只是把 batch 取出来
        n_batch += 1
    t1 = time.time()
    print(f"[BENCH] soft dataloader: {n_batch} batches, "
          f"total {t1 - t0:.2f}s, avg {(t1 - t0) / n_batch:.4f}s / batch")


def soft_train(model, args, step_list, epoch=None, real_teacher=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')

    model.train()
    real_teacher.train()
    t1 = time.time()
    
    args.soft_train_loader.sampler.set_batch_list(step_list)

    for batch_idx, batch in enumerate(args.soft_train_loader):

        # Dataloader 返回：images_tensor: [B, C, H, W]
        images_tensor, target, indices = batch

        target = target.cuda(non_blocking=True)
        images_tensor = images_tensor.cuda(non_blocking=True)

        global_batch_id = step_list[batch_idx]
        (
            coords_list,
            flip_list_or_cutout,
            mix_index,
            mix_lam,
            mix_bbox,
            soft_label,
        ) = args.fkd_manager.get_batch_config(global_batch_id)

        # 普通数据集：第二项就是 flip_list_or_cutout
        images = apply_fkd_transform_batch_tensor(
            images_tensor,
            coords_list,
            flip_list_or_cutout,
            args.input_size,
            mean=args.mean_norm,
            std=args.std_norm,
        )

        images = images.cuda(non_blocking=True)
        soft_label = soft_label.cuda().float()        
        
        if isinstance(mix_index, torch.Tensor):
            mix_index = mix_index.cuda(non_blocking=True)

        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)
        

        # assert torch.allclose(
        #     soft_label, real_soft_label,
        #     rtol=1e-4, atol=5e-4
        # ), "Strict mismatch between pre-generated and teacher soft labels!"

            
        optimizer.zero_grad()


        output = model(images)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        output = F.log_softmax(output / 20, dim=1)
        loss = loss_function_kl(output, soft_label)

        loss.backward()

        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        optimizer.step()

    # wandb 统计保持不变
    metrics = {
        "train/loss": objs.avg,
        "train/Top1": top1.avg,
        "train/Top5": top5.avg,
        "train/lr": scheduler.get_last_lr()[0],
        "train/epoch": epoch,
    }
    wandb_metrics.update(metrics)
    
    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    print(printInfo)
    t1 = time.time()



def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1  = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
        'val/epoch': epoch,
    }
    wandb_metrics.update(metrics)

    return top1.avg


def save_checkpoint(state, is_best, output_dir=None,epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)


if __name__ == "__main__":
    seed_everything(42)
    main()
    wandb.finish()
