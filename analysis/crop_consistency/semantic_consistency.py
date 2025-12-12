#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from collections import defaultdict
from contextlib import nullcontext

# -----------------------------
# Utilities
# -----------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transform(crop_size: int, normalize: bool=True) -> T.Compose:
    tfms = [
        T.RandomResizedCrop(size=crop_size),
        T.ToTensor(),
    ]
    if normalize:
        tfms.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(tfms)

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    # p, q: (..., C) probabilities (sum to 1, >=0)
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)

def cosine_similarity(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # p, q: (..., C)
    p_norm = F.normalize(p, dim=-1)
    q_norm = F.normalize(q, dim=-1)
    return torch.sum(p_norm * q_norm, dim=-1)

def load_model(arch: str, num_classes: int, checkpoint_path: Path, device: str):
    # Build torchvision model by name and load state dict from checkpoint
    if not hasattr(models, arch):
        raise ValueError(f"Unknown torchvision model arch '{arch}'.")

    model_ctor = getattr(models, arch)
    # Try to construct with num_classes; fallback to replace head
    try:
        model = model_ctor(num_classes=num_classes)
    except TypeError:
        model = model_ctor()
        if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
            in_f = model.fc.in_features
            model.fc = torch.nn.Linear(in_f, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, torch.nn.Sequential):
                last_idx = len(model.classifier) - 1
                if isinstance(model.classifier[last_idx], torch.nn.Linear):
                    in_f = model.classifier[last_idx].in_features
                    model.classifier[last_idx] = torch.nn.Linear(in_f, num_classes)
            elif isinstance(model.classifier, torch.nn.Linear):
                in_f = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_f, num_classes)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)  # support raw state_dict
    # Strip DataParallel "module." if present
    new_sd = { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.to(device)
    model.eval()
    # 可选：channels_last，提升吞吐
    try:
        model.to(memory_format=torch.channels_last)
    except Exception:
        pass
    return model

# -----------------------------
# Data pipeline: Two-crop collate
# -----------------------------

# Same images two independent crops
class TwoCropsCollate:
    """Given a batch of (PIL, class_idx), return:
       x: Tensor [2B, C, H, W] (two independent RandomResizedCrop for each image)
       y: Tensor [B] (class indices)
    """
    def __init__(self, crop_tfms: T.Compose):
        self.crop_tfms = crop_tfms

    def __call__(self, batch):
        imgs1, imgs2, labels = [], [], []
        for img, y in batch:
            imgs1.append(self.crop_tfms(img))
            imgs2.append(self.crop_tfms(img))
            labels.append(y)
        x1 = torch.stack(imgs1, dim=0)
        x2 = torch.stack(imgs2, dim=0)
        x  = torch.cat([x1, x2], dim=0)  # [2B, C, H, W]
        y  = torch.tensor(labels, dtype=torch.long)
        return x, y

# Different images two independent crops
# class TwoCropsCollate:
#     """Return random crops from **different images** in the same batch.
#        x: [2B, C, H, W], y: [B] (labels of first half)
#     """
#     def __init__(self, crop_tfms: T.Compose):
#         self.crop_tfms = crop_tfms

#     def __call__(self, batch):
#         imgs, labels = zip(*batch)
#         imgs = list(imgs)
#         labels = list(labels)
#         B = len(imgs)
#         # 打乱索引形成不同图片的配对
#         perm = torch.randperm(B)
#         imgs1 = [self.crop_tfms(imgs[i]) for i in range(B)]
#         imgs2 = [self.crop_tfms(imgs[perm[i]]) for i in range(B)]
#         x1 = torch.stack(imgs1, dim=0)
#         x2 = torch.stack(imgs2, dim=0)
#         x = torch.cat([x1, x2], dim=0)  # [2B, C, H, W]
#         y = torch.tensor(labels, dtype=torch.long)
#         return x, y


def build_loader(data_root: Path, crop_tfms: T.Compose,
                 batch_size: int=512, workers: int=12, pin_memory: bool=True):
    ds = ImageFolder(root=str(data_root))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=TwoCropsCollate(crop_tfms),
        drop_last=False,
        persistent_workers=(workers > 0),
    )
    return ds, loader

# -----------------------------
# Fast evaluation (global loader, early stop)
# -----------------------------

from contextlib import nullcontext
from tqdm import tqdm
import torch
import torch.nn.functional as F

@torch.no_grad()
def eval_consistency_fast(
    model,
    device: str,
    ds,                    # torchvision.datasets.ImageFolder
    loader,                # DataLoader that yields (x:[2B,C,H,W], y:[B])
    n_pairs_per_class: int,
    amp: bool = True,
    max_epochs: int = 50,  # 安全上限，避免极端长循环
):
    """
    计算每个类别的 crop-level 语义一致性（JS 越低越好 / Cosine 越高越好），
    直到每个类累计到 n_pairs_per_class 对随机裁剪（或达到 max_epochs）。

    假设 DataLoader 使用 TwoCropsCollate：
      - 输入 x 的形状为 [2B, C, H, W]，前 B/后 B 为两次独立裁剪
      - y 的形状为 [B]，为该批原图所属类别索引
    """

    num_classes = len(ds.classes)
    js_sum = [0.0] * num_classes
    cs_sum = [0.0] * num_classes
    cnt    = [0]   * num_classes

    use_amp = (amp and str(device).startswith("cuda"))
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    # 反复跑多轮 DataLoader，直到每个类都达到 n_pairs_per_class
    for ep in range(max_epochs):
        pbar = tqdm(loader, desc=f"Batches (epoch {ep+1})")
        finished = True  # 假设这一轮结束后能满足，过程中若发现未达标则置 False

        for x, y in pbar:
            B = y.size(0)
            x = x.to(device, non_blocking=True)   # [2B, C, H, W]
            y = y.to(device, non_blocking=True)   # [B]

            with autocast_ctx():
                logits = model(x)                 # [2B, C]
                probs  = F.softmax(logits, dim=1)

            p1, p2 = probs[:B], probs[B:]         # [B, C], [B, C]

            # --- 指标 ---
            # JS
            eps = 1e-12
            p = torch.clamp(p1, eps, 1.0)
            q = torch.clamp(p2, eps, 1.0)
            m = 0.5 * (p + q)
            js = 0.5 * (torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1) +
                        torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1))  # [B]

            # Cosine
            p1n = F.normalize(p1, dim=-1)
            p2n = F.normalize(p2, dim=-1)
            cs  = torch.sum(p1n * p2n, dim=-1)    # [B]

            # 仅为“未达标”的类继续累积
            any_short = False
            for i in range(B):
                c = int(y[i])
                if cnt[c] < n_pairs_per_class:
                    js_sum[c] += float(js[i])
                    cs_sum[c] += float(cs[i])
                    cnt[c]    += 1
                if cnt[c] < n_pairs_per_class:
                    any_short = True

            finished = (not any_short)
            if finished:
                break  # 本轮已达标，跳出

        if finished:
            break  # 所有类都达标，结束评估

    # 聚合 per-class & overall
    per_class = {}
    js_all, cs_all = [], []
    for c, name in enumerate(ds.classes):
        denom = max(1, cnt[c])  # 防御：某些极端类达不到目标
        js_mean = js_sum[c] / denom
        cs_mean = cs_sum[c] / denom
        per_class[name] = {"js_mean": js_mean, "cos_mean": cs_mean}
        js_all.append(js_mean)
        cs_all.append(cs_mean)

    overall_js  = float(sum(js_all) / len(js_all))
    overall_cos = float(sum(cs_all) / len(cs_all))
    return per_class, overall_js, overall_cos

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute crop-level semantic consistency (fast)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root dir with ImageFolder layout: root/class_x/*.jpg")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (with 'state_dict' or raw state_dict)")
    parser.add_argument("--arch", type=str, default="resnet50",
                        help="Torchvision model name (e.g., resnet50, efficientnet_b0)")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Number of classes of the model head")
    parser.add_argument("--n-pairs-per-class", type=int, default=500,
                        help="Pairs per class; each sample contributes one pair (two crops)")
    parser.add_argument("--crop-size", type=int, default=224,
                        help="Crop output size for RandomResizedCrop")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply ImageNet normalization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="B here means images per batch; model sees 2*B crops")
    parser.add_argument("--amp", action="store_true",
                        help="Enable autocast AMP for faster inference on CUDA")
    parser.add_argument("--output", type=str, default="semantic_consistency_results.json")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True  # 固定尺寸时更快

    data_root = Path(args.data_root)
    device = args.device
    print(f"Using device: {device}")

    model = load_model(args.arch, args.num_classes, Path(args.checkpoint), device=device)

    crop_tfms = build_transform(args.crop_size, normalize=args.normalize)
    ds, loader = build_loader(
        data_root=data_root,
        crop_tfms=crop_tfms,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=True
    )

    per_class, overall_js, overall_cos = eval_consistency_fast(
        model=model,
        device=device,
        ds=ds,
        loader=loader,
        n_pairs_per_class=args.n_pairs_per_class,
        amp=args.amp
    )

    result = {
        "data_root": str(data_root),
        "checkpoint": str(Path(args.checkpoint)),
        "arch": args.arch,
        "num_classes": args.num_classes,
        "n_pairs_per_class": args.n_pairs_per_class,
        "crop_size": args.crop_size,
        "normalize": bool(args.normalize),
        "seed": args.seed,
        "device": device,
        "per_class": per_class,
        "overall": {
            "js_mean": overall_js,
            "cos_mean": overall_cos
        }
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Semantic Consistency (averaged over classes) ===")
    print(f"JS divergence (lower is better): {overall_js:.6f}")
    print(f"Cosine similarity (higher is better): {overall_cos:.6f}")
    print(f"Saved full results to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
