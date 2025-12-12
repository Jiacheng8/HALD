#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def identity_collate(batch: List[Tuple[object, int]]):
    # batch: list of (PIL.Image.Image, label)
    return batch  # keep as-is (avoid default_collate on PIL)


def build_model(name: str, device: str):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif name in ("shufflenet_v2", "shufflenet_v2_x1_0"):
        m = models.shufflenet_v2_x1_0(
            weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    m.eval().to(device)
    return m


# -----------------------------
# LVSD core
# -----------------------------
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]


def make_transforms(use_color_jitter: bool):
    weak_aug = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ]
    )

    strong_list = [
        transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if use_color_jitter:
        strong_list.insert(1, transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
    strong_list += [transforms.ToTensor(), transforms.Normalize(IMNET_MEAN, IMNET_STD)]
    strong_aug = transforms.Compose(strong_list)

    return weak_aug, strong_aug


@torch.no_grad()
def compute_tr_sigma(model, image_pil, aug, device: str, num_samples: int = 8) -> float:
    """Compute Tr(Σ) for a single image by repeated random augmentations."""
    preds = []
    for _ in range(num_samples):
        x = aug(image_pil)  # -> Tensor [C,H,W]
        if x.dim() == 3:
            x = x.unsqueeze(0)  # [1,C,H,W]
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = F.softmax(logits, dim=-1)
        preds.append(p.cpu())
    preds = torch.cat(preds, dim=0)  # [s, C]

    mean = preds.mean(dim=0, keepdim=True)
    cov = (preds - mean).T @ (preds - mean) / (preds.size(0) - 1)
    tr = torch.trace(cov).item()
    return tr


def measure_lvsd(
    model,
    dataset,
    aug,
    device: str,
    num_samples: int,
    num_workers: int,
):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=identity_collate,
        pin_memory=True,
    )
    traces = []
    for batch in tqdm(loader, total=len(dataset)):
        img_pil, _ = batch[0]  # batch size is 1
        tr = compute_tr_sigma(model, img_pil, aug, device, num_samples=num_samples)
        traces.append(tr)
    return np.array(traces, dtype=np.float64)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Measure LVSD via Tr(Σ) on an ImageFolder dataset."
    )
    parser.add_argument("--data-root", type=str, required=True, help="ImageFolder root")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "mobilenet_v2", "shufflenet_v2_x1_0"],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--samples-per-image", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-color-jitter", action="store_true", help="Disable ColorJitter in strong aug")
    parser.add_argument("--save-prefix", type=str, default="lvsd_imagenet")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = args.device
    model = build_model(args.model, device)
    weak_aug, strong_aug = make_transforms(use_color_jitter=not args.no_color_jitter)

    # ImageFolder returns PIL (we keep transform=None)
    full_dataset = datasets.ImageFolder(root=args.data_root, transform=None)
    print(f"[Info] Total images: {len(full_dataset)}")

    # subset for speed
    if args.max_images < len(full_dataset):
        idx = random.sample(range(len(full_dataset)), args.max_images)
        dataset = Subset(full_dataset, idx)
    else:
        dataset = full_dataset

    print("[Info] Measuring LVSD (weak aug) ...")
    tr_weak = measure_lvsd(
        model,
        dataset,
        weak_aug,
        device=device,
        num_samples=args.samples_per_image,
        num_workers=args.num_workers,
    )

    print("[Info] Measuring LVSD (strong aug) ...")
    tr_strong = measure_lvsd(
        model,
        dataset,
        strong_aug,
        device=device,
        num_samples=args.samples_per_image,
        num_workers=args.num_workers,
    )

    ratio = tr_strong / (tr_weak + 1e-8)

    print(f"\n[LVSD Results on {len(ratio)} samples | model={args.model}]")
    print(f"Mean Tr(Σ)_weak   = {tr_weak.mean():.6f}")
    print(f"Mean Tr(Σ)_strong = {tr_strong.mean():.6f}")
    print(f"Median Ratio (strong/weak) = {np.median(ratio):.2f}×")
    print(f"Pct(ratio>1) = {(ratio>1).mean()*100:.1f}%")

    # Save npz
    npz_path = f"{args.save_prefix}_{args.model}.npz"
    np.savez_compressed(npz_path, tr_weak=tr_weak, tr_strong=tr_strong, ratio=ratio)
    print(f"[Saved] Stats to {npz_path}")

    # Plot histogram
    plt.figure(figsize=(7, 4.5))
    plt.hist(np.log10(ratio), bins=40, alpha=0.8)
    plt.axvline(np.log10(np.median(ratio)), linestyle="--", label="Median")
    plt.xlabel("log10(TrΣ_strong / TrΣ_weak)")
    plt.ylabel("Count")
    plt.title(f"LVSD Ratio Distribution ({args.model}, N={len(ratio)})")
    plt.legend()
    plt.tight_layout()
    fig_path = f"{args.save_prefix}_{args.model}_ratio_hist.png"
    plt.savefig(fig_path, dpi=300)
    print(f"[Saved] Figure to {fig_path}")


if __name__ == "__main__":
    main()
