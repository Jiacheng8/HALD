#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import os
from torchvision.models import ResNet18_Weights
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Reuse your provided loader
# -----------------------------

# ③ 在 load_model 前面加一个专门加载官方 resnet18 的小函数
def load_torchvision_resnet18(num_classes: int, device: str):
    # 官方 ImageNet 预训练（1k 类别）；DEFAULT 等价于最新推荐版本
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # 如果你评估的数据类别数不是 1000，就把最后一层替换掉
    if num_classes != 1000 and hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, num_classes)

    model.to(device)
    model.eval()
    try:
        model.to(memory_format=torch.channels_last)
    except Exception:
        pass
    return model

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
    # Optional: channels_last for throughput
    try:
        model.to(memory_format=torch.channels_last)
    except Exception:
        pass
    return model

# -----------------------------
# Metrics
# -----------------------------
def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    p, q: [B, C] probability distributions (rows sum to 1).
    returns: [B] JS divergence per sample (natural log base).
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    return 0.5 * (torch.sum(p * (p.log() - m.log()), dim=1)) + \
           0.5 * (torch.sum(q * (q.log() - m.log()), dim=1))

def cosine_similarity_batch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Cosine similarity per sample, p,q: [B, C]"""
    p_norm = F.normalize(p, dim=1)
    q_norm = F.normalize(q, dim=1)
    return torch.sum(p_norm * q_norm, dim=1)

# -----------------------------
# Image/crop pipeline
# -----------------------------
def build_crop_transform(size: int, mean, std, scale=(0.5, 1.0)):
    return transforms.Compose([
        transforms.RandomResizedCrop(size,),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def list_images(root: Path):
    """Return list of (image_path, class_index) following ImageFolder layout, but we only need paths."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for cls in sorted(os.listdir(root)):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        for f in os.listdir(cls_dir):
            p = cls_dir / f
            if p.suffix.lower() in exts:
                paths.append(p)
    return paths

# -----------------------------
# Main
# -----------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Average Prediction Similarity vs Reference")
    parser.add_argument("--arch", type=str, required=True, help="torchvision model name (e.g., resnet18)")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ref-ckpt", type=Path, required=False)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="root of ImageFolder-style dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-crops", type=int, default=10, help="crops per image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-min", type=float, default=0.5, help="RandomResizedCrop scale min")
    parser.add_argument("--scale-max", type=float, default=1.0, help="RandomResizedCrop scale max")
    # ① argparse 里新增一个布尔开关
    parser.add_argument(
        "--ref-torchvision",
        action="store_true",
        help="Use torchvision official pretrained resnet18 as the reference model."
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load models
    # ④ 在 main() 里加载 reference model 的地方改成：
    if args.ref_torchvision:
        ref_model = load_torchvision_resnet18(args.num_classes, device=args.device)
    else:
        if args.ref_ckpt is None:
            raise ValueError("Either pass --ref-torchvision or provide --ref-ckpt.")
        ref_model = load_model(args.arch, args.num_classes, args.ref_ckpt, device=args.device)
    stu_model = load_model(args.arch, args.num_classes, args.student_ckpt, device=args.device)

    # Transforms (ImageNet normalization)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    crop_tf = build_crop_transform(args.input_size, mean, std, scale=(args.scale_min, args.scale_max))

    # Enumerate images
    images = list_images(args.data_dir)
    if len(images) == 0:
        raise RuntimeError(f"No images found under {args.data_dir}. Expect ImageFolder layout.")

    total_samples = 0
    js_sum = 0.0
    cos_sum = 0.0

    # For reproducible crops per image, set a per-image generator
    g_cpu = torch.Generator(device="cpu")

    for idx, img_path in enumerate(images):
        # Fix seed per image to make crops reproducible
        g_cpu.manual_seed(args.seed + idx)

        # Build K crops for this image into a batch
        crops = []
        img = Image.open(img_path).convert("RGB")
        for _ in range(args.num_crops):
            # Use functional transform with fixed RNG by temporarily setting torch's RNG state on CPU
            # (RandomResizedCrop uses torch.rand under the hood; we can set the global seed before each call)
            torch.manual_seed(int(g_cpu.initial_seed()) + _)
            crops.append(crop_tf(img))
        batch = torch.stack(crops, dim=0).to(args.device, memory_format=torch.channels_last)

        # Forward both models
        ref_logits = ref_model(batch)
        stu_logits = stu_model(batch)
        T = 0.5  # 或者 0.1，越小越尖锐
        ref_probs = (ref_logits / T).softmax(dim=1)
        stu_probs = (stu_logits / T).softmax(dim=1)

        # Metrics per crop
        js_vals = js_divergence(stu_probs, ref_probs)           # [K]
        cos_vals = cosine_similarity_batch(stu_probs, ref_probs) # [K]

        # Average over crops of this image, then accumulate
        js_img = js_vals.mean().item()
        cos_img = cos_vals.mean().item()

        js_sum  += js_img
        cos_sum += cos_img
        total_samples += 1

        if (idx + 1) % 100 == 0:
            print(f"[{idx+1}/{len(images)}] running means -> JS: {js_sum/total_samples:.6f}, Cos: {cos_sum/total_samples:.6f}")

    js_mean  = js_sum / total_samples
    cos_mean = cos_sum / total_samples

    print("\n=== Average Prediction Similarity with Reference Model ===")
    print(f"Images evaluated     : {total_samples}")
    print(f"Crops per image      : {args.num_crops}")
    print(f"JS Divergence (↓)    : {js_mean:.6f}")
    print(f"Cosine Similarity (↑): {cos_mean:.6f}")

if __name__ == "__main__":
    main()
