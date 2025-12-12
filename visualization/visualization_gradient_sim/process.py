import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import argparse
import os
import math
import numpy as np
from timm.data import Mixup

# ---------------------- Arguments ----------------------
def parse_args():
    parser = argparse.ArgumentParser("Train & Record Gradient Similarity")
    parser.add_argument('--adamw-lr', type=float, default=0.001)
    parser.add_argument('--eta', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--train-dir', type=str, default='')
    parser.add_argument('--save-path', type=str, default='')
    return parser.parse_args()

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Dataset ----------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder(root=args.train_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

# ---------------------- Model ----------------------
model = models.resnet18(pretrained=False).to(device)
teacher = models.resnet18(pretrained=True).to(device)
model.train()
teacher.train()

criterion_kl = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.AdamW(model.parameters(), lr=args.adamw_lr, weight_decay=1e-4)

# Cosine-style annealing LR scheduler
scheduler = LambdaLR(optimizer, lambda step: 0.5 * (1. + math.cos(math.pi * step / args.epoch / args.eta)))

mixup_fn = Mixup(
    mixup_alpha=0.0,
    cutmix_alpha=1.0,
    label_smoothing=0.8,
    num_classes=1000
)

def soft_cross_entropy(logits, soft_targets):
    """Soft cross-entropy used for KL-based soft label training."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()

def get_flattened_grad(model):
    """Extract all gradients from model parameters as a single flattened vector."""
    return torch.cat([
        p.grad.detach().clone().view(-1) for p in model.parameters() if p.grad is not None
    ])

# ---------------------- Training ----------------------
all_cos_sims = []  # Store cosine similarity for each batch

for epoch in range(args.epoch):
    print(f"🚀 Epoch {epoch+1}/{args.epoch}")
    for batch_idx, (inputs, labels) in enumerate(train_loader):

        # Ensure even batch size for mixup pairing
        if len(inputs) % 2 != 0:
            inputs, labels = inputs[:-1], labels[:-1]

        # Apply mixup
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = mixup_fn(inputs, labels)

        # ========== KL Gradient ==========
        model.zero_grad()
        outputs_kl = F.log_softmax(model(inputs)/20, dim=1)
        soft_targets = F.softmax(teacher(inputs)/20, dim=1)

        loss_kl = criterion_kl(outputs_kl, soft_targets)
        loss_kl.backward(retain_graph=True)

        grad_kl = get_flattened_grad(model)

        # ========== CE Gradient ==========
        model.zero_grad()
        outputs_ce = model(inputs)
        loss_ce = soft_cross_entropy(outputs_ce, labels)
        loss_ce.backward(retain_graph=True)

        grad_ce = get_flattened_grad(model)

        # ========== Cosine Similarity ==========
        cos_sim = F.cosine_similarity(grad_kl.unsqueeze(0), grad_ce.unsqueeze(0)).item()
        all_cos_sims.append(cos_sim)

        # ========== Actual Training Step ==========
        model.zero_grad()
        outputs = model(inputs)
        loss_total = criterion_kl(F.log_softmax(outputs/20, dim=1), soft_targets)
        loss_total.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f"[Epoch {epoch+1} | Batch {batch_idx+1}] CosSim: {cos_sim:.4f}")

    scheduler.step()

# ---------------------- Save Results ----------------------
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
np.save(args.save_path, np.array(all_cos_sims))
print(f"✅ Cosine similarities saved to: {args.save_path}")
