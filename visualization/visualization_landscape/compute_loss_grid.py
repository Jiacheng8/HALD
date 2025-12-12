import argparse
import numpy as np
import os
import torch
from landscape_utils import get_data_loader, compute_loss_grid

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--ref", type=str, required=True)
parser.add_argument("--trainset", type=str, required=True)
parser.add_argument("--testset", type=str, required=True)
parser.add_argument("--prefix", type=str, required=True)  # 保存文件前缀
parser.add_argument("--resolution", type=int, default=21)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = get_data_loader(args.trainset)
val_loader = get_data_loader(args.testset)

train_loss, test_loss = compute_loss_grid(
    model_path=args.model,
    reference_path=args.ref,
    train_loader=train_loader,
    val_loader=val_loader,
    resolution=args.resolution,
    device=device
)

os.makedirs("loss_data", exist_ok=True)
np.save(f"loss_data/{args.prefix}_train.npy", train_loss)
np.save(f"loss_data/{args.prefix}_test.npy", test_loss)
print(f"✅ Saved: loss_data/{args.prefix}_train.npy, {args.prefix}_test.npy")
