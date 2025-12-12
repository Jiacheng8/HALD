import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_data_loader(data_path, batch_size=256):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


def load_model(path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 1000)
    state_dict = torch.load(path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    return model


def get_parameters(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_parameters(model, vec):
    pointer = 0
    for p in model.parameters():
        num_param = p.numel()
        p.data.copy_(vec[pointer:pointer+num_param].view_as(p))
        pointer += num_param


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / total_samples


import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

# 你若已有自定义 load_model，可用你的；这里给一个鲁棒加载示例（以 resnet18 为例）
def _load_resnet18_from_ckpt(ckpt_path: str, device):
    import torchvision.models as models
    m = models.resnet18(pretrained=False)
    m.fc = nn.Linear(512, 1000)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    # 兼容可能带"module."前缀
    new_sd = {}
    for k, v in sd.items():
        new_sd[k.replace("module.", "")] = v
    missing, unexpected = m.load_state_dict(new_sd, strict=False)
    if len(unexpected) > 0:
        # 忽略优化器等杂项
        pass
    m.to(device)
    return m

def _flatten_params(named_params):
    # 返回扁平向量 & 每个张量的 view 信息
    flats = []
    shapes = []
    keys = []
    for k, p in named_params:
        flats.append(p.data.view(-1))
        shapes.append(p.data.shape)
        keys.append(k)
    return torch.cat(flats), shapes, keys

def _unflatten_to_model(vec, model, shapes, keys):
    # 将 vec 拆回到 model 的参数里
    offset = 0
    with torch.no_grad():
        for (name, p), shape, k in zip(model.named_parameters(), shapes, keys):
            n = int(np.prod(shape))
            chunk = vec[offset: offset + n].view(shape).to(p.device, dtype=p.dtype)
            p.copy_(chunk)
            offset += n
    assert offset == vec.numel()

def _state_to_vec(model):
    return torch.nn.utils.parameters_to_vector([p.data for p in model.parameters()])

def _vec_to_state(vec, model):
    torch.nn.utils.vector_to_parameters(vec, [p for p in model.parameters()])

def _filterwise_norm_like(param_tensor, dir_tensor, is_weight=True):
    """
    对方向 dir_tensor 做“以输出通道为单位”的 L2 归一：
    - Conv weight: (out_c, in_c, kH, kW) -> 按 out_c 维度逐滤波器归一
    - Linear weight: (out_f, in_f) -> 按 out_f 逐行归一
    - bias / 非权重: 返回全零（不沿 bias 方向走，避免不稳）
    """
    if not is_weight:
        return torch.zeros_like(dir_tensor)

    if dir_tensor.ndim == 4:  # Conv
        oc = dir_tensor.shape[0]
        d = dir_tensor.view(oc, -1)
        n = d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        d = (d / n).view_as(dir_tensor)
        return d
    elif dir_tensor.ndim == 2:  # Linear
        d = dir_tensor
        n = d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        d = d / n
        return d
    else:
        # 其它形状（如 running stats）或 scalar/bias：不沿其方向前进
        return torch.zeros_like(dir_tensor)

def _make_uv_directions(model, ref, device):
    with torch.no_grad():
        u_tensors = []
        v_tensors = []
        for (n_m, p_m), (n_r, p_r) in zip(model.named_parameters(), ref.named_parameters()):
            assert n_m == n_r
            is_weight = ('weight' in n_m) and (p_m.ndim in (2, 4))

            u_raw = p_m.data - p_r.data   # 原始差向量
            v_raw = torch.randn_like(p_m.data)

            u_fw = _filterwise_norm_like(p_m.data, u_raw, is_weight=is_weight)
            v_fw = _filterwise_norm_like(p_m.data, v_raw, is_weight=is_weight)

            u_tensors.append(u_fw)
            v_tensors.append(v_fw)

        u = torch.cat([t.reshape(-1) for t in u_tensors]).to(device)
        v = torch.cat([t.reshape(-1) for t in v_tensors]).to(device)

        # ⚠️ 保留 u_raw 的整体 norm
        u_raw_vec = torch.cat([ (p_m.data - p_r.data).reshape(-1) for (p_m, p_r) in 
                                zip(model.parameters(), ref.parameters()) ]).to(device)

        # Gram-Schmidt
        u = u / (u.norm() + 1e-12)
        v = v - (u @ v) * u
        v = v / (v.norm() + 1e-12)

    return u, v, u_raw_vec.norm()


@torch.no_grad()
def _eval_avg_loss(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss, total_cnt = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        total_cnt += targets.numel()
    return total_loss / max(total_cnt, 1)


def compute_loss_grid(model_path: str,
                      reference_path: str,
                      train_loader,
                      val_loader,
                      resolution: int = 21,
                      device: torch.device = torch.device("cpu")):
    """
    修正版：使用原始 θ_model - θ_ref 的范数 unorm 作为扰动缩放系数。
    """
    # 1) 加载模型和参考模型
    model = _load_resnet18_from_ckpt(model_path, device)
    ref   = _load_resnet18_from_ckpt(reference_path, device)

    # 2) 构造方向向量 u, v，并保留原始差值范数 unorm
    with torch.no_grad():
        u_tensors = []
        v_tensors = []
        u_raw_list = []
        for (n_m, p_m), (n_r, p_r) in zip(model.named_parameters(), ref.named_parameters()):
            assert n_m == n_r
            is_weight = ('weight' in n_m) and (p_m.ndim in (2, 4))
            u_raw = p_m.data - p_r.data
            v_raw = torch.randn_like(p_m.data)

            u_fw = _filterwise_norm_like(p_m.data, u_raw, is_weight=is_weight)
            v_fw = _filterwise_norm_like(p_m.data, v_raw, is_weight=is_weight)

            u_tensors.append(u_fw)
            v_tensors.append(v_fw)
            u_raw_list.append(u_raw.view(-1))

        u = torch.cat([t.view(-1) for t in u_tensors]).to(device)
        v = torch.cat([t.view(-1) for t in v_tensors]).to(device)
        u_raw_vec = torch.cat(u_raw_list).to(device)
        unorm = u_raw_vec.norm().item()

        # 正交化
        u = u / (u.norm() + 1e-12)
        v = v - (u @ v) * u
        v = v / (v.norm() + 1e-12)

    print(f"✅ u_raw norm (unorm) = {unorm:.4f}")
    print(f"✅ u norm: {u.norm():.4f}, v norm: {v.norm():.4f}, cosine = {(u @ v).item():.4f}")

    # 3) 获取模型初始参数向量
    theta0 = _state_to_vec(model).to(device)

    # 4) 网格设置
    alphas = np.linspace(-1.0, 1.0, resolution)
    betas  = np.linspace(-1.0, 1.0, resolution)
    scale  = 0.1  # 乘 unorm 后成为真实扰动强度

    train_Z = np.zeros((resolution, resolution), dtype=np.float64)
    test_Z  = np.zeros((resolution, resolution), dtype=np.float64)

    # 5) 设置 eval + 冻结 BN 统计
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = True
            m.momentum = 0.0

    # 6) 遍历 grid 并评估 loss
    from tqdm import tqdm
    for i, a in enumerate(tqdm(alphas, desc="Alpha Loop", position=0)):
        for j, b in enumerate(tqdm(betas, desc=f"Beta Loop α={a:.2f}", leave=False, position=1)):
            delta = scale * unorm * (a * u + b * v)
            theta_prime = theta0 + delta
            _vec_to_state(theta_prime, model)

            tloss = _eval_avg_loss(model, train_loader, device)
            vloss = _eval_avg_loss(model, val_loader, device)

            train_Z[i, j] = tloss
            test_Z[i, j]  = vloss

            tqdm.write(f"[α={a:.2f}, β={b:.2f}] Δθ norm = {delta.norm().item():.4f} | train={tloss:.2f}, test={vloss:.2f}")

    return train_Z, test_Z

def plot_contour_comparison(Z1, Z2, title, filename, levels=30):
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))
    ALPHA, BETA = np.meshgrid(np.linspace(-1, 1, Z1.shape[0]),
                              np.linspace(-1, 1, Z1.shape[1]))

    axs[0].contour(ALPHA, BETA, Z1.T, levels=levels, colors='green', linewidths=1.2)
    axs[0].contour(ALPHA, BETA, Z2.T, levels=levels, colors='blue', linewidths=1.2)
    axs[0].set_title(f"{title} (train)")
    axs[0].legend(['w/ smoothing', 'w/o smoothing'])

    axs[1].contour(ALPHA, BETA, Z1.T, levels=levels, colors='green', linewidths=1.2)
    axs[1].contour(ALPHA, BETA, Z2.T, levels=levels, colors='blue', linewidths=1.2)
    axs[1].set_title(f"{title} (test)")

    plt.tight_layout()
    plt.savefig(filename + "_comparison.png")
    plt.close()
    print(f"✅ Saved: {filename}_comparison.png")
