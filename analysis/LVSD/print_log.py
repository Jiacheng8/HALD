import numpy as np

# 加载结果文件
data = np.load('./lvsd_imagenet_resnet18.npz')

# 查看有哪些键
print("Keys in file:", data.files)

# 分别打印内容
print("Mean Tr(Σ)_weak:", data['tr_weak'].mean())
print("Mean Tr(Σ)_strong:", data['tr_strong'].mean())
ratio = data['ratio']
log_ratio = np.log10(ratio + 1e-12)  # 避免 log(0)
print("Mean log10(R):", log_ratio.mean())
print("Median log10(R):", np.median(log_ratio))
print("Pct(ratio>1):", (ratio > 1).mean() * 100, "%")
