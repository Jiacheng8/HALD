import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import gaussian_filter1d

def parse_args():
    parser = argparse.ArgumentParser("Visualize Gradient Cosine Similarity (Two Runs, Smoothed Only)")
    parser.add_argument('--npy-path1', type=str, default='./data/grad_similarity_fadrm.npy')
    parser.add_argument('--npy-path2', type=str, default='./data/grad_similarity_rded.npy')
    parser.add_argument('--smooth', type=int, default=70)
    parser.add_argument('--save', type=str, default='./grad_sim_compare.pdf')
    parser.add_argument('--ymin', type=float, default=0.1)
    parser.add_argument('--ymax', type=float, default=0.5)
    parser.add_argument('--font', type=str, default='Times New Roman', help="Matplotlib font family")
    parser.add_argument('--color1', type=str, default='#1f77b4', help="Color for Run1 curve")
    parser.add_argument('--color2', type=str, default='#ff7f0e', help="Color for Run2 curve")
    return parser.parse_args()

args = parse_args()

# Set font
plt.rcParams['font.family'] = args.font

# Load data
cos_sims1 = np.load(args.npy_path1)
cos_sims2 = np.load(args.npy_path2)

# Starting index (only plot values after index 5000)
start_idx = 0
cos_sims1 = cos_sims1[start_idx:]
cos_sims2 = cos_sims2[start_idx:]

# Smoothing function (moving average)
def smooth_curve(values, window):
    if window <= 1:
        return values
    smoothed = np.convolve(values, np.ones(window) / window, mode='valid')
    return smoothed

# Apply Gaussian smoothing
cos_sims1_smooth = gaussian_filter1d(cos_sims1, sigma=args.smooth)
cos_sims2_smooth = gaussian_filter1d(cos_sims2, sigma=args.smooth)

# x starts from 0
x1 = np.arange(len(cos_sims1_smooth))
x2 = np.arange(len(cos_sims2_smooth))

# Visualization
plt.figure(figsize=(8, 6))

ax = plt.gca()  # Get current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.plot(x1, cos_sims1_smooth, label='Optimization Based Dataset', color=args.color1, linestyle='-', linewidth=4.3)
plt.plot(x2, cos_sims2_smooth, label='Crops from Real Dataset', color=args.color2, linestyle='-', linewidth=4.3)

# Add dashed line at y = 0 (disabled)
# plt.axhline(y=0, color='red', linestyle='--', linewidth=2.0, alpha=0.8)

plt.tick_params(axis='both', which='both', direction='in', length=6, width=0.5)

plt.grid(True, axis='both', linestyle='--', alpha=0.6)

plt.ylim(args.ymin, args.ymax)   # Set y-axis range

# Custom x ticks (6 evenly spaced)
x_ticks = np.linspace(0, max(len(x1), len(x2)), 6)
plt.xticks(x_ticks, fontsize=23)

# Custom y ticks (5 evenly spaced)
yticks = np.linspace(args.ymin, args.ymax, 5)
plt.yticks(yticks, fontsize=23)

plt.xlim(-100, max(len(x1), len(x2)))  # x starts from 0

plt.legend(fontsize=25)

plt.tight_layout()
plt.savefig(args.save, dpi=300)
print(f"✅ Plot saved to: {args.save}")