# import numpy as np
# import matplotlib.pyplot as plt


# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

# import numpy as np
# import matplotlib.pyplot as plt


# def plot_train_test_contour_comparison(trainA, trainB, testA, testB, filename="landscape_2x2_fullrange.png", levels=20):
#     """
#     可视化两个模型在 train/test 上的 loss landscape，分别画成 2×2 四张图。
#     每张图只画一个模型在一个 split 上的结果，完整展示 loss 值范围。
#     """
#     assert trainA.shape == trainB.shape == testA.shape == testB.shape, "All input grids must have same shape."
#     res = trainA.shape[0]
#     assert trainA.shape[1] == res, "Grids must be square"

#     alpha = np.linspace(-1.0, 1.0, res)
#     beta  = np.linspace(-1.0, 1.0, res)
#     ALPHA, BETA = np.meshgrid(alpha, beta, indexing='ij')

#     fig, axs = plt.subplots(2, 2, figsize=(15, 9))



#     # 直接使用原始 min/max（无剪裁）
#     def get_levels(Z, base_levels=20):
#         lo = np.nanmin(Z)
#         hi = np.nanmax(Z)
#         if hi <= lo:
#             hi = lo + 1.0
#         return np.linspace(lo, hi, base_levels), lo, hi

#     levels_A_train, loA_train, hiA_train = get_levels(trainA)
#     levels_A_test,  loA_test,  hiA_test  = get_levels(testA)
#     levels_B_train, loB_train, hiB_train = get_levels(trainB)
#     levels_B_test,  loB_test,  hiB_test  = get_levels(testB)

#     # ---- A模型 - Train ----
#     cf0 = axs[0, 0].contourf(ALPHA, BETA, trainA, levels=levels_A_train, cmap='Greens', alpha=0.7)
#     cs0 = axs[0, 0].contour (ALPHA, BETA, trainA, levels=levels_A_train, colors='green', linewidths=1.0)
#     axs[0, 0].set_title("SHS - Train")
#     fig.colorbar(cf0, ax=axs[0, 0], fraction=0.046, pad=0.04)

#     # ---- B模型 - Train ----
#     cf1 = axs[0, 1].contourf(ALPHA, BETA, trainB, levels=levels_B_train, cmap='Blues', alpha=0.7)
#     cs1 = axs[0, 1].contour (ALPHA, BETA, trainB, levels=levels_B_train, colors='blue', linewidths=1.0)
#     axs[0, 1].set_title("Soft Only - Train")
#     fig.colorbar(cf1, ax=axs[0, 1], fraction=0.046, pad=0.04)

#     # ---- A模型 - Test ----
#     cf2 = axs[1, 0].contourf(ALPHA, BETA, testA, levels=levels_A_test, cmap='Greens', alpha=0.7)
#     cs2 = axs[1, 0].contour (ALPHA, BETA, testA, levels=levels_A_test, colors='green', linewidths=1.0)
#     axs[1, 0].set_title("SHS - Test")
#     fig.colorbar(cf2, ax=axs[1, 0], fraction=0.046, pad=0.04)

#     # ---- B模型 - Test ----
#     cf3 = axs[1, 1].contourf(ALPHA, BETA, testB, levels=levels_B_test, cmap='Blues', alpha=0.7)
#     cs3 = axs[1, 1].contour (ALPHA, BETA, testB, levels=levels_B_test, colors='blue', linewidths=1.0)
#     axs[1, 1].set_title("Soft Only - Test")
#     fig.colorbar(cf3, ax=axs[1, 1], fraction=0.046, pad=0.04)

#     # 通用坐标设置
#     for ax in axs.flat:
#         ax.set_xlabel("Alpha")
#         ax.set_ylabel("Beta")

#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)
#     plt.close()
#     print(f"✅ Saved 2×2 full-range contour plot: {filename}")

# import numpy as np

# def check_grid(Z, name="Z"):
#     print(name, "shape:", Z.shape)
#     print("min / max:", np.nanmin(Z), np.nanmax(Z))
#     print("nan count:", np.isnan(Z).sum(), "inf count:", np.isinf(Z).sum())
#     # 百分位，看看是否被极端值主导
#     for p in [0, 10, 25, 50, 75, 90, 99]:
#         print(f"{p}th pct:", np.nanpercentile(Z, p))

# if __name__ == "__main__":
#     # === 路径根据你之前保存的 npy 文件来修改 ===
#     trainA = np.load("./loss_data/ours_fadrm_train.npy")
#     testA  = np.load("./loss_data/ours_fadrm_test.npy")
#     trainB = np.load("./loss_data/lpld_fadrm_train.npy")
#     testB  = np.load("./loss_data/lpld_fadrm_test.npy")

#     # 输出文件名
#     plot_train_test_contour_comparison(trainA, trainB, testA, testB, filename="landscape_combined.png")
#     # check_grid(trainA, name="trainA")
#     # check_grid(testA, name="testA")
#     # check_grid(trainB, name="trainB")
#     # check_grid(testB, name="testB")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'

def plot_train_test_contour_comparison(
    trainA, trainB, testA, testB,
    filename="landscape_2x2_fullrange.png",
    levels=20,
    font_sizes=None,
    cmap_config=None
):
    """
    可视化两个模型在 train/test 上的 loss landscape，分别画成 2×2 四张图。
    """
    assert trainA.shape == trainB.shape == testA.shape == testB.shape, "All input grids must have same shape."
    res = trainA.shape[0]
    assert trainA.shape[1] == res, "Grids must be square"

    # 默认字体大小设置
    if font_sizes is None:
        font_sizes = {
            "title": 18,
            "label": 14,
            "ticks": 12,
            "clabel": 10
        }

    # 默认颜色设置
    if cmap_config is None:
        cmap_config = {
            "trainA": {"cmap": "Greens", "contour_color": "green"},
            "trainB": {"cmap": "Blues", "contour_color": "blue"},
            "testA":  {"cmap": "Greens", "contour_color": "green"},
            "testB":  {"cmap": "Blues", "contour_color": "blue"},
        }

    alpha = np.linspace(-1.0, 1.0, res)
    beta  = np.linspace(-1.0, 1.0, res)
    ALPHA, BETA = np.meshgrid(alpha, beta, indexing='ij')

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    def get_levels(Z, base_levels=20):
        lo = np.nanmin(Z)
        hi = np.nanmax(Z)
        if hi <= lo:
            hi = lo + 1.0
        return np.linspace(lo, hi, base_levels), lo, hi

    levels_A_train, _, _ = get_levels(trainA, levels)
    levels_B_train, _, _ = get_levels(trainB, levels)
    levels_A_test,  _, _ = get_levels(testA,  levels)
    levels_B_test,  _, _ = get_levels(testB,  levels)

    # Plotting Helper
    def plot_single(ax, Z, levels, cmap, contour_color, title):
        # cmap = create_morandi_cmap("morandi_green")
        
        cf = ax.contourf(ALPHA, BETA, Z, levels=levels, cmap=cmap, alpha=0.7)
        cs = ax.contour (ALPHA, BETA, Z, levels=levels, colors=contour_color, linewidths=1.0)
        cl = ax.clabel(cs, inline=True, fontsize=font_sizes["clabel"], fmt="%.2f")  # 等高线数值
        ax.set_title(title, fontsize=font_sizes["title"])
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.tick_params(labelsize=font_sizes["ticks"])
        # fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    plot_single(axs[0, 0], trainA, levels_A_train, cmap_config["trainA"]["cmap"], cmap_config["trainA"]["contour_color"], "Finite Soft Supervision - Train")
    plot_single(axs[0, 1], trainB, levels_B_train, cmap_config["trainB"]["cmap"], cmap_config["trainB"]["contour_color"], "Ours - Train")
    plot_single(axs[1, 0], testA,  levels_A_test,  cmap_config["testA"]["cmap"],  cmap_config["testA"]["contour_color"],  "Finite Soft Supervision - Test")
    plot_single(axs[1, 1], testB,  levels_B_test,  cmap_config["testB"]["cmap"],  cmap_config["testB"]["contour_color"],  "Ours - Test")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Saved 2×2 full-range contour plot: {filename}")
    
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap

def morandi_dual_cmap(low_hex, high_hex, name="morandi_dual"):
    """从低值到高值：灰浅色 → 莫兰迪深色"""
    return LinearSegmentedColormap.from_list(name, [low_hex, high_hex])


# 检查 grid 的辅助函数
def check_grid(Z, name="Z"):
    print(name, "shape:", Z.shape)
    print("min / max:", np.nanmin(Z), np.nanmax(Z))
    print("nan count:", np.isnan(Z).sum(), "inf count:", np.isinf(Z).sum())
    for p in [0, 10, 25, 50, 75, 90, 99]:
        print(f"{p}th pct:", np.nanpercentile(Z, p))


def create_morandi_cmap(name="morandi_pink"):
    color_presets = {
        "morandi_purple": ["#F4F3F3", "#E8DFF5", "#C3BCD9", "#A89EB0"],
        "morandi_blue":   ["#F4F3F3", "#D9E1E8", "#A3BFD9", "#7F97B7"],
        "morandi_green":  ["#F4F3F3", "#D4E2D4", "#B4C9B4", "#90AA90"],
        "morandi_pink":   ["#F4F3F3", "#E8D8D7", "#D5BCBB", "#C4A9A8"],
    }
    colors = color_presets.get(name, color_presets["morandi_purple"])
    return LinearSegmentedColormap.from_list(name, colors)


if __name__ == "__main__":
    trainB = np.load("./loss_data/ours_fadrm_train.npy")
    testB  = np.load("./loss_data/ours_fadrm_test.npy")
    trainA = np.load("./loss_data/lpld_fadrm_train.npy")
    testA  = np.load("./loss_data/lpld_fadrm_test.npy")
    # trainA = np.load("./loss_data/SHS_rded_train.npy")
    # testA  = np.load("./loss_data/SHS_rded_test.npy")
    # trainB = np.load("./loss_data/Soft_rded_train.npy")
    # testB  = np.load("./loss_data/Soft_rded_test.npy")
    # 你可以在这里自定义字体大小和颜色
    font_sizes = {
        "title": 43,
        "label": 23,
        "ticks": 20,
        "clabel": 10
    }

    # 你的图里紫色区块色带：白 → #D7CDE7
    purple_cmap = LinearSegmentedColormap.from_list(
        "purple_fadrm",
        [
            "#FFFFFF",  # 白色起点
            "#EEEAF4",  # 非常浅的灰紫
            "#E3DCF0",
            "#D7CDE7",  # 原图背景紫
            "#C4BAD9",
            "#B3A9CD",
            "#A598C3",  # 深紫
            "#8F7DB5",
            "#7B68A1"   # 接近蓝紫
        ]
    )


    # 绿色 residual 区块色带：白 → #D7E2DC
    green_cmap = LinearSegmentedColormap.from_list("green_fadrm", ["#FFFFFF", "#D7E2DC", "#A0B3AA"])
    
    blue_cmap = LinearSegmentedColormap.from_list(
        "blue_fadrm",
        [
            "#FFFFFF",  # 0. 极浅白
            "#F2F5F9",  # 1. 雾白蓝
            "#E6EEF6",  # 2. 非常浅蓝灰
            "#D9E5F1",  # 3. 浅蓝灰
            "#C7CFE8",  # 4. 浅莫兰迪蓝（你图里原始块）
            "#B0BEDD",  # 5. 中灰蓝
            "#A3BFD9",  # 6. 雾蓝（主色）
            "#8FA9C6",  # 7. 深蓝灰（靠近主线条）
            "#5F6A82"   # 8. 深蓝（主 block 外框）
        ]
    )
    cmap_config = {
        "trainB": {
            "cmap": purple_cmap,              # Ours (SHS) - Training phase, soft Morandi purple gradient
            "contour_color": "#6F6080"        # Deep lilac-gray for clean, high-contrast contours
        },
        "trainA": {
            "cmap": blue_cmap,                # Baseline - Training phase, soft Morandi blue gradient
            "contour_color": "#5F6A82"        # Muted green-gray for smooth, elegant outlines
        },
        "testB": {
            "cmap": purple_cmap,              # Ours (SHS) - Testing phase, same purple palette
            "contour_color": "#6F6080"        # Consistent deep lilac-gray contour lines
        },
        "testA": {
            "cmap": blue_cmap,                # Baseline - Testing phase, same blue palette
            "contour_color": "#5F6A82"        # Consistent muted green-gray contour lines
        }
    }


    plot_train_test_contour_comparison(
        trainA, trainB, testA, testB,
        filename="landscape_combined.pdf",
        font_sizes=font_sizes,
        cmap_config=cmap_config
    )
