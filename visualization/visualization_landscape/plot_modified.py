import numpy as np
import matplotlib.pyplot as plt

def estimate_sharpness(Z):
    """
    基于二维 finite difference 的 Hessian 近似，返回 sharpness (最大特征值)。
    Z: [res, res] loss surface
    """
    res = Z.shape[0]
    sharpness = np.zeros_like(Z)

    for i in range(1, res-1):
        for j in range(1, res-1):
            f_xx = Z[i+1, j] - 2*Z[i, j] + Z[i-1, j]
            f_yy = Z[i, j+1] - 2*Z[i, j] + Z[i, j-1]
            f_xy = (Z[i+1, j+1] - Z[i+1, j-1] - Z[i-1, j+1] + Z[i-1, j-1]) / 4.0
            H = np.array([[f_xx, f_xy],
                          [f_xy, f_yy]])
            eigvals = np.linalg.eigvalsh(H)
            sharpness[i, j] = np.max(eigvals)

    return sharpness


def plot_sharpness_contours(trainA, trainB, testA, testB,
                            filename="landscape_sharpness_only.png",
                            levels=20):
    """
    画 2×2 sharpness contour 图 (TrainA, TrainB, TestA, TestB)。
    """
    assert trainA.shape == trainB.shape == testA.shape == testB.shape
    res = trainA.shape[0]
    assert trainA.shape[1] == res, "Grids must be square"

    alpha = np.linspace(-1.0, 1.0, res)
    beta  = np.linspace(-1.0, 1.0, res)
    ALPHA, BETA = np.meshgrid(alpha, beta, indexing='ij')

    # === 计算 sharpness ===
    sharp_trainA = estimate_sharpness(trainA)
    sharp_trainB = estimate_sharpness(trainB)
    sharp_testA  = estimate_sharpness(testA)
    sharp_testB  = estimate_sharpness(testB)

    def get_levels(Z, base_levels=20):
        lo, hi = np.nanmin(Z), np.nanmax(Z)
        if hi <= lo: hi = lo + 1.0
        return np.linspace(lo, hi, base_levels)

    levels_trainA = get_levels(sharp_trainA, base_levels=levels)
    levels_trainB = get_levels(sharp_trainB, base_levels=levels)
    levels_testA  = get_levels(sharp_testA,  base_levels=levels)
    levels_testB  = get_levels(sharp_testB,  base_levels=levels)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    # ---- A模型 - Train ----
    cf0 = axs[0, 0].contourf(ALPHA, BETA, sharp_trainA, levels=levels_trainA, cmap='Reds')
    axs[0, 0].set_title("SHS - Train (Sharpness)")
    fig.colorbar(cf0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    # ---- B模型 - Train ----
    cf1 = axs[0, 1].contourf(ALPHA, BETA, sharp_trainB, levels=levels_trainB, cmap='Reds')
    axs[0, 1].set_title("Soft Only - Train (Sharpness)")
    fig.colorbar(cf1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    # ---- A模型 - Test ----
    cf2 = axs[1, 0].contourf(ALPHA, BETA, sharp_testA, levels=levels_testA, cmap='Reds')
    axs[1, 0].set_title("SHS - Test (Sharpness)")
    fig.colorbar(cf2, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # ---- B模型 - Test ----
    cf3 = axs[1, 1].contourf(ALPHA, BETA, sharp_testB, levels=levels_testB, cmap='Reds')
    axs[1, 1].set_title("Soft Only - Test (Sharpness)")
    fig.colorbar(cf3, ax=axs[1, 1], fraction=0.046, pad=0.04)

    # 通用坐标设置
    for ax in axs.flat:
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Beta")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Saved 2×2 sharpness contour plot: {filename}")


if __name__ == "__main__":
    trainA = np.load("./loss_data/SHS_rded_train.npy")
    testA  = np.load("./loss_data/SHS_rded_test.npy")
    trainB = np.load("./loss_data/Soft_rded_train.npy")
    testB  = np.load("./loss_data/Soft_rded_test.npy")

    plot_sharpness_contours(trainA, trainB, testA, testB,
                            filename="landscape_sharpness_only.png")
