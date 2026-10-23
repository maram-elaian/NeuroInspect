import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.mvtec import IMAGENET_MEAN, IMAGENET_STD


def denorm(t):
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]
    return np.clip(t.numpy() * std + mean, 0, 1).transpose(1, 2, 0)


def save_heatmaps(ds, maps, masks, labels, out_path, n=4):
    """صورة | ground truth | heatmap فوق الصورة، لـ n صور معيبة موزّعة بالتساوي."""
    idx = np.where(labels == 1)[0]
    idx = idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
    vmin, vmax = maps.min(), maps.max()
    fig, ax = plt.subplots(len(idx), 3, figsize=(9, 3 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        img = denorm(ds[i]["image"])
        ax[r, 0].imshow(img)
        ax[r, 1].imshow(masks[i, 0], cmap="gray")
        ax[r, 2].imshow(img)
        ax[r, 2].imshow(maps[i], cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
        if r == 0:
            ax[r, 0].set_title("image"); ax[r, 1].set_title("ground truth"); ax[r, 2].set_title("anomaly map")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
