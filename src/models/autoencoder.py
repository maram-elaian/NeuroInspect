import copy
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, random_split


def _down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))


def _up(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))


class ConvAE(nn.Module):
    """224x224 -> bottleneck (latent_ch x 7 x 7) -> 224x224."""

    def __init__(self, latent_ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            _down(3, 32), _down(32, 64), _down(64, 128), _down(128, 256), _down(256, 256),
            nn.Conv2d(256, latent_ch, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, 256, 1),
            _up(256, 256), _up(256, 128), _up(128, 64), _up(64, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae(train_ds, device, epochs=40, bs=32, lr=1e-3, val_frac=0.1, seed=0):
    """يرجّع (model بأفضل epoch حسب val loss، best_val_loss)."""
    g = torch.Generator().manual_seed(seed)
    n_val = max(1, int(len(train_ds) * val_frac))
    tr, va = random_split(train_ds, [len(train_ds) - n_val, n_val], generator=g)
    tl = DataLoader(tr, bs, shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(va, bs, num_workers=2)

    model = ConvAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for b in tl:
            x = b["image"].to(device)
            loss = ((model(x) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for b in vl:
                x = b["image"].to(device)
                tot += ((model(x) - x) ** 2).mean().item() * len(x)
        v = tot / len(va)
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict_maps(model, ds, device, bs=32, sigma=4):
    """يرجّع (maps[N,H,W], masks[N,1,H,W], labels[N])."""
    model.eval()
    maps, masks, labels = [], [], []
    for b in DataLoader(ds, bs, num_workers=2):
        x = b["image"].to(device)
        err = ((model(x) - x) ** 2).mean(1).cpu().numpy()   # (B,H,W)
        maps.append(err)
        masks.append(b["mask"].numpy())
        labels.append(b["label"].numpy())
    maps = np.concatenate(maps)
    maps = np.stack([gaussian_filter(m, sigma=sigma) for m in maps])
    return maps, np.concatenate(masks), np.concatenate(labels)
