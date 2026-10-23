import argparse, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.data.mvtec import MVTecDataset, CATEGORIES
from src.eval.metrics import image_auroc, pixel_auroc, image_scores_from_maps
from src.eval.pro import aupro
from src.eval.viz import save_heatmaps
from src.models.autoencoder import train_ae, predict_maps
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="/kaggle/working/outputs/ae")
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    csv_path = Path(a.out) / f"results_seed{a.seed}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["category"] for r in rows}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for cat in a.categories:
        if cat in done:
            print("skip (already done):", cat)
            continue
        t0 = time.time()
        set_seed(a.seed)
        train_ds = MVTecDataset(a.data_root, cat, "train")
        test_ds = MVTecDataset(a.data_root, cat, "test")
        model, val_loss = train_ae(train_ds, device, epochs=a.epochs, seed=a.seed)
        maps, masks, labels = predict_maps(model, test_ds, device)

        row = dict(
            category=cat,
            image_auroc=image_auroc(labels, image_scores_from_maps(maps)),
            pixel_auroc=pixel_auroc(masks, maps),
            aupro=aupro(masks, maps),
            val_loss=val_loss,
            seconds=round(time.time() - t0),
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)       # حفظ بعد كل فئة
        save_heatmaps(test_ds, maps, masks, labels, Path(a.out) / f"heatmaps_{cat}.png")
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    df = pd.DataFrame(rows)
    print(df[["image_auroc", "pixel_auroc", "aupro"]].mean().round(4))


if __name__ == "__main__":
    main()
