# NeuroInspect

Unsupervised visual anomaly detection on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) benchmark: train on defect-free images only, then detect and localize defects in new images with anomaly heatmaps.

> **Status: work in progress.** Dataset loader, evaluation metrics (Image AUROC, Pixel AUROC, AUPRO) and a convolutional autoencoder baseline are implemented. PatchCore and the research experiments are next.

## Motivation

Defects in manufacturing are rare and highly varied, so collecting labeled defect images for every defect type is impractical. Here the model only sees normal images during training and must flag anything that deviates from them.

## Dataset

MVTec AD: 15 categories (5 textures, 10 objects), 5,354 images in total. The training set contains only defect-free images; the test set contains good and defective images with pixel-level ground-truth masks.

| | Count |
|---|---|
| Categories | 15 |
| Train (good) | 3,629 |
| Test (good) | 467 |
| Test (defective) | 1,258 |

Per-category statistics are in [`results/dataset_stats.csv`](results/dataset_stats.csv).

The data is **not** included in this repository. Download it from the official page (license: CC BY-NC-SA 4.0, non-commercial use only).

## Evaluation metrics

- **Image AUROC:** can the model separate defective from good images? Image score = max of the anomaly map.
- **Pixel AUROC:** localization quality over all test pixels. It can look inflated because defects are small compared to the image.
- **AUPRO:** per-region overlap, integrated up to FPR = 0.3. Every defect region has equal weight regardless of its size. Implemented in `src/eval/pro.py` (4-connected components, histogram-based threshold sweep) and unit-tested against cases with known answers; values may differ slightly from other libraries.

## Methods

| Method | Status |
|---|---|
| Convolutional autoencoder (reconstruction error) | Implemented |
| PatchCore (WideResNet50) | Planned |
| PaDiM | Planned |

**Autoencoder baseline.** Trained per category on good images only. 10% of the training images are held out for validation and the best epoch is chosen by validation loss; the test set is used only for the final evaluation. Anomaly map = per-pixel squared reconstruction error, smoothed with a Gaussian filter (sigma = 4).

## Results

Baseline results will be added once the full run is complete (40 epochs, seed 0).

| Category | Image AUROC | Pixel AUROC | AUPRO |
|---|---|---|---|
| _to be filled_ | | | |

## Repository structure

```
NeuroInspect/
├── configs/            # experiment configs (planned)
├── scripts/
│   └── run_ae.py       # train + evaluate the autoencoder on all categories
├── src/
│   ├── data/mvtec.py   # dataset and transforms
│   ├── models/         # autoencoder (PatchCore / PaDiM planned)
│   ├── eval/           # metrics.py, pro.py, viz.py
│   └── utils/seed.py
├── tests/              # unit tests for the metrics and model shapes
├── results/
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q

python scripts/run_ae.py --data-root /path/to/mvtec_ad --epochs 40 --seed 0 --out outputs/ae
```

The script writes `results_seed<N>.csv` and one heatmap figure per category. It saves after every category and skips finished ones if you rerun it.

On Kaggle: add the MVTec AD dataset as input, enable the GPU and Internet, clone this repo into `/kaggle/working`, then run the command above.

## Roadmap

- [x] Dataset loader and data checks
- [x] Image AUROC, Pixel AUROC, AUPRO with tests
- [x] Autoencoder baseline
- [ ] PatchCore from scratch, checked against published numbers
- [ ] Few-shot curve (number of training images vs. performance)
- [ ] Backbone comparison (WideResNet50 vs. self-supervised ViT)
- [ ] Failure analysis and Gradio demo

## Limitations

- The autoencoder is expected to be weak on object categories; it is included as a baseline, not as a competitive method.
- Results are reported for a single seed so far; multi-seed mean ± std will be added for the final experiments.

## References

- P. Bergmann et al., *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- P. Bergmann et al., *The MVTec Anomaly Detection Dataset*, IJCV 2021 (PRO metric).
- K. Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.
- T. Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, 2021.

## License

Code: add a license of your choice (e.g. MIT). The MVTec AD dataset has its own license (CC BY-NC-SA 4.0).
