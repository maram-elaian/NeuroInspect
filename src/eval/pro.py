import numpy as np
from scipy import ndimage

from src.eval.metrics import _to_hw


def pro_curve(gt_masks, anomaly_maps, num_bins=1000):
    """يرجّع (fpr, pro) مرتّبين تصاعديًا حسب fpr.

    - كل منطقة عيب (connected component) بوزن متساوي.
    - FPR محسوب على كل البكسلات السليمة (بما فيها صور good بالكامل).
    """
    gt = _to_hw(gt_masks) > 0.5
    am = _to_hw(anomaly_maps).astype(np.float64)
    assert gt.shape == am.shape, f"shape mismatch {gt.shape} vs {am.shape}"
    assert gt.any(), "no anomalous pixels in ground truth"

    lo, hi = am.min(), am.max()
    assert hi > lo, "anomaly maps are constant"
    edges = np.linspace(lo, hi, num_bins + 1)

    # البكسلات السليمة
    neg_hist, _ = np.histogram(am[~gt], bins=edges)
    n_neg = neg_hist.sum()

    # كل منطقة عيب: histogram مطبّع (نسبة بكسلاتها بكل bin)
    region_hists = []
    for g, a in zip(gt, am):
        if not g.any():
            continue
        labeled, n = ndimage.label(g)          # 4-connectivity افتراضيًا
        for r in range(1, n + 1):
            h, _ = np.histogram(a[labeled == r], bins=edges)
            region_hists.append(h / h.sum())
    region_hists = np.stack(region_hists)       # (R, bins)

    # عند threshold = edges[i]: نعتبر البكسل شاذًا إذا score >= edges[i]
    overlap = region_hists[:, ::-1].cumsum(axis=1)[:, ::-1]   # (R, bins)
    pro = overlap.mean(axis=0)
    fpr = neg_hist[::-1].cumsum()[::-1] / n_neg

    # نقطة (0,0) عند threshold أعلى من أعلى score
    fpr = np.append(fpr, 0.0)
    pro = np.append(pro, 0.0)
    return fpr[::-1], pro[::-1]


def aupro(gt_masks, anomaly_maps, fpr_limit=0.3, num_bins=1000):
    fpr, pro = pro_curve(gt_masks, anomaly_maps, num_bins)
    keep = fpr <= fpr_limit
    f, p = fpr[keep], pro[keep]
    if f[-1] < fpr_limit:                        # قصّ المنحنى عند الحد بالضبط
        f = np.append(f, fpr_limit)
        p = np.append(p, np.interp(fpr_limit, fpr, pro))
    area = np.sum(np.diff(f) * (p[1:] + p[:-1]) / 2)   # trapezoid
    return float(area / fpr_limit)
