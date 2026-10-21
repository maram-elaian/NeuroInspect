import numpy as np
from sklearn.metrics import roc_auc_score


def _to_hw(x):
    """يقبل (N,H,W) أو (N,1,H,W) ويرجّع (N,H,W)."""
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    assert x.ndim == 3, f"expected (N,H,W), got {x.shape}"
    return x


def image_scores_from_maps(anomaly_maps):
    """درجة الصورة = أعلى قيمة في الـ anomaly map."""
    maps = _to_hw(anomaly_maps)
    return maps.reshape(len(maps), -1).max(axis=1)


def image_auroc(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    assert len(labels) == len(scores)
    assert len(np.unique(labels)) == 2, "need both good and defective images"
    return float(roc_auc_score(labels, scores))


def pixel_auroc(gt_masks, anomaly_maps):
    gt = (_to_hw(gt_masks) > 0.5).astype(np.uint8)
    am = _to_hw(anomaly_maps).astype(np.float32)
    assert gt.shape == am.shape, f"shape mismatch {gt.shape} vs {am.shape}"
    assert gt.max() == 1, "no anomalous pixels in ground truth"
    return float(roc_auc_score(gt.ravel(), am.ravel()))
