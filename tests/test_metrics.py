import numpy as np
from src.eval.metrics import image_auroc, pixel_auroc, image_scores_from_maps


def test_image_auroc_perfect_and_reversed():
    labels = [0, 0, 1, 1]
    assert image_auroc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert image_auroc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_pixel_auroc_perfect():
    gt = np.zeros((2, 8, 8)); gt[1, 2:4, 2:4] = 1
    assert pixel_auroc(gt, gt.copy()) == 1.0


def test_random_is_near_half():
    rng = np.random.default_rng(0)
    gt = np.zeros((20, 32, 32)); gt[10:, 8:16, 8:16] = 1
    auc = pixel_auroc(gt, rng.random(gt.shape))
    assert 0.45 < auc < 0.55


def test_image_score_is_max():
    m = np.zeros((2, 4, 4)); m[1, 1, 1] = 5
    assert list(image_scores_from_maps(m)) == [0.0, 5.0]
