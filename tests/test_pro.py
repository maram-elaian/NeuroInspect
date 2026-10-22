import numpy as np
from src.eval.pro import pro_curve, aupro


def test_perfect_map_gives_one():
    gt = np.zeros((2, 16, 16)); gt[1, 4:8, 4:8] = 1
    assert np.isclose(aupro(gt, gt.copy()), 1.0, atol=1e-3)


def test_random_map_is_about_015():
    # للخريطة العشوائية PRO ≈ FPR، فالمساحة = 0.3²/2 ثم /0.3 = 0.15
    rng = np.random.default_rng(0)
    gt = np.zeros((20, 32, 32)); gt[10:, 8:16, 8:16] = 1
    v = aupro(gt, rng.random(gt.shape))
    assert 0.10 < v < 0.20


def test_regions_weighted_equally():
    # منطقة كبيرة (16 بكسل) ومنطقة صغيرة (4 بكسل)، والموديل التقط الكبيرة فقط
    gt = np.zeros((1, 16, 16))
    gt[0, 0:4, 0:4] = 1
    gt[0, 10:12, 10:12] = 1
    am = gt.copy(); am[0, 10:12, 10:12] = 0
    fpr, pro = pro_curve(gt, am)
    # PRO = (1 + 0) / 2 = 0.5، وليس 16/20 = 0.8 كما لو كان الوزن بالبكسل
    assert np.isclose(pro[fpr == 0].max(), 0.5)


def test_curve_is_monotone():
    rng = np.random.default_rng(1)
    gt = np.zeros((6, 16, 16)); gt[3:, 4:9, 4:9] = 1
    fpr, pro = pro_curve(gt, rng.random(gt.shape))
    assert np.all(np.diff(fpr) >= -1e-12)
    assert np.all(np.diff(pro) >= -1e-12)
