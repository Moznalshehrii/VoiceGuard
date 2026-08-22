"""Equal Error Rate (EER) -- the standard metric for anti-spoofing/deepfake detection.

EER is the point on the ROC curve where the false-acceptance rate (spoof
scored as bonafide) equals the false-rejection rate (bonafide scored as
spoof). Lower is better. Reported here as a percentage.
"""

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(labels, scores) -> tuple[float, float]:
    """
    labels: array-like of 0/1, where 1 = bonafide (genuine), 0 = spoof
    scores: array-like of float, higher = more likely bonafide

    Returns (eer_percent, threshold_at_eer)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr  # false-rejection rate: bonafide wrongly called spoof

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer * 100), float(thresholds[idx])
