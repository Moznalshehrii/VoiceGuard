"""EER and min-DCF -- the two standard metrics for anti-spoofing/deepfake detection.

EER is the point on the ROC curve where the false-acceptance rate (spoof
scored as bonafide) equals the false-rejection rate (bonafide scored as
spoof). Lower is better. Reported here as a percentage.

min-DCF (minimum normalized Detection Cost Function) is a cost-weighted
alternative to EER: unlike EER, it lets false accepts and false rejects
carry different costs and reflects how rare/common attacks are assumed to
be (p_target), rather than treating both error types as equally bad at a
50/50 base rate. Common in ASVspoof-style anti-spoofing papers alongside
EER. Defaults below (p_target=0.05, c_miss=1, c_fa=1) follow the
NIST/ASVspoof-style convention; adjust them if your evaluation protocol
specifies different official values.
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


def compute_min_dcf(
    labels, scores, p_target: float = 0.05, c_miss: float = 1.0, c_fa: float = 1.0
) -> tuple[float, float]:
    """
    labels: array-like of 0/1, where 1 = bonafide (genuine), 0 = spoof
    scores: array-like of float, higher = more likely bonafide
    p_target: prior probability of a bonafide trial
    c_miss: cost of rejecting a real bonafide trial (false reject)
    c_fa: cost of accepting a spoof trial (false accept)

    Returns (min_dcf_normalized, threshold_at_min_dcf). Normalized so that
    a trivial always-accept or always-reject system scores 1.0 -- lower is
    better, and a working system should score well below 1.0.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr  # P_miss at each threshold
    p_fa = fpr  # P_false_alarm at each threshold

    dcf = c_miss * fnr * p_target + c_fa * p_fa * (1 - p_target)
    idx = np.argmin(dcf)

    dcf_default = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf_normalized = dcf[idx] / dcf_default
    return float(min_dcf_normalized), float(thresholds[idx])
