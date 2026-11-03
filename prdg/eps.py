"""EPS for uint8 BGR images; masks are bool, True denotes plasma."""
import cv2
import numpy as np

def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

def background_mask(bgr):
    rgb = _rgb(bgr)
    intensity = rgb.mean(axis=2)
    threshold = np.quantile(intensity, .80)
    mask = intensity >= threshold
    if mask.mean() < .05:
        mask = intensity >= np.quantile(intensity, .65)
    return mask

def background_mask_v2(bgr, return_audit=False):
    """Low-saturation/high-value background mask with an expanded dark exclusion."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    otsu, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidate = (s <= otsu) & (v >= np.median(v))
    dark_thr = float(np.quantile(v, .15))
    dark = (v < dark_thr).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    excluded = cv2.dilate(dark, kernel, iterations=1).astype(bool)
    candidate &= ~excluded
    fallback = bool(candidate.mean() < .03)
    mask = background_mask(bgr) if fallback else candidate
    audit = dict(otsu_saturation_threshold=float(otsu), value_median=float(np.median(v)),
                 value_p15=dark_thr, candidate_fraction=float(candidate.mean()),
                 fallback= fallback, final_fraction=float(mask.mean()),
                 rule='S<=Otsu(S) AND V>=median(V) AND outside(dilate(V<p15,31x31))')
    return (mask, audit) if return_audit else mask

eps = background_mask_v2
