"""BGR image I/O; reference/target vectors are RGB. No labels are used."""
import cv2
import numpy as np
from .eps import _rgb, background_mask_v2

def _bgr(rgb):
    return cv2.cvtColor(np.rint(np.clip(rgb, 0, 255)).astype(np.uint8), cv2.COLOR_RGB2BGR)

def masked_mean(bgr, mask):
    rgb = _rgb(bgr)
    if mask.sum() < 64:
        mask = np.ones(mask.shape, dtype=bool)
    return rgb[mask].mean(axis=0)

def channel_normalize(bgr, target, mask):
    rgb = _rgb(bgr)
    source = masked_mean(bgr, mask)
    gain = np.asarray(target, np.float32) / np.maximum(source, 1.0)
    return _bgr(rgb * np.clip(gain, .5, 2.0)[None, None, :])

def fit_target(images):
    """Unweighted mean of per-image plasma means; TRAIN split only.

    Deliberately retain the original float64 reduction of uint8 pixels here,
    unlike masked_mean's float32 reduction used during transformation.
    """
    means = []
    for image in images:
        mask = background_mask_v2(image)
        means.append(image[:, :, ::-1][mask].mean(0))
    if not means:
        raise ValueError('Empty training split')
    return np.asarray(np.mean(means, axis=0).astype(float).tolist(), np.float32)

def normalize(image, target):
    return channel_normalize(image, target, background_mask_v2(image))
