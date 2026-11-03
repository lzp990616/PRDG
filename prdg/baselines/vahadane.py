"""Original NumPy sparse-NMF backend; not SPAMS."""
import numpy as np
from prdg.eps import _rgb
from prdg.normalization import _bgr

def optical_density(bgr):
    rgb = _rgb(bgr)
    return -np.log((rgb.reshape(-1, 3) + 1.) / 256.)

def _initial_stains(od):
    sample = od[od.sum(1) > .15]
    if len(sample) < 8:
        return np.array([[.65, .70, .29], [.07, .99, .11]], np.float64)
    _, _, vt = np.linalg.svd(sample - sample.mean(0), full_matrices=False)
    plane = sample @ vt[:2].T
    angle = np.arctan2(plane[:, 1], plane[:, 0])
    lo, hi = np.quantile(angle, [.01, .99])
    stains = np.stack([vt[:2].T @ np.array([np.cos(a), np.sin(a)]) for a in (lo, hi)])
    stains = np.maximum(stains, .01)
    stains /= np.maximum(np.linalg.norm(stains, axis=1, keepdims=True), 1e-8)
    return stains

def _sparse_factor(od, dictionary, iterations=35, penalty=.015):
    # Multiplicative sparse-NMF updates for V ~= H @ D, H,D >= 0.
    v = np.maximum(od, 0.).astype(np.float64)
    d = np.maximum(dictionary, 1e-5).astype(np.float64)
    h = np.maximum(v @ d.T, 1e-5)
    gram = d @ d.T
    cross = v @ d.T
    for _ in range(iterations):
        h *= cross / np.maximum(h @ gram + penalty, 1e-9)
    return h

def vahadane_template(bgr):
    od = optical_density(bgr)
    rng = np.random.default_rng(1234)
    sample = od[od.sum(1) > .15]
    if len(sample) > 12000:
        sample = sample[rng.choice(len(sample), 12000, replace=False)]
    d = _initial_stains(sample)
    h = _sparse_factor(sample, d, iterations=80, penalty=.01)
    # Alternate dictionary and concentration updates to fit the sparse OD basis.
    for _ in range(35):
        d *= (h.T @ sample) / np.maximum((h.T @ h) @ d, 1e-9)
        d = np.maximum(d, 1e-5)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
        h = _sparse_factor(sample, d, iterations=8, penalty=.01)
    order = np.argsort(d[:, 0] / np.maximum(d.sum(1), 1e-8))[::-1]
    d = d[order]
    h = _sparse_factor(od, d, iterations=35, penalty=.01)
    scale = np.maximum(np.quantile(h, .99, axis=0), 1e-6)
    return dict(stains=d.tolist(), concentration_p99=scale.tolist())

def vahadane_transform(bgr, target):
    od = optical_density(bgr)
    source_stains = _initial_stains(od)
    sample = od[od.sum(1) > .15]
    if len(sample) > 12000:
        sample = sample[np.random.default_rng(1234).choice(len(sample), 12000, replace=False)]
    hfit = _sparse_factor(sample, source_stains, iterations=45, penalty=.01)
    for _ in range(20):
        source_stains *= (hfit.T @ sample) / np.maximum((hfit.T @ hfit) @ source_stains, 1e-9)
        source_stains = np.maximum(source_stains, 1e-5)
        source_stains /= np.maximum(np.linalg.norm(source_stains, axis=1, keepdims=True), 1e-8)
        hfit = _sparse_factor(sample, source_stains, iterations=6, penalty=.01)
    order = np.argsort(source_stains[:, 0] / np.maximum(source_stains.sum(1), 1e-8))[::-1]
    source_stains = source_stains[order]
    concentrations = _sparse_factor(od, source_stains, iterations=30, penalty=.01)
    source_scale = np.maximum(np.quantile(concentrations, .99, axis=0), 1e-6)
    target_stains = np.asarray(target['stains'], np.float64)
    target_scale = np.asarray(target['concentration_p99'], np.float64)
    normalized = concentrations * (target_scale / source_scale)[None, :]
    rgb = 256. * np.exp(-normalized @ target_stains) - 1.
    return _bgr(rgb.reshape(bgr.shape[0], bgr.shape[1], 3))
