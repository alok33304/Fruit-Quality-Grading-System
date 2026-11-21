"""OpenCV Feature Extraction Module.

Extracts explainable, physics-grounded visual descriptors:
  1. Color Moments & Histograms (HSV & CIE LAB):
     - HSV Histograms (16 Hue bins, 8 Saturation bins, 8 Value bins = 32 features)
     - Mean and Standard Deviation in HSV and LAB color spaces (12 features)
  2. Defect & Oxidation Descriptors:
     - Necrosis / dark decay pixel ratio (Value < 60)
     - Browning / oxidation pixel ratio (Hue 5-25, Saturation > 60, Value < 140)
     - Gaussian difference dark spot area ratio
     - Connected-component dark spot count (normalized)
  3. Texture & Surface Irregularity:
     - Laplacian variance (surface roughness / sharpness)
     - Sobel gradient magnitude (edge irregularity)
     - Local Binary Patterns (LBP) histogram (16 bins)
  4. Shape & Morphological Geometry:
     - Area ratio (fruit area / bounding crop area)
     - Circularity (4 * pi * Area / Perimeter^2)
     - Solidity (Area / Convex Hull Area)

Total Feature Dimension: 32 + 12 + 4 + 2 + 16 + 3 = 69 features.
"""
import cv2
import numpy as np


def _lbp_hist(gray, mask, bins=16):
    """Compute Local Binary Pattern (LBP) histogram on masked fruit surface."""
    c = gray[1:-1, 1:-1]
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    code = np.zeros_like(c, dtype=np.uint8)
    for i, (dy, dx) in enumerate(shifts):
        nb = gray[1 + dy:gray.shape[0] - 1 + dy, 1 + dx:gray.shape[1] - 1 + dx]
        code |= ((nb >= c).astype(np.uint8) << i)

    m = mask[1:-1, 1:-1] > 0
    if m.sum() == 0:
        return np.zeros(bins, dtype=np.float32)

    hist, _ = np.histogram(code[m], bins=bins, range=(0, 256))
    total = hist.sum()
    return (hist / max(total, 1)).astype(np.float32)


def extract_features(rgb, mask):
    """Extract 69-dimensional OpenCV feature vector from fruit RGB image and mask.

    Args:
        rgb (np.ndarray): HxWx3 uint8 RGB image.
        mask (np.ndarray): HxW uint8 binary mask (0 or 255).

    Returns:
        np.ndarray: 1-D float32 feature vector of length 69.
    """
    m = mask > 0
    if m.sum() < 50:
        m = np.ones_like(m)
    mask8 = (m * 255).astype(np.uint8)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    feats = []

    # 1. Color Histograms: H (16 bins), S (8 bins), V (8 bins) -> 32 features
    for ch, bins, rng in [(0, 16, [0, 180]), (1, 8, [0, 256]), (2, 8, [0, 256])]:
        h = cv2.calcHist([hsv], [ch], mask8, [bins], rng).flatten()
        feats.extend(h / max(h.sum(), 1))

    # 2. Color Statistics: HSV (6 features) + LAB (6 features) -> 12 features
    for img in (hsv, lab):
        for ch in range(3):
            v = img[..., ch][m].astype(np.float32)
            feats.extend([float(v.mean() / 255.0), float(v.std() / 255.0)])

    # 3. Defect & Oxidation Descriptors -> 4 features
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    n_pixels = float(m.sum())
    dark_ratio = float(((V < 60) & m).sum()) / n_pixels
    browning_ratio = float(((H >= 5) & (H <= 25) & (S > 60) & (V < 140) & m).sum()) / n_pixels
    feats.extend([dark_ratio, browning_ratio])

    # Dark spot lesions via Difference of Gaussians
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    spots = ((blur.astype(int) - gray.astype(int)) > 25) & m
    n_spots, _ = cv2.connectedComponents(spots.astype(np.uint8))
    spot_area_ratio = float(spots.sum()) / n_pixels
    spot_count_norm = min(float(n_spots - 1), 50.0) / 50.0
    feats.extend([spot_area_ratio, spot_count_norm])

    # 4. Texture & Surface Irregularity -> 18 features (2 + 16 LBP)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F)[m].var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad_mean = float(np.sqrt(gx ** 2 + gy ** 2)[m].mean())
    feats.extend([float(np.log1p(lap_var) / 10.0), float(grad_mean / 255.0)])
    feats.extend(_lbp_hist(gray, mask8, bins=16))

    # 5. Shape & Geometry -> 3 features
    cnts, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_ratio = circularity = solidity = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        a = float(cv2.contourArea(c))
        p = float(cv2.arcLength(c, True))
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))

        area_ratio = a / float(mask.size)
        circularity = (4.0 * np.pi * a) / (p * p) if p > 0 else 0.0
        solidity = a / hull_area if hull_area > 0 else 0.0

    feats.extend([area_ratio, circularity, solidity])

    return np.asarray(feats, dtype=np.float32)


FEATURE_DIM = 32 + 12 + 4 + 2 + 16 + 3  # = 69 features


def get_feature_names():
    """Return human-readable names for all 69 features."""
    names = []
    names.extend([f"hist_hue_bin_{i}" for i in range(16)])
    names.extend([f"hist_sat_bin_{i}" for i in range(8)])
    names.extend([f"hist_val_bin_{i}" for i in range(8)])
    names.extend(["hsv_h_mean", "hsv_h_std", "hsv_s_mean", "hsv_s_std", "hsv_v_mean", "hsv_v_std"])
    names.extend(["lab_l_mean", "lab_l_std", "lab_a_mean", "lab_a_std", "lab_b_mean", "lab_b_std"])
    names.extend(["defect_dark_ratio", "defect_browning_ratio", "defect_spot_area_ratio", "defect_spot_count"])
    names.extend(["texture_laplacian_var", "texture_sobel_gradient"])
    names.extend([f"texture_lbp_bin_{i}" for i in range(16)])
    names.extend(["shape_area_ratio", "shape_circularity", "shape_solidity"])
    return names


def summarize_features(feats):
    """Convert raw 69-D feature vector into human-interpretable quality metrics."""
    # Indices based on feature construction:
    # 32 (hist) + 12 (moments) = 44
    dark_ratio = feats[44]
    browning_ratio = feats[45]
    spot_area = feats[46]
    spot_count = feats[47] * 50.0  # de-normalize
    lap_var = feats[48] * 10.0     # log scale
    circ = feats[67]
    solidity = feats[68]

    # Calculate composite defect index (0% to 100%)
    defect_severity = min(100.0, (dark_ratio * 50.0 + browning_ratio * 40.0 + spot_area * 30.0) * 100.0)

    return {
        "defect_severity_pct": float(defect_severity),
        "dark_decay_ratio": float(dark_ratio * 100.0),
        "browning_index": float(browning_ratio * 100.0),
        "blemish_spot_count": int(spot_count),
        "texture_roughness": float(lap_var),
        "shape_circularity": float(circ),
        "shape_solidity": float(solidity)
    }
