"""OpenCV Image Preprocessing Pipeline.

Pipeline Stages:
  1. Spatial resizing: Standardizes dimensions for consistent computation.
  2. Edge-preserving Denoising: Bilateral filter removes sensor noise while keeping boundaries sharp.
  3. CIE LAB CLAHE: Contrast Limited Adaptive Histogram Equalization on L-channel enhances
     blemishes and surface defects without altering natural fruit color balance.
  4. Border-color based Otsu Segmentation & Morphology: Separates fruit from background
     using morphological opening (removes dust) and closing (fills internal holes).
  5. Contour Analysis & Bounding Box Extraction: Isolates largest connected fruit component.
"""
import cv2
import numpy as np
from config import IMG_SIZE


def load_image(path):
    """Load an image from disk using OpenCV, returning BGR array."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image from path: {path}")
    return img


def segment_fruit(bgr):
    """Separate the fruit from background in CIE LAB color space.

    Estimates the background color from image border pixels, computes distance,
    applies Otsu thresholding, cleans with morphological filters, and extracts
    the largest fruit contour.

    Returns:
        mask (uint8 array, 0 or 255): Binary fruit mask.
        bbox (tuple): Bounding box (x, y, w, h).
    """
    h, w = bgr.shape[:2]
    # Denoise before color distance
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Sample border pixels (5% border width) to estimate background color
    b = max(3, int(0.05 * min(h, w)))
    border_pixels = np.concatenate([
        lab[:b, :].reshape(-1, 3),
        lab[-b:, :].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3),
        lab[:, -b:].reshape(-1, 3)
    ], axis=0)
    bg_color = np.median(border_pixels, axis=0)

    # Compute Euclidean distance from estimated background color
    dist = np.linalg.norm(lab - bg_color, axis=2)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu automatic thresholding
    _, mask = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations: Close small holes, then open to remove speckles
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # Find external contours and select the largest
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        # Verify contour is at least 4% of the image area
        if area >= 0.04 * h * w:
            clean_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)
            return clean_mask, cv2.boundingRect(c)

    # Fallback to entire image if background was non-uniform
    return np.full((h, w), 255, dtype=np.uint8), (0, 0, w, h)


def _square_crop(img, mask, bbox, margin=0.10):
    """Crop square region centered on bounding box with padding margin."""
    x, y, bw, bh = bbox
    side = int(max(bw, bh) * (1.0 + margin))
    cx, cy = x + bw // 2, y + bh // 2
    pad = side

    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    mask_padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    x0 = cx - side // 2 + pad
    y0 = cy - side // 2 + pad
    return img_padded[y0:y0 + side, x0:x0 + side], mask_padded[y0:y0 + side, x0:x0 + side]


def preprocess(bgr, remove_background=True, size=IMG_SIZE):
    """Full OpenCV Preprocessing Pipeline.

    Returns:
        dict containing:
          - 'rgb': 224x224x3 uint8 RGB array (ready for neural network)
          - 'mask': 224x224 uint8 binary mask (0 or 255)
          - 'enhanced_full': Contrast-enhanced BGR image
          - 'overlay': Original BGR with green contour and bounding box
          - 'bbox': (x, y, w, h) bounding box
    """
    # 1. Resize down if very high resolution for speed and memory efficiency
    h, w = bgr.shape[:2]
    max_dim = max(h, w)
    if max_dim > 640:
        scale = 640.0 / max_dim
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 2. Edge-preserving Denoising (Bilateral Filter)
    denoised = cv2.bilateralFilter(bgr, d=7, sigmaColor=60, sigmaSpace=60)

    # 3. Contrast enhancement: CLAHE on CIE LAB L-channel
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

    # 4. Background segmentation on denoised image
    mask, bbox = segment_fruit(denoised)

    # 5. Square crop around fruit, optional background removal, and resize to standard IMG_SIZE
    crop_bgr, crop_mask = _square_crop(enhanced, mask, bbox)
    crop_bgr = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(crop_mask, (size, size), interpolation=cv2.INTER_NEAREST)

    if remove_background:
        crop_bgr = cv2.bitwise_and(crop_bgr, crop_bgr, mask=crop_mask)

    # 6. Prepare visual overlay on original image
    overlay = bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    bx, by, bw, bh = bbox
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)

    return {
        "rgb": cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
        "mask": crop_mask,
        "enhanced_full": enhanced,
        "overlay": overlay,
        "bbox": bbox
    }


def get_step_by_step_stages(bgr, size=IMG_SIZE):
    """Generate all individual visual steps for reports, figures, and UI demonstration."""
    # Step 1: Input image (original RGB)
    step1_input = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Scale if necessary
    h, w = bgr.shape[:2]
    max_dim = max(h, w)
    if max_dim > 640:
        scale = 640.0 / max_dim
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Step 2: Denoised
    denoised = cv2.bilateralFilter(bgr, d=7, sigmaColor=60, sigmaSpace=60)
    step2_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    # Step 3: CLAHE contrast enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
    step3_clahe = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    # Step 4: Otsu morphological binary segmentation mask
    mask, bbox = segment_fruit(denoised)
    step4_mask = mask.copy()

    # Step 5: Segmented fruit (background masked out)
    segmented_bgr = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    step5_segmented = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)

    # Step 6: Final normalized input (224x224 cropped)
    crop_bgr, crop_mask = _square_crop(enhanced, mask, bbox)
    crop_bgr = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(crop_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    final_bgr = cv2.bitwise_and(crop_bgr, crop_bgr, mask=crop_mask)
    step6_final = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

    return {
        "step1_input": step1_input,
        "step2_denoised": step2_denoised,
        "step3_clahe": step3_clahe,
        "step4_mask": step4_mask,
        "step5_segmented": step5_segmented,
        "step6_final": step6_final
    }
