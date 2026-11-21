"""Dataset Loader and Feature Pipeline.

Loads fruit images from data/<split>/<class>/*.jpg, applies OpenCV preprocessing,
extracts handcrafted features, and caches results to disk for fast subsequent runs.
"""
from pathlib import Path
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from config import DATA_DIR
from preprocessing import load_image, preprocess
from features import extract_features

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_classes(split_dir):
    """Retrieve sorted subfolder names representing class labels."""
    return sorted(d.name for d in Path(split_dir).iterdir() if d.is_dir())


def _process_single_image(args):
    """Worker function for concurrent image preprocessing and feature extraction."""
    file_path, class_idx = args
    try:
        bgr = load_image(file_path)
        out = preprocess(bgr)
        feat = extract_features(out["rgb"], out["mask"])
        return out["rgb"], feat, class_idx
    except Exception:
        return None


def load_split(split_dir, classes=None, cache=True, verbose=True):
    """Load, preprocess, and extract features for a given dataset split.

    Args:
        split_dir (str or Path): Path to split directory (e.g., data/train).
        classes (list, optional): List of class names. If None, derived from folders.
        cache (bool): If True, caches processed arrays to data/cache_<split>.npz.
        verbose (bool): Print progress.

    Returns:
        X (np.ndarray): (N, 224, 224, 3) uint8 RGB images.
        F (np.ndarray): (N, 69) float32 handcrafted features.
        y (np.ndarray): (N,) int32 class labels.
        classes (list): Class names.
    """
    split_dir = Path(split_dir)
    split_name = split_dir.name
    cache_path = DATA_DIR / f"cache_{split_name}.npz"

    # Derive or check classes
    classes = classes or list_classes(split_dir)

    # Check disk cache
    if cache and cache_path.exists():
        if verbose:
            print(f"Loading cached preprocessed data from {cache_path.name} ...")
        cached = np.load(cache_path, allow_pickle=True)
        cached_classes = list(cached["classes"])
        if cached_classes == classes:
            return cached["X"], cached["F"], cached["y"], cached_classes

    if verbose:
        print(f"Processing '{split_name}' split with {len(classes)} classes: {classes} ...")

    # Gather tasks
    tasks = []
    for idx, cls_name in enumerate(classes):
        cls_dir = split_dir / cls_name
        files = [p for p in cls_dir.rglob("*") if p.suffix.lower() in EXTS]
        if verbose:
            print(f"  Found {len(files):4d} images in '{cls_name}'")
        for f in files:
            tasks.append((f, idx))

    # Parallel processing across CPU threads
    X, F, y = [], [], []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for result in executor.map(_process_single_image, tasks):
            if result is not None:
                img_rgb, feat, label = result
                X.append(img_rgb)
                F.append(feat)
                y.append(label)

    X_arr = np.stack(X)
    F_arr = np.stack(F)
    y_arr = np.array(y, dtype=np.int32)

    if verbose:
        print(f"Successfully processed {len(X_arr)} images for '{split_name}'.")

    # Save cache
    if cache:
        np.savez_compressed(cache_path, X=X_arr, F=F_arr, y=y_arr, classes=np.array(classes))
        if verbose:
            print(f"Saved cache -> {cache_path.name}")

    return X_arr, F_arr, y_arr, classes
