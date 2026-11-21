"""Generate Visual Figures, Charts, and Screenshots for Project Report.

Generates:
  1. outputs/preprocessing_pipeline.png: 6-stage OpenCV visual pipeline.
  2. outputs/feature_extraction_analysis.png: Feature distributions (Fresh vs Rotten).
  3. outputs/sample_predictions_grid.png: 3x3 test prediction grid with bounding boxes.
  4. outputs/system_architecture_diagram.png: Flowchart of the end-to-end architecture.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import config
from preprocessing import load_image, get_step_by_step_stages, preprocess
from features import extract_features, summarize_features
from inference import FruitGrader


def generate_preprocessing_pipeline_figure(sample_image_path):
    """Generate 6-stage visual pipeline figure for project report."""
    print(f"Generating preprocessing pipeline figure from {sample_image_path} ...")
    bgr = load_image(sample_image_path)
    stages = get_step_by_step_stages(bgr)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=150)
    fig.suptitle("OpenCV Image Preprocessing Pipeline Stages", fontsize=15, fontweight="bold", y=0.98)

    titles = [
        ("(a) Raw Input Image", stages["step1_input"], "original"),
        ("(b) Spatial Rescaling (224x224)", stages["step6_final"], "resized"),
        ("(c) Bilateral Denoising", stages["step2_denoised"], "denoised"),
        ("(d) CIE LAB CLAHE Contrast Enhancement", stages["step3_clahe"], "clahe"),
        ("(e) Otsu Morphological Binary Mask", stages["step4_mask"], "gray"),
        ("(f) Segmented Fruit (Background Removed)", stages["step5_segmented"], "segmented")
    ]

    for ax, (title, img, mode) in zip(axes.flat, titles):
        if mode == "gray":
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    plt.tight_layout()
    out_path = config.OUTPUT_DIR / "preprocessing_pipeline.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def generate_feature_analysis_figure(data_dir=config.DATA_DIR):
    """Generate comparative distributions of extracted OpenCV features."""
    print("Generating feature analysis figure ...")
    fresh_files = list((data_dir / "test" / "fresh").glob("*.jpg"))[:40]
    rotten_files = list((data_dir / "test" / "rotten").glob("*.jpg"))[:40]

    fresh_metrics = []
    for f in fresh_files:
        try:
            out = preprocess(load_image(f))
            feats = extract_features(out["rgb"], out["mask"])
            fresh_metrics.append(summarize_features(feats))
        except Exception:
            continue

    rotten_metrics = []
    for f in rotten_files:
        try:
            out = preprocess(load_image(f))
            feats = extract_features(out["rgb"], out["mask"])
            rotten_metrics.append(summarize_features(feats))
        except Exception:
            continue

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.suptitle("Statistical Comparison of Handcrafted OpenCV Features", fontsize=15, fontweight="bold", y=0.98)

    # 1. Defect Severity Percentage
    ax = axes[0, 0]
    fresh_val = [m["defect_severity_pct"] for m in fresh_metrics]
    rotten_val = [m["defect_severity_pct"] for m in rotten_metrics]
    ax.boxplot([fresh_val, rotten_val], labels=["Fresh (Grade A)", "Rotten (Grade C)"],
               patch_artist=True, boxprops=dict(facecolor="#90EE90", color="green"),
               medianprops=dict(color="black", lw=2))
    ax.set_title("Defect Severity Index (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Severity (%)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 2. Browning Oxidation Ratio
    ax = axes[0, 1]
    fresh_val = [m["browning_index"] for m in fresh_metrics]
    rotten_val = [m["browning_index"] for m in rotten_metrics]
    ax.boxplot([fresh_val, rotten_val], labels=["Fresh (Grade A)", "Rotten (Grade C)"],
               patch_artist=True, boxprops=dict(facecolor="#FFE4B5", color="#D2691E"),
               medianprops=dict(color="black", lw=2))
    ax.set_title("Browning & Oxidation Ratio (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Browning Area (%)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 3. Surface Texture Roughness (Laplacian Variance)
    ax = axes[1, 0]
    fresh_val = [m["texture_roughness"] for m in fresh_metrics]
    rotten_val = [m["texture_roughness"] for m in rotten_metrics]
    ax.boxplot([fresh_val, rotten_val], labels=["Fresh (Grade A)", "Rotten (Grade C)"],
               patch_artist=True, boxprops=dict(facecolor="#ADD8E6", color="blue"),
               medianprops=dict(color="black", lw=2))
    ax.set_title("Surface Texture Roughness (Laplacian Var)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Variance (Log Scale)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 4. Shape Circularity
    ax = axes[1, 1]
    fresh_val = [m["shape_circularity"] for m in fresh_metrics]
    rotten_val = [m["shape_circularity"] for m in rotten_metrics]
    ax.boxplot([fresh_val, rotten_val], labels=["Fresh (Grade A)", "Rotten (Grade C)"],
               patch_artist=True, boxprops=dict(facecolor="#D8BFD8", color="purple"),
               medianprops=dict(color="black", lw=2))
    ax.set_title("Contour Circularity (Firmness / Shape)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Circularity (4*pi*Area / P^2)")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = config.OUTPUT_DIR / "defect_detection_analysis.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def generate_sample_predictions_grid():
    """Generate 3x3 grid of real test predictions with grading annotations."""
    print("Generating sample predictions grid ...")
    grader = FruitGrader()

    test_samples = [
        ("fresh", list((config.DATA_DIR / "test" / "fresh").glob("*.jpg"))[:3]),
        ("medium", list((config.DATA_DIR / "test" / "medium").glob("*.jpg"))[:3]),
        ("rotten", list((config.DATA_DIR / "test" / "rotten").glob("*.jpg"))[:3]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 13), dpi=150)
    fig.suptitle("Sample Test Predictions with Visual OpenCV Bounding Boxes", fontsize=15, fontweight="bold", y=0.98)

    row = 0
    for true_cls, files in test_samples:
        for col, fpath in enumerate(files):
            ax = axes[row, col]
            bgr = load_image(fpath)
            res = grader.predict(bgr)

            # Display annotated RGB
            rgb_overlay = cv2.cvtColor(res["annotated_overlay"], cv2.COLOR_BGR2RGB)
            ax.imshow(rgb_overlay)

            # Determine match status
            is_correct = res["grade"] == true_cls
            status_text = "CORRECT" if is_correct else "MISMATCH"
            badge_color = "green" if is_correct else "red"

            pred_label = res["label"].split("(")[0].strip()
            title = f"True: {true_cls.capitalize()} | Pred: {pred_label}\nConf: {res['confidence']:.1%} [{status_text}]"
            ax.set_title(title, fontsize=10, fontweight="bold", color=badge_color, pad=6)
            ax.axis("off")
        row += 1

    plt.tight_layout()
    out_path = config.OUTPUT_DIR / "sample_predictions_grid.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def generate_architecture_diagram():
    """Generate clean system architecture block diagram for report."""
    print("Generating system architecture diagram ...")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Colors
    c_input = "#E3F2FD"      # Light Blue
    c_cv = "#E8F5E9"         # Light Green
    c_cnn = "#FFF3E0"        # Light Orange
    c_fusion = "#EDE7F6"     # Light Purple
    c_output = "#FCE4EC"     # Light Pink

    # Helper box drawer
    def draw_box(x, y, w, h, text, subtitle="", color="#FFFFFF", edge="#333333"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                      facecolor=color, edgecolor=edge, lw=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.15, text, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#111111")
        if subtitle:
            ax.text(x + w / 2, y + h / 2 - 0.22, subtitle, ha="center", va="center",
                    fontsize=8.5, color="#444444")

    # Flowchart boxes
    # Stage 1: Input
    draw_box(0.5, 2.7, 2.0, 1.6, "Input Image", "RGB Fruit Photo\n(Any resolution)", c_input)

    # Stage 2: OpenCV Preprocessing
    draw_box(3.2, 2.7, 2.4, 1.6, "OpenCV Preprocessing", "Bilateral Denoising\nCIE LAB + CLAHE\nOtsu Morph Segmentation", c_cv)

    # Stage 3A: MobileNetV2
    draw_box(6.3, 4.4, 2.5, 1.5, "MobileNetV2 Backbone", "Pretrained on ImageNet\n1280-D Deep Embedding", c_cnn)

    # Stage 3B: Handcrafted Features
    draw_box(6.3, 1.1, 2.5, 1.5, "Feature Extraction", "69-D Physical Descriptors\nColor, Texture, Defects, Shape", c_cv)

    # Stage 4: Feature Fusion
    draw_box(9.5, 2.7, 2.0, 1.6, "Feature Fusion & Head", "Concatenate (1344-D)\nDense(128, ReLU)\nSoftmax(Grade)", c_fusion)

    # Stage 5: Outputs
    draw_box(9.5, 0.4, 2.0, 1.4, "Deployment & GUI", "Streamlit Web UI\nShelf-life Advice", c_output)
    draw_box(9.5, 5.0, 2.0, 1.4, "Grading Output", "Grade A (Fresh)\nGrade B (Medium)\nGrade C (Rotten)", c_output)

    # Connecting Arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"))

    draw_arrow(2.7, 3.5, 3.2, 3.5)
    draw_arrow(5.8, 3.8, 6.3, 5.0)
    draw_arrow(5.8, 3.2, 6.3, 2.0)
    draw_arrow(9.0, 5.0, 9.5, 3.8)
    draw_arrow(9.0, 2.0, 9.5, 3.2)
    draw_arrow(10.5, 4.5, 10.5, 5.0)
    draw_arrow(10.5, 2.5, 10.5, 2.0)

    plt.title("Fruit Quality Grading System - End-to-End System Architecture",
              fontsize=14, fontweight="bold", pad=15)
    out_path = config.OUTPUT_DIR / "system_architecture_diagram.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("GENERATING PUBLICATION-QUALITY REPORT VISUALS & CHARTS")
    print("=" * 65)

    # 1. Preprocessing Pipeline Figure
    sample_imgs = list((config.DATA_DIR / "test" / "fresh").glob("*.jpg"))
    if sample_imgs:
        generate_preprocessing_pipeline_figure(sample_imgs[0])

    # 2. Feature Extraction Distributions
    generate_feature_analysis_figure()

    # 3. System Architecture Diagram
    generate_architecture_diagram()

    # 4. Sample Predictions Grid (if model is trained)
    if config.MODEL_PATH.exists():
        generate_sample_predictions_grid()

    print("\n" + "=" * 65)
    print("All report figures saved to 'outputs/' successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
