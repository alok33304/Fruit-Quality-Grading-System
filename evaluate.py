"""Comprehensive Model Evaluation and Benchmark Suite.

Evaluates trained models on the held-out test split (602 real-world fruit images).
Outputs:
  - outputs/metrics.json: Quantitative evaluation metrics.
  - outputs/confusion_matrix.png: Normalized and count confusion matrices.
  - outputs/roc_curves.png: Multi-class One-vs-Rest ROC curves with AUC.
  - outputs/evaluation_summary.txt: Publication-ready metrics table.
"""
import argparse
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

import config
from dataset import load_split


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fruit Quality Grading System")
    parser.add_argument("--data_dir", default=str(config.DATA_DIR), help="Dataset root directory")
    args = parser.parse_args()

    print("=" * 65)
    print("FRUIT QUALITY GRADING SYSTEM - EVALUATION SUITE")
    print("=" * 65)

    # 1. Load Classes, Configuration, and Test Data
    classes = json.load(open(config.CLASSES_PATH))
    cfg = json.load(open(config.MODEL_CFG_PATH))
    print(f"Loading test split from {args.data_dir}/test ...")
    X_te, F_te, y_te, _ = load_split(f"{args.data_dir}/test", classes=classes)

    scaler = joblib.load(config.SCALER_PATH)
    F_te_scaled = scaler.transform(F_te)
    model = tf.keras.models.load_model(config.MODEL_PATH)

    # 2. Deep Learning Predictions
    test_inputs = [X_te, F_te_scaled] if cfg["use_features"] else X_te
    print("Running inference on test dataset ...")
    proba = model.predict(test_inputs, verbose=0)
    pred = proba.argmax(axis=1)

    # 3. Compute Metrics
    dl_acc = accuracy_score(y_te, pred)
    dl_p_macro, dl_r_macro, dl_f1_macro, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    dl_p_wt, dl_r_wt, dl_f1_wt, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print("PROPOSED HYBRID DEEP MODEL RESULTS (TEST SPLIT)")
    print("=" * 50)
    print(f"Overall Accuracy:  {dl_acc * 100:.2f}%")
    print(f"Macro Precision:   {dl_p_macro * 100:.2f}%")
    print(f"Macro Recall:      {dl_r_macro * 100:.2f}%")
    print(f"Macro F1-Score:    {dl_f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score: {dl_f1_wt * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_te, pred, target_names=[config.GRADE_LABELS.get(c, c) for c in classes], digits=4))

    metrics = {
        "deep_hybrid_model": {
            "accuracy": float(dl_acc),
            "macro_precision": float(dl_p_macro),
            "macro_recall": float(dl_r_macro),
            "macro_f1": float(dl_f1_macro),
            "weighted_f1": float(dl_f1_wt),
            "sample_count": len(y_te)
        }
    }

    # 4. Classical Baseline (Random Forest)
    if config.RF_PATH.exists():
        rf = joblib.load(config.RF_PATH)
        rf_pred = rf.predict(F_te)
        rf_acc = accuracy_score(y_te, rf_pred)
        rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(y_te, rf_pred, average="macro", zero_division=0)
        print("\n" + "=" * 50)
        print("CLASSICAL BASELINE (RANDOM FOREST ON OPENCV FEATURES)")
        print("=" * 50)
        print(f"Baseline Accuracy: {rf_acc * 100:.2f}%")
        print(f"Baseline Macro F1: {rf_f1 * 100:.2f}%")
        metrics["rf_baseline"] = {
            "accuracy": float(rf_acc),
            "macro_precision": float(rf_p),
            "macro_recall": float(rf_r),
            "macro_f1": float(rf_f1)
        }

    # 5. Plot Confusion Matrix
    cm = confusion_matrix(y_te, pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax1,
                annot_kws={"size": 12, "weight": "bold"})
    ax1.set_title("Confusion Matrix (Sample Counts)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Predicted Grade", fontsize=11)
    ax1.set_ylabel("True Grade", fontsize=11)

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax2,
                annot_kws={"size": 12, "weight": "bold"})
    ax2.set_title("Normalized Confusion Matrix (Recall %)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Predicted Grade", fontsize=11)
    ax2.set_ylabel("True Grade", fontsize=11)

    plt.tight_layout()
    cm_path = config.OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"\nSaved confusion matrix plot -> {cm_path}")

    # 6. Plot Multi-class ROC Curves
    plt.figure(figsize=(7, 6), dpi=150)
    if len(classes) > 2:
        y_bin = label_binarize(y_te, classes=range(len(classes)))
        for i, cls_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2.2, label=f"{cls_name.capitalize()} (AUC = {roc_auc:.3f})")
    else:
        fpr, tpr, _ = roc_curve(y_te, proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.2, label=f"AUC = {roc_auc:.3f}")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    roc_path = config.OUTPUT_DIR / "roc_curves.png"
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curves plot -> {roc_path}")

    # 7. Save JSON and Formatted Text Summary
    json_path = config.OUTPUT_DIR / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    summary_path = config.OUTPUT_DIR / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("FRUIT QUALITY GRADING SYSTEM - EVALUATION BENCHMARK\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Samples: {len(y_te)} verified real-world images\n")
        f.write(f"Classes:      {classes}\n\n")
        f.write("-" * 60 + "\n")
        f.write("MODEL COMPARISON TABLE\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<35} | {'Accuracy':<10} | {'Macro F1':<10}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Random Forest (OpenCV Features)':<35} | {metrics.get('rf_baseline', {}).get('accuracy', 0)*100:6.2f}%    | {metrics.get('rf_baseline', {}).get('macro_f1', 0)*100:6.2f}%\n")
        f.write(f"{'Hybrid MobileNetV2 + Feature Fusion':<35} | {dl_acc*100:6.2f}%    | {dl_f1_macro*100:6.2f}%\n")
        f.write("-" * 60 + "\n")

    print(f"Saved evaluation metrics JSON -> {json_path}")
    print(f"Saved text report summary -> {summary_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
