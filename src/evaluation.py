"""Metrics, reporting, and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
  accuracy_score,
  confusion_matrix,
  f1_score,
  precision_score,
  recall_score,
  roc_auc_score,
  roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
  y_true = np.asarray(y_true, dtype=int)
  y_pred = np.asarray(y_pred, dtype=int)
  y_score = np.asarray(y_score, dtype=float)

  metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
  }
  if len(np.unique(y_true)) > 1:
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    metrics["recall_at_fpr_5"] = float(recall_at_fpr(y_true, y_score, target_fpr=0.05))
  else:
    metrics["roc_auc"] = float("nan")
    metrics["recall_at_fpr_5"] = float("nan")
  return metrics


def recall_at_fpr(
  y_true: np.ndarray,
  y_score: np.ndarray,
  target_fpr: float = 0.05,
) -> float:
  """Recall when false positive rate is capped (fraud-industry standard metric)."""
  y_true = np.asarray(y_true, dtype=int)
  y_score = np.asarray(y_score, dtype=float)
  fpr, tpr, _ = roc_curve(y_true, y_score)
  valid = fpr <= target_fpr
  if not np.any(valid):
    return 0.0
  return float(np.max(tpr[valid]))


def metrics_to_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
  df = pd.DataFrame(results).T
  columns = ["accuracy", "precision", "recall", "f1", "roc_auc", "recall_at_fpr_5"]
  return df[[col for col in columns if col in df.columns]]


def print_metrics_table(results: Dict[str, Dict[str, float]]) -> None:
  df = metrics_to_dataframe(results)
  print("\n=== Model comparison ===")
  print(df.round(4).to_string())


def save_metrics(results: Dict[str, Dict[str, float]], output_dir: Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  df = metrics_to_dataframe(results)
  df.to_csv(output_dir / "metrics.csv")
  with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, ensure_ascii=False)


def plot_confusion_matrix(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  title: str,
  output_path: Path,
) -> None:
  matrix = confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(5, 4))
  sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legit", "Fraud"],
    yticklabels=["Legit", "Fraud"],
  )
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title(title)
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()


def plot_roc_curves(
  y_true: np.ndarray,
  model_scores: Dict[str, np.ndarray],
  output_path: Path,
) -> None:
  plt.figure(figsize=(7, 5))
  for name, scores in model_scores.items():
    if len(np.unique(y_true)) < 2:
      continue
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

  plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC curves")
  plt.legend()
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()


def plot_metric_bars(results: Dict[str, Dict[str, float]], output_path: Path) -> None:
  df = metrics_to_dataframe(results)[["precision", "recall", "f1"]]
  ax = df.plot(kind="bar", figsize=(9, 5), rot=0)
  ax.set_ylabel("Score")
  ax.set_ylim(0, 1.05)
  ax.set_title("Model metrics comparison")
  ax.legend(loc="lower right")
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()


def plot_error_distribution(
  errors_normal: np.ndarray,
  errors_fraud: np.ndarray,
  threshold: float,
  output_path: Path,
) -> None:
  plt.figure(figsize=(8, 5))
  plt.hist(errors_normal, bins=40, alpha=0.7, label="Legitimate", density=True)
  plt.hist(errors_fraud, bins=40, alpha=0.7, label="Fraud", density=True)
  plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.3f}")
  plt.xlabel("Anomaly score")
  plt.ylabel("Density")
  plt.title("MHSNet anomaly score distribution")
  plt.legend()
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()


def generate_report(
  results: Dict[str, Dict[str, float]],
  y_test: np.ndarray,
  predictions: Dict[str, np.ndarray],
  scores: Dict[str, np.ndarray],
  hopfield_threshold: float,
  hopfield_errors: np.ndarray,
  output_dir: Path,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  save_metrics(results, output_dir)

  plot_metric_bars(results, output_dir / "metrics_comparison.png")
  plot_roc_curves(y_test, scores, output_dir / "roc_curves.png")

  for model_name, y_pred in predictions.items():
    plot_confusion_matrix(
      y_test,
      y_pred,
      title=f"Confusion matrix: {model_name}",
      output_path=output_dir / f"confusion_{model_name}.png",
    )

  plot_error_distribution(
    hopfield_errors[y_test == 0],
    hopfield_errors[y_test == 1],
    hopfield_threshold,
    output_dir / "hopfield_error_distribution.png",
  )

  metrics_df = metrics_to_dataframe(results).round(4)
  metrics_table = metrics_df.to_string()

  lines = [
    "# Experiment report",
    "",
    "## Metrics",
    "",
    "```",
    metrics_table,
    "```",
    "",
    f"Hopfield threshold: `{hopfield_threshold:.6f}`",
    "",
    "## Files",
    "- `metrics.csv` / `metrics.json`",
    "- `metrics_comparison.png`",
    "- `roc_curves.png`",
    "- `hopfield_error_distribution.png`",
    "- `confusion_<model>.png`",
  ]
  (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
