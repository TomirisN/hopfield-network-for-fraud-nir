"""Main experiment runner for CaixaBank fraud detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

from src.baselines import predict_baselines, train_baselines
from src.config import CAIXABANK_DIR, ExperimentConfig, KAGGLE_DATASET
from src.data_loader import prepare_data
from src.evaluation import compute_metrics, generate_report, print_metrics_table
from src.mhsnet_detector import MHSNetFraudDetector


REQUIRED_FILES = (
  "transactions_data.csv",
  "cards_data.csv",
  "train_fraud_labels.json",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "MHSNet fraud detection on CaixaBank dataset "
      f"(Kaggle: {KAGGLE_DATASET}). Pipeline: Kernel PCA -> LOF -> Hopfield."
    ),
  )
  parser.add_argument(
    "--data-dir",
    type=Path,
    default=None,
    help=f"Path to dataset directory (default: {CAIXABANK_DIR}).",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Directory for metrics and plots (default: outputs/).",
  )
  parser.add_argument(
    "--patterns",
    type=int,
    default=64,
    help="Maximum number of patterns stored in Hopfield network.",
  )
  parser.add_argument(
    "--max-normal",
    type=int,
    default=80_000,
    help="Max normal transactions to load (all fraud transactions are kept).",
  )
  parser.add_argument(
    "--eval-per-class",
    type=int,
    default=500,
    help="Balanced validation/test size per class.",
  )
  parser.add_argument(
    "--kpca-components",
    type=int,
    default=15,
    help="Kernel PCA output dimensions.",
  )
  parser.add_argument(
    "--lof-neighbors",
    type=int,
    default=20,
    help="LOF k-neighbors parameter.",
  )
  parser.add_argument(
    "--pipeline-mode",
    choices=["fusion", "cascade"],
    default="fusion",
    help="fusion = weighted LOF+Hopfield; cascade = LOF gate then Hopfield.",
  )
  parser.add_argument(
    "--no-kpca",
    action="store_true",
    help="Disable Kernel PCA stage.",
  )
  parser.add_argument(
    "--no-lof",
    action="store_true",
    help="Disable LOF stage.",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed.",
  )
  return parser.parse_args()


def _check_dataset(data_dir: Path) -> bool:
  missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
  if missing:
    print(
      f"Dataset not found in: {data_dir}\n"
      f"Missing files: {', '.join(missing)}\n\n"
      "Download the dataset:\n"
      "  python scripts/download_dataset.py\n\n"
      f"Source: https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
      file=sys.stderr,
    )
    return False
  return True


def _project_features(detector: MHSNetFraudDetector, X) -> object:
  return detector._transform_features(X)


def run_experiment(config: ExperimentConfig) -> int:
  print("=" * 60)
  print("MHSNet Fraud Detection — CaixaBank Dataset")
  print("Kernel PCA -> LOF -> Hopfield (SSRN 5335578)")
  print("=" * 60)
  print(f"\nDataset: {KAGGLE_DATASET}")
  print(f"Path:    {config.caixabank_dir}")

  print("\n[1/6] Loading and preprocessing data...")
  data = prepare_data(config)
  metadata = data.get("metadata", {})
  if metadata:
    print(f"  Loaded rows: {metadata.get('total_loaded')}")
    print(f"  Fraud in sample: {metadata.get('fraud_loaded')} ({metadata.get('fraud_rate_loaded', 0):.4%})")
    print(f"  Labeled fraud in source: {metadata.get('labeled_fraud_in_source')}")

  print(f"  Features: {len(data['feature_names'])}")
  print(f"  Train size: {len(data['y_train'])} (fraud: {int(data['y_train'].sum())})")
  print(f"  Val size:   {len(data['y_val'])} (fraud: {int(data['y_val'].sum())})")
  print(f"  Test size:  {len(data['y_test'])} (fraud: {int(data['y_test'].sum())})")

  print("\n[2/6] Kernel PCA feature extraction...")
  print(f"  Enabled: {config.use_kernel_pca}")
  if config.use_kernel_pca:
    print(f"  Components: {config.kpca_components}, kernel: {config.kpca_kernel}")

  print("\n[3/6] Training LOF + Hopfield (MHSNet)...")
  detector = MHSNetFraudDetector(config)
  detector.fit(data["X_train"], data["y_train"])
  lof_gate, threshold = detector.tune_thresholds(
    data["X_val"],
    data["y_val"],
    metric=config.threshold_metric,
  )
  print(f"  LOF enabled: {config.use_lof}, neighbors: {config.lof_neighbors}")
  print(f"  Pipeline mode: {config.pipeline_mode}")
  print(f"  LOF gate threshold: {lof_gate:.4f}")
  print(f"  Final threshold: {threshold:.4f}")
  print(f"  Hopfield patterns: {len(detector.hopfield.stored_patterns)}")

  if config.use_kernel_pca and detector.kpca is not None:
    sample_proj = detector.kpca.transform(data["X_train"][:5])
    print(f"  KPCA output dim: {sample_proj.shape[1]}")

  mhs_pred = detector.predict(data["X_test"])
  mhs_scores = detector.predict_scores(data["X_test"])

  print("\n[4/6] Training baseline models on same feature space...")
  X_train_bl = _project_features(detector, data["X_train"])
  X_test_bl = _project_features(detector, data["X_test"])
  baselines = train_baselines(
    X_train_bl,
    data["y_train"],
    random_state=config.random_state,
  )
  baseline_results = predict_baselines(baselines, X_test_bl)

  print("\n[5/6] Evaluating models...")
  all_results: dict[str, dict[str, float]] = {}
  predictions: dict[str, object] = {"mhsnet": mhs_pred}
  scores: dict[str, object] = {"mhsnet": mhs_scores}

  all_results["mhsnet"] = compute_metrics(
    data["y_test"],
    mhs_pred,
    mhs_scores,
  )

  for result in baseline_results:
    all_results[result.name] = compute_metrics(
      data["y_test"],
      result.y_pred,
      result.y_score,
    )
    predictions[result.name] = result.y_pred
    scores[result.name] = result.y_score

  print_metrics_table(all_results)

  print("\n[6/6] Saving outputs...")
  config.output_dir.mkdir(parents=True, exist_ok=True)

  if config.save_models:
    joblib.dump(detector, config.output_dir / "mhsnet_detector.joblib")
    joblib.dump(baselines, config.output_dir / "baseline_models.joblib")

  if config.save_plots:
    generate_report(
      results=all_results,
      y_test=data["y_test"],
      predictions=predictions,
      scores=scores,
      hopfield_threshold=detector.threshold,
      hopfield_errors=mhs_scores,
      output_dir=config.output_dir,
    )

  with open(config.output_dir / "experiment_metadata.json", "w", encoding="utf-8") as file:
    json.dump(
      {
        "kaggle_dataset": KAGGLE_DATASET,
        "data_dir": str(config.caixabank_dir),
        "features": len(data["feature_names"]),
        "metadata": metadata,
        "pipeline": {
          "kernel_pca": config.use_kernel_pca,
          "kpca_components": config.kpca_components,
          "lof": config.use_lof,
          "lof_neighbors": config.lof_neighbors,
          "pipeline_mode": config.pipeline_mode,
          "lof_gate_threshold": lof_gate,
          "final_threshold": threshold,
        },
      },
      file,
      indent=2,
      ensure_ascii=False,
    )

  print(f"\nDone. Results saved to: {config.output_dir.resolve()}")
  return 0


def main() -> int:
  args = parse_args()
  data_dir = args.data_dir if args.data_dir else CAIXABANK_DIR

  if not _check_dataset(data_dir):
    return 1

  config = ExperimentConfig(
    caixabank_dir=data_dir,
    output_dir=args.output if args.output else ExperimentConfig().output_dir,
    max_stored_patterns=args.patterns,
    max_train_normal=args.max_normal,
    balanced_eval_per_class=args.eval_per_class,
    kpca_components=args.kpca_components,
    lof_neighbors=args.lof_neighbors,
    pipeline_mode=args.pipeline_mode,
    use_kernel_pca=not args.no_kpca,
    use_lof=not args.no_lof,
    random_state=args.seed,
  )

  return run_experiment(config)


if __name__ == "__main__":
  raise SystemExit(main())
