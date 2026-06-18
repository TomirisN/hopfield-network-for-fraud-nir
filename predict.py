"""Run inference with a saved MHSNet detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Predict fraud using saved MHSNet model.")
  parser.add_argument(
    "--model-dir",
    type=Path,
    default=Path("outputs/caixabank"),
    help="Directory with mhsnet_detector.joblib",
  )
  parser.add_argument(
    "--input",
    type=Path,
    required=True,
    help="CSV with raw feature matrix (same columns as training, before KPCA).",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Optional path to save predictions CSV.",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  model_path = args.model_dir / "mhsnet_detector.joblib"
  legacy_path = args.model_dir / "hopfield_detector.joblib"

  if model_path.exists():
    detector = joblib.load(model_path)
  elif legacy_path.exists():
    detector = joblib.load(legacy_path)
    print("Warning: using legacy hopfield_detector.joblib (without KPCA+LOF pipeline).")
  else:
    raise FileNotFoundError(
      f"No model found in {args.model_dir}. Run: python run.py"
    )

  df = pd.read_csv(args.input)
  X = df.to_numpy(dtype=float)
  predictions = detector.predict(X)
  scores = detector.predict_scores(X)

  result = df.copy()
  result["fraud_prediction"] = predictions
  result["fraud_score"] = scores
  if hasattr(detector, "lof_scores"):
    result["lof_score"] = detector.lof_scores(X)
  if hasattr(detector, "hopfield_scores"):
    result["hopfield_score"] = detector.hopfield_scores(X)

  print(result[["fraud_prediction", "fraud_score"]].head(10).to_string(index=False))
  print(f"\nPredicted fraud transactions: {int(predictions.sum())} / {len(predictions)}")

  if args.output:
    result.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
