"""Download CaixaBank / computingvictor fraud dataset from Kaggle."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CAIXABANK_DIR, KAGGLE_DATASET


REQUIRED_FILES = (
  "transactions_data.csv",
  "cards_data.csv",
  "train_fraud_labels.json",
)


def download_dataset(target_dir: Path | None = None) -> Path:
  """Download dataset and copy required files into project data folder."""
  target_dir = Path(target_dir or CAIXABANK_DIR)
  target_dir.mkdir(parents=True, exist_ok=True)

  cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
  copied = []
  for filename in REQUIRED_FILES:
    source = cache_path / filename
    if not source.exists():
      nested = cache_path / "gd_card_flaud_demo" / filename
      source = nested if nested.exists() else source
    if not source.exists():
      raise FileNotFoundError(f"Expected file missing in Kaggle archive: {filename}")

    destination = target_dir / filename
    if not destination.exists() or destination.stat().st_size != source.stat().st_size:
      shutil.copy2(source, destination)
    copied.append(destination.name)

  print(f"Dataset ready in: {target_dir}")
  print(f"Source: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
  print("Files:", ", ".join(copied))
  return target_dir


if __name__ == "__main__":
  download_dataset()
