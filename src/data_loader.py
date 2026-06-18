"""Dataset loading for CaixaBank / computingvictor Kaggle dataset."""

from __future__ import annotations

import numpy as np

from .caixabank_loader import prepare_caixabank_data
from .config import ExperimentConfig


def to_binary_patterns(X_scaled: np.ndarray) -> np.ndarray:
  """Map standardized features to bipolar states {-1, +1} for Hopfield network."""
  return np.where(X_scaled >= 0, 1.0, -1.0)


def prepare_data(config: ExperimentConfig) -> dict:
  """Load and preprocess the CaixaBank transactions fraud dataset."""
  return prepare_caixabank_data(config)
