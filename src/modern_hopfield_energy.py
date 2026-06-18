"""Modern Hopfield energy for anomaly / outlier scoring.

Inspired by:
- Ramsauer et al. (2021) "Hopfield Networks is All You Need"
- Hofmann et al. (2024) "Hopfield Boosting for Out-of-Distribution Detection"
- Energy-based outlier detection ideas used in MHSNet-SNN (SSRN 5335578)

Low energy  -> transaction matches stored normal prototypes (in-distribution)
High energy -> likely fraud / outlier
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def modern_hopfield_energy(
  queries: np.ndarray,
  memory_patterns: np.ndarray,
  beta: float = 2.0,
) -> np.ndarray:
  """Compute Modern Hopfield Energy for each query vector.

  E(xi) = -logsumexp(beta * X @ xi) + 0.5 * ||xi||^2

  Args:
    queries: shape (n_samples, n_features)
    memory_patterns: shape (n_patterns, n_features), normal prototypes
    beta: inverse temperature, controls sharpness of associative retrieval
  """
  queries = np.asarray(queries, dtype=np.float64)
  memory_patterns = np.asarray(memory_patterns, dtype=np.float64)

  similarities = memory_patterns @ queries.T
  log_partition = logsumexp(beta * similarities, axis=0)
  query_norm = 0.5 * np.sum(queries ** 2, axis=1)
  return (-log_partition + query_norm).astype(np.float64)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
  """Map scores to [0, 1] using min-max on the provided batch."""
  scores = np.asarray(scores, dtype=np.float64)
  min_val = float(np.min(scores))
  max_val = float(np.max(scores))
  if max_val - min_val < 1e-12:
    return np.zeros_like(scores)
  return (scores - min_val) / (max_val - min_val)
