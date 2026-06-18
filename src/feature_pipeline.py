"""Kernel PCA feature extraction (Zhao et al., MHSNet-SNN pipeline)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import KernelPCA


class KernelPCAExtractor:
  """Nonlinear dimensionality reduction before LOF and Hopfield stages."""

  def __init__(
    self,
    n_components: int = 15,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
    fit_max_samples: int = 8_000,
    random_state: int = 42,
  ) -> None:
    self.n_components = n_components
    self.kernel = kernel
    self.gamma = gamma
    self.fit_max_samples = fit_max_samples
    self.random_state = random_state
    self.model: Optional[KernelPCA] = None

  def fit(self, X: np.ndarray) -> "KernelPCAExtractor":
    X = np.asarray(X, dtype=np.float64)
    if len(X) > self.fit_max_samples:
      rng = np.random.default_rng(self.random_state)
      indices = rng.choice(len(X), size=self.fit_max_samples, replace=False)
      X_fit = X[indices]
    else:
      X_fit = X
    n_components = min(self.n_components, X.shape[1], max(1, X_fit.shape[0] - 1))

    self.model = KernelPCA(
      n_components=n_components,
      kernel=self.kernel,
      gamma=self.gamma,
      fit_inverse_transform=False,
    )
    self.model.fit(X_fit)
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    if self.model is None:
      raise RuntimeError("KernelPCAExtractor is not fitted.")
    return self.model.transform(np.asarray(X, dtype=np.float64))

  def fit_transform(self, X: np.ndarray) -> np.ndarray:
    return self.fit(X).transform(X)
