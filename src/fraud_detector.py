"""Hopfield-based fraud detector with energy-based outlier scoring."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from .config import ExperimentConfig
from .data_loader import to_binary_patterns
from .hopfield_network import HopfieldNetwork
from .modern_hopfield_energy import modern_hopfield_energy, normalize_scores


class HopfieldFraudDetector:
  """Unsupervised / semi-supervised fraud detector.

  Pipeline:
  1. Store normal transaction prototypes in a classical Hopfield network.
  2. Score anomalies via reconstruction error + Modern Hopfield Energy.
  3. Tune threshold on validation split (MHSNet-SNN / Hopfield Boosting idea:
     energy separates in-distribution normals from outlier fraud cases).
  """

  def __init__(self, config: ExperimentConfig) -> None:
    self.config = config
    self.network: Optional[HopfieldNetwork] = None
    self.threshold: float = 0.5
    self.stored_patterns: Optional[np.ndarray] = None
    self.continuous_prototypes: Optional[np.ndarray] = None

  def _select_training_patterns(self, X_normal_binary: np.ndarray) -> np.ndarray:
    n_patterns = min(self.config.max_stored_patterns, len(X_normal_binary))
    if n_patterns <= 0:
      raise ValueError("No normal transactions available for training")

    if self.config.pattern_selection == "kmeans" and n_patterns < len(X_normal_binary):
      kmeans = KMeans(
        n_clusters=n_patterns,
        random_state=self.config.random_state,
        n_init=10,
      )
      kmeans.fit(X_normal_binary)
      patterns = np.sign(kmeans.cluster_centers_)
      patterns[patterns == 0] = 1.0
      return patterns

    rng = np.random.default_rng(self.config.random_state)
    indices = rng.choice(len(X_normal_binary), size=n_patterns, replace=False)
    return X_normal_binary[indices]

  def _select_continuous_prototypes(self, X_normal: np.ndarray) -> np.ndarray:
    n_patterns = min(self.config.max_stored_patterns, len(X_normal))
    if self.config.pattern_selection == "kmeans" and n_patterns < len(X_normal):
      kmeans = KMeans(
        n_clusters=n_patterns,
        random_state=self.config.random_state,
        n_init=10,
      )
      return kmeans.fit(X_normal).cluster_centers_
    rng = np.random.default_rng(self.config.random_state)
    indices = rng.choice(len(X_normal), size=n_patterns, replace=False)
    return X_normal[indices]

  def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "HopfieldFraudDetector":
    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]
    if len(X_normal) == 0:
      raise ValueError("Training set must contain normal (class 0) transactions")

    X_normal_binary = to_binary_patterns(X_normal)
    self.stored_patterns = self._select_training_patterns(X_normal_binary)
    self.continuous_prototypes = self._select_continuous_prototypes(X_normal)

    self.network = HopfieldNetwork(n_neurons=X_train.shape[1])
    self.network.train(self.stored_patterns)
    return self

  def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
    if self.network is None:
      raise RuntimeError("Detector is not fitted. Call fit() first.")

    X_binary = to_binary_patterns(X)
    errors = np.zeros(len(X_binary), dtype=np.float64)
    for i, pattern in enumerate(X_binary):
      recovered = self.network.recall(pattern, max_iterations=self.config.max_recall_iterations)
      errors[i] = self.network.reconstruction_error(pattern, recovered)
    return errors

  def min_pattern_distances(self, X: np.ndarray) -> np.ndarray:
    if self.stored_patterns is None or self.network is None:
      raise RuntimeError("Detector is not fitted. Call fit() first.")

    X_binary = to_binary_patterns(X)
    distances = np.zeros(len(X_binary), dtype=np.float64)
    for i, pattern in enumerate(X_binary):
      mismatches = np.sum(self.stored_patterns != pattern, axis=1)
      distances[i] = np.min(mismatches) / self.network.n_neurons
    return distances

  def energy_scores(self, X: np.ndarray) -> np.ndarray:
    if self.continuous_prototypes is None:
      raise RuntimeError("Detector is not fitted. Call fit() first.")
    return modern_hopfield_energy(
      queries=X,
      memory_patterns=self.continuous_prototypes,
      beta=self.config.hopfield_beta,
    )

  def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
    reconstruction = normalize_scores(self.reconstruction_errors(X))
    distance = normalize_scores(self.min_pattern_distances(X))

    if self.config.use_modern_energy:
      energy = normalize_scores(self.energy_scores(X))
      return (
        self.config.reconstruction_weight * reconstruction
        + self.config.distance_weight * distance
        + self.config.energy_weight * energy
      )

    total = self.config.reconstruction_weight + self.config.distance_weight
    return (
      (self.config.reconstruction_weight / total) * reconstruction
      + (self.config.distance_weight / total) * distance
    )

  def tune_threshold(
    self,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1",
  ) -> float:
    scores = self.anomaly_scores(X_val)
    y_val = np.asarray(y_val, dtype=int)

    candidates = np.quantile(scores, np.linspace(0.40, 0.995, 120))
    best_threshold = float(np.median(scores))
    best_score = -1.0

    for threshold in candidates:
      y_pred = (scores > threshold).astype(int)
      if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
        continue
      score = self._metric_score(y_val, y_pred, metric)
      if score > best_score:
        best_score = score
        best_threshold = float(threshold)

    self.threshold = best_threshold
    return self.threshold

  def predict(self, X: np.ndarray) -> np.ndarray:
    scores = self.anomaly_scores(X)
    return (scores > self.threshold).astype(int)

  def predict_scores(self, X: np.ndarray) -> np.ndarray:
    return self.anomaly_scores(X)

  @staticmethod
  def _metric_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if metric == "precision":
      return precision
    if metric == "recall":
      return recall
    if metric == "f1":
      if precision + recall == 0:
        return 0.0
      return 2 * precision * recall / (precision + recall)
    raise ValueError(f"Unsupported metric: {metric}")
