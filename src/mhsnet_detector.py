"""MHSNet fraud detector: Kernel PCA -> LOF -> Hopfield (Zhao et al., SSRN 5335578)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from .config import ExperimentConfig
from .feature_pipeline import KernelPCAExtractor
from .fraud_detector import HopfieldFraudDetector
from .modern_hopfield_energy import normalize_scores


class MHSNetFraudDetector:
  """Two-stage outlier detection pipeline from MHSNet-SNN paper.

  Stage 1 — Kernel PCA: nonlinear feature extraction.
  Stage 2 — LOF: local density outlier scoring on normal transactions.
  Stage 3 — Hopfield MHS: reconstruction + Modern Hopfield Energy on suspects.
  """

  def __init__(self, config: ExperimentConfig) -> None:
    self.config = config
    self.kpca: Optional[KernelPCAExtractor] = None
    self.lof: Optional[LocalOutlierFactor] = None
    self.hopfield = HopfieldFraudDetector(config)
    self.lof_gate_threshold: float = 1.0
    self.threshold: float = 0.5

  def _transform_features(self, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if self.config.use_kernel_pca and self.kpca is not None:
      return self.kpca.transform(X)
    return X

  def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MHSNetFraudDetector":
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]

    if len(X_normal) == 0:
      raise ValueError("Training set must contain normal (class 0) transactions")

    if self.config.use_kernel_pca:
      self.kpca = KernelPCAExtractor(
        n_components=self.config.kpca_components,
        kernel=self.config.kpca_kernel,
        gamma=self.config.kpca_gamma,
        fit_max_samples=self.config.kpca_fit_max_samples,
        random_state=self.config.random_state,
      )
      X_train_proj = self.kpca.fit_transform(X_train)
    else:
      X_train_proj = X_train

    if self.config.use_lof:
      self.lof = LocalOutlierFactor(
        n_neighbors=self.config.lof_neighbors,
        contamination=self.config.lof_contamination,
        novelty=True,
        n_jobs=-1,
      )
      self.lof.fit(X_train_proj[normal_mask])
    else:
      self.lof = None

    self.hopfield.fit(X_train_proj, y_train)
    return self

  def lof_scores(self, X: np.ndarray) -> np.ndarray:
    if self.lof is None:
      return np.zeros(len(X), dtype=np.float64)
    X_proj = self._transform_features(X)
    return (-self.lof.score_samples(X_proj)).astype(np.float64)

  def hopfield_scores(self, X: np.ndarray) -> np.ndarray:
    X_proj = self._transform_features(X)
    return self.hopfield.anomaly_scores(X_proj)

  def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
    lof_raw = self.lof_scores(X)
    hopfield_raw = self.hopfield_scores(X)
    lof_norm = normalize_scores(lof_raw)
    hopfield_norm = normalize_scores(hopfield_raw)

    if self.config.pipeline_mode == "cascade":
      gated = np.where(lof_norm >= self.lof_gate_threshold, hopfield_norm, 0.0)
      return (
        self.config.lof_weight * lof_norm
        + self.config.hopfield_weight * gated
      )

    return (
      self.config.lof_weight * lof_norm
      + self.config.hopfield_weight * hopfield_norm
    )

  def tune_thresholds(
    self,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1",
  ) -> tuple[float, float]:
    y_val = np.asarray(y_val, dtype=int)
    lof_norm = normalize_scores(self.lof_scores(X_val))

    if self.config.pipeline_mode == "cascade" and self.config.use_lof:
      lof_candidates = np.quantile(lof_norm, np.linspace(0.50, 0.95, 15))
      best_lof_gate = float(lof_candidates[0])
      best_score = -1.0
      for gate in lof_candidates:
        self.lof_gate_threshold = float(gate)
        score = self._best_threshold_score(X_val, y_val, metric)
        if score > best_score:
          best_score = score
          best_lof_gate = float(gate)
      self.lof_gate_threshold = best_lof_gate
    else:
      self.lof_gate_threshold = float(np.quantile(lof_norm, 0.75))

    self.threshold = self._tune_final_threshold(X_val, y_val, metric)
    return self.lof_gate_threshold, self.threshold

  def _best_threshold_score(
    self,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str,
  ) -> float:
    scores = self.anomaly_scores(X_val)
    candidates = np.quantile(scores, np.linspace(0.40, 0.995, 80))
    best = -1.0
    for threshold in candidates:
      y_pred = (scores > threshold).astype(int)
      if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
        continue
      best = max(best, self._metric_score(y_val, y_pred, metric))
    return best

  def _tune_final_threshold(
    self,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str,
  ) -> float:
    scores = self.anomaly_scores(X_val)
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
    return best_threshold

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
