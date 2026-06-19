"""Hybrid MHSNet + Random Forest (semi-supervised cascade)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .mhsnet_detector import MHSNetFraudDetector


class HybridMHSNetRF:
  """Two-stage detector: MHSNet anomaly scores + supervised Random Forest.

  Stage 1 — MHSNet produces unsupervised anomaly signals (LOF + Hopfield).
  Stage 2 — Random Forest uses projected features enriched with those scores.
  """

  def __init__(self, random_state: int = 42) -> None:
    self.random_state = random_state
    self.detector: MHSNetFraudDetector | None = None
    self.rf = RandomForestClassifier(
      n_estimators=200,
      class_weight="balanced",
      random_state=random_state,
      n_jobs=-1,
    )
    self.threshold: float = 0.5

  def _enriched_features(self, X: np.ndarray) -> np.ndarray:
    if self.detector is None:
      raise RuntimeError("Model is not fitted.")
    projected = self.detector._transform_features(X)
    mhs = self.detector.anomaly_scores(X).reshape(-1, 1)
    lof = self.detector.lof_scores(X).reshape(-1, 1)
    hopfield = self.detector.hopfield_scores(X).reshape(-1, 1)
    return np.hstack([projected, mhs, lof, hopfield])

  def fit(
    self,
    detector: MHSNetFraudDetector,
    X_train: np.ndarray,
    y_train: np.ndarray,
  ) -> "HybridMHSNetRF":
    self.detector = detector
    X_enriched = self._enriched_features(X_train)
    self.rf.fit(X_enriched, y_train)
    return self

  def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
    scores = self.predict_scores(X_val)
    candidates = np.quantile(scores, np.linspace(0.40, 0.995, 80))
    best_threshold = float(np.median(scores))
    best_f1 = -1.0
    y_val = np.asarray(y_val, dtype=int)

    for threshold in candidates:
      y_pred = (scores > threshold).astype(int)
      if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
        continue
      tp = np.sum((y_val == 1) & (y_pred == 1))
      fp = np.sum((y_val == 0) & (y_pred == 1))
      fn = np.sum((y_val == 1) & (y_pred == 0))
      precision = tp / (tp + fp) if (tp + fp) else 0.0
      recall = tp / (tp + fn) if (tp + fn) else 0.0
      f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
      if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(threshold)

    self.threshold = best_threshold
    return self.threshold

  def predict_scores(self, X: np.ndarray) -> np.ndarray:
    return self.rf.predict_proba(self._enriched_features(X))[:, 1]

  def predict(self, X: np.ndarray) -> np.ndarray:
    scores = self.predict_scores(X)
    return (scores > self.threshold).astype(int)
