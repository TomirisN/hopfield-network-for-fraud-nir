"""Baseline models for comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class BaselineResult:
  name: str
  y_pred: np.ndarray
  y_score: np.ndarray


def train_baselines(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> Dict[str, object]:
  """Fit baseline models on the training split."""
  models = {
    "logistic_regression": LogisticRegression(
      max_iter=1000,
      class_weight="balanced",
      random_state=random_state,
    ),
    "random_forest": RandomForestClassifier(
      n_estimators=100,
      class_weight="balanced",
      random_state=random_state,
      n_jobs=-1,
    ),
    "isolation_forest": IsolationForest(
      n_estimators=200,
      contamination=max(float(np.mean(y_train == 1)), 0.001),
      random_state=random_state,
      n_jobs=-1,
    ),
  }

  for name, model in models.items():
    if name == "isolation_forest":
      model.fit(X_train[y_train == 0])
    else:
      model.fit(X_train, y_train)
  return models


def predict_baselines(models: Dict[str, object], X_test: np.ndarray) -> list[BaselineResult]:
  """Generate predictions and scores for all baselines."""
  results: list[BaselineResult] = []

  lr = models["logistic_regression"]
  y_score = lr.predict_proba(X_test)[:, 1]
  results.append(
    BaselineResult(
      name="logistic_regression",
      y_pred=lr.predict(X_test).astype(int),
      y_score=y_score,
    )
  )

  rf = models["random_forest"]
  y_score = rf.predict_proba(X_test)[:, 1]
  results.append(
    BaselineResult(
      name="random_forest",
      y_pred=rf.predict(X_test).astype(int),
      y_score=y_score,
    )
  )

  iso = models["isolation_forest"]
  iso_pred = iso.predict(X_test)
  y_pred = np.where(iso_pred == -1, 1, 0)
  y_score = -iso.score_samples(X_test)
  results.append(
    BaselineResult(
      name="isolation_forest",
      y_pred=y_pred.astype(int),
      y_score=y_score,
    )
  )

  return results
