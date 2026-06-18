"""Loader for computingvictor / CaixaBank transactions fraud dataset.

Kaggle: https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets

Files:
  - transactions_data.csv  (~13M transactions)
  - cards_data.csv         (card metadata)
  - train_fraud_labels.json (fraud labels)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ExperimentConfig


def _parse_money(value: object) -> float:
  if pd.isna(value):
    return 0.0
  if isinstance(value, (int, float, np.number)):
    return float(value)
  cleaned = re.sub(r"[^0-9.\-]", "", str(value))
  return float(cleaned) if cleaned else 0.0


def _load_fraud_labels(path: Path) -> dict[int, int]:
  with open(path, encoding="utf-8") as file:
    payload = json.load(file)
  target = payload.get("target", {})
  labels: dict[int, int] = {}
  for transaction_id, flag in target.items():
    labels[int(transaction_id)] = 1 if str(flag).lower() in {"yes", "1", "true"} else 0
  return labels


def _iter_transaction_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
  usecols = [
    "id",
    "date",
    "client_id",
    "card_id",
    "amount",
    "use_chip",
    "merchant_id",
    "errors",
  ]
  for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
    yield chunk[chunk["errors"].isna()].drop(columns=["errors"])


def _load_transactions_sample(
  transactions_path: Path,
  fraud_ids: set[int],
  max_normal: int,
  random_state: int,
  chunksize: int,
) -> pd.DataFrame:
  """Stream 13M+ rows and keep all fraud + a random normal subset."""
  rng = np.random.default_rng(random_state)
  fraud_frames: list[pd.DataFrame] = []
  normal_reservoir: list[pd.DataFrame] = []
  normal_count = 0

  for chunk in _iter_transaction_chunks(transactions_path, chunksize=chunksize):
    chunk_fraud = chunk[chunk["id"].isin(fraud_ids)]
    if len(chunk_fraud):
      fraud_frames.append(chunk_fraud)

    chunk_normal = chunk[~chunk["id"].isin(fraud_ids)]
    if len(chunk_normal) == 0:
      continue

    if normal_count + len(chunk_normal) <= max_normal:
      normal_reservoir.append(chunk_normal)
      normal_count += len(chunk_normal)
      continue

    remaining = max_normal - normal_count
    if remaining > 0:
      picked = chunk_normal.sample(n=remaining, random_state=int(rng.integers(1_000_000)))
      normal_reservoir.append(picked)
      normal_count += remaining

  if not fraud_frames and not normal_reservoir:
    raise ValueError("No transactions loaded. Check dataset files.")

  parts = fraud_frames + normal_reservoir
  return pd.concat(parts, ignore_index=True)


def _engineer_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
  frame = df.copy()
  frame["amount"] = frame["amount"].map(_parse_money)
  frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
  frame["hour"] = frame["date"].dt.hour.fillna(0).astype(int)
  frame["day_of_week"] = frame["date"].dt.dayofweek.fillna(0).astype(int)
  frame["month"] = frame["date"].dt.month.fillna(0).astype(int)
  frame["merchant_id"] = pd.to_numeric(frame["merchant_id"], errors="coerce").fillna(0)
  return frame.drop(columns=["date"])


def _prepare_cards(cards_path: Path) -> pd.DataFrame:
  cards = pd.read_csv(cards_path)
  cards["credit_limit"] = cards["credit_limit"].map(_parse_money)
  cards = cards.drop(
    columns=[
      "client_id",
      "card_number",
      "expires",
      "cvv",
      "acct_open_date",
    ],
    errors="ignore",
  )
  cards = cards.rename(columns={"id": "card_ref_id"})
  return cards


def build_feature_matrix(
  transactions: pd.DataFrame,
  cards: pd.DataFrame,
  labels: dict[int, int],
) -> tuple[pd.DataFrame, pd.Series]:
  tx = _engineer_transaction_features(transactions)
  tx["is_fraud"] = tx["id"].map(labels).fillna(0).astype(int)

  merged = tx.merge(cards, left_on="card_id", right_on="card_ref_id", how="left")
  merged = merged.drop(columns=["id", "client_id", "card_id", "card_ref_id"], errors="ignore")

  y = merged["is_fraud"].astype(int)
  X = merged.drop(columns=["is_fraud"])

  numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
  categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

  transformers = [("num", StandardScaler(), numeric_cols)]
  if categorical_cols:
    transformers.append(
      (
        "cat",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        categorical_cols,
      )
    )

  preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

  X_processed = preprocessor.fit_transform(X)
  columns = preprocessor.get_feature_names_out()
  return pd.DataFrame(X_processed, columns=columns), y


def _balanced_subset(
  X: pd.DataFrame,
  y: pd.Series,
  size_per_class: int,
  random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
  frames = []
  labels = []
  for class_id in (0, 1):
    mask = y == class_id
    subset = X.loc[mask].sample(
      n=min(size_per_class, mask.sum()),
      random_state=random_state,
    )
    frames.append(subset)
    labels.append(y.loc[subset.index])

  X_balanced = pd.concat(frames).sample(frac=1, random_state=random_state)
  y_balanced = pd.concat(labels).loc[X_balanced.index]
  return X_balanced, y_balanced


def prepare_caixabank_data(config: ExperimentConfig) -> dict:
  """Load, merge, sample, and split CaixaBank dataset."""
  data_dir = config.caixabank_dir
  transactions_path = data_dir / "transactions_data.csv"
  cards_path = data_dir / "cards_data.csv"
  labels_path = data_dir / "train_fraud_labels.json"

  for path in (transactions_path, cards_path, labels_path):
    if not path.exists():
      raise FileNotFoundError(
        f"Missing {path.name}. Run: python scripts/download_dataset.py"
      )

  labels = _load_fraud_labels(labels_path)
  fraud_ids = {transaction_id for transaction_id, flag in labels.items() if flag == 1}

  transactions = _load_transactions_sample(
    transactions_path=transactions_path,
    fraud_ids=fraud_ids,
    max_normal=config.max_train_normal,
    random_state=config.random_state,
    chunksize=config.chunk_size,
  )
  cards = _prepare_cards(cards_path)
  X, y = build_feature_matrix(transactions, cards, labels)

  X_train, X_holdout, y_train, y_holdout = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=config.random_state,
    stratify=y,
  )

  X_val, X_test, y_val, y_test = train_test_split(
    X_holdout,
    y_holdout,
    test_size=0.5,
    random_state=config.random_state,
    stratify=y_holdout,
  )

  if config.balanced_eval_per_class > 0:
    X_val, y_val = _balanced_subset(
      X_val,
      y_val,
      size_per_class=config.balanced_eval_per_class,
      random_state=config.random_state,
    )
    X_test, y_test = _balanced_subset(
      X_test,
      y_test,
      size_per_class=config.balanced_eval_per_class,
      random_state=config.random_state + 1,
    )

  return {
    "dataset_name": "caixabank",
    "feature_names": list(X.columns),
    "X_train": X_train.to_numpy(dtype=np.float64),
    "X_val": X_val.to_numpy(dtype=np.float64),
    "X_test": X_test.to_numpy(dtype=np.float64),
    "y_train": y_train.to_numpy(),
    "y_val": y_val.to_numpy(),
    "y_test": y_test.to_numpy(),
    "scaler": None,
    "metadata": {
      "total_loaded": int(len(X)),
      "fraud_loaded": int(y.sum()),
      "fraud_rate_loaded": float(y.mean()),
      "labeled_fraud_in_source": len(fraud_ids),
    },
  }
