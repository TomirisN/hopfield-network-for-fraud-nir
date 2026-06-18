"""Classical binary Hopfield network implementation."""

from __future__ import annotations

import numpy as np


class HopfieldNetwork:
  """Auto-associative Hopfield network with Hebbian learning.

  States are binary vectors in {-1, +1}. The network stores patterns as
  attractors of the energy landscape and can recover them from noisy input.
  """

  def __init__(self, n_neurons: int) -> None:
    if n_neurons <= 0:
      raise ValueError("n_neurons must be positive")
    self.n_neurons = n_neurons
    self.weights = np.zeros((n_neurons, n_neurons), dtype=np.float64)

  def train(self, patterns: np.ndarray) -> None:
    """Train with outer-product (Hebbian) rule.

    Args:
      patterns: array of shape (n_patterns, n_neurons) with values in {-1, +1}.
    """
    patterns = np.asarray(patterns, dtype=np.float64)
    if patterns.ndim != 2 or patterns.shape[1] != self.n_neurons:
      raise ValueError(
        f"Expected patterns with shape (n, {self.n_neurons}), got {patterns.shape}"
      )
    if len(patterns) == 0:
      raise ValueError("At least one pattern is required for training")

    self.weights = np.zeros((self.n_neurons, self.n_neurons), dtype=np.float64)
    for pattern in patterns:
      self.weights += np.outer(pattern, pattern)
    self.weights /= len(patterns)
    np.fill_diagonal(self.weights, 0.0)

  def recall(self, pattern: np.ndarray, max_iterations: int = 20) -> np.ndarray:
    """Recover a stored pattern by asynchronous-like full update."""
    state = np.asarray(pattern, dtype=np.float64).copy()
    for _ in range(max_iterations):
      previous = state.copy()
      activations = self.weights @ state
      state = np.sign(activations)
      state[state == 0] = previous[state == 0]
      if np.array_equal(state, previous):
        break
    return state

  def energy(self, pattern: np.ndarray) -> float:
    """Compute Hopfield energy for a state vector."""
    pattern = np.asarray(pattern, dtype=np.float64)
    return float(-0.5 * pattern @ self.weights @ pattern)

  def reconstruction_error(self, original: np.ndarray, recovered: np.ndarray) -> float:
    """Normalized Hamming distance between original and recovered states."""
    original = np.asarray(original, dtype=np.float64)
    recovered = np.asarray(recovered, dtype=np.float64)
    mismatches = np.sum(original != recovered)
    return mismatches / self.n_neurons
