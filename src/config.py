"""Project configuration."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# computingvictor/transactions-fraud-datasets (Kaggle)
CAIXABANK_DIR = DATA_DIR / "caixabank"
KAGGLE_DATASET = "computingvictor/transactions-fraud-datasets"


@dataclass
class ExperimentConfig:
  """Hyperparameters for fraud detection on CaixaBank dataset."""

  caixabank_dir: Path = CAIXABANK_DIR

  # Sampling from 13M+ rows (~0.1% fraud in full dataset)
  max_train_normal: int = 80_000
  chunk_size: int = 250_000
  balanced_eval_per_class: int = 500

  random_state: int = 42

  # Classical Hopfield network
  max_stored_patterns: int = 64
  pattern_selection: str = "kmeans"  # "kmeans" | "random"
  max_recall_iterations: int = 20
  threshold_metric: str = "f1"

  # Modern Hopfield energy (MHSNet / Hopfield Boosting inspired)
  use_modern_energy: bool = True
  hopfield_beta: float = 2.0
  energy_weight: float = 0.45
  reconstruction_weight: float = 0.35
  distance_weight: float = 0.20

  # Kernel PCA + LOF pipeline (Zhao et al., SSRN 5335578)
  use_kernel_pca: bool = True
  kpca_components: int = 15
  kpca_kernel: str = "rbf"
  kpca_gamma: float | None = None
  kpca_fit_max_samples: int = 8_000

  use_lof: bool = True
  lof_neighbors: int = 20
  lof_contamination: float = 0.05

  pipeline_mode: str = "fusion"  # "fusion" | "cascade"
  lof_weight: float = 0.35
  hopfield_weight: float = 0.65

  # Output
  output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
  save_plots: bool = True
  save_models: bool = True

  def __post_init__(self) -> None:
    self.caixabank_dir = Path(self.caixabank_dir)
    self.output_dir = Path(self.output_dir)
