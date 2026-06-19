"""Run ablation experiments for MHSNet pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ABLATIONS = [
  {
    "name": "full_pipeline",
    "args": [],
    "description": "Kernel PCA + LOF + Hopfield + hybrid",
  },
  {
    "name": "no_kpca",
    "args": ["--no-kpca"],
    "description": "Without Kernel PCA",
  },
  {
    "name": "no_lof",
    "args": ["--no-lof"],
    "description": "Without LOF",
  },
  {
    "name": "cascade_mode",
    "args": ["--pipeline-mode", "cascade"],
    "description": "Cascade LOF gate -> Hopfield",
  },
  {
    "name": "mhsnet_only",
    "args": ["--no-hybrid"],
    "description": "MHSNet without hybrid RF",
  },
]


def main() -> int:
  quick = "--quick" in sys.argv
  output_root = PROJECT_ROOT / "outputs" / "ablation"
  output_root.mkdir(parents=True, exist_ok=True)

  common = [
    sys.executable,
    str(PROJECT_ROOT / "run.py"),
    "--max-normal",
    "15000" if quick else "50000",
    "--eval-per-class",
    "150" if quick else "300",
  ]

  summary: dict[str, dict] = {}
  for item in ABLATIONS:
    out_dir = output_root / item["name"]
    cmd = common + item["args"] + ["--output", str(out_dir)]
    print("\n" + "=" * 60)
    print(f"Ablation: {item['name']} — {item['description']}")
    print(" ".join(cmd))
    print("=" * 60)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
      print(f"FAILED: {item['name']}", file=sys.stderr)
      continue

    meta_path = out_dir / "experiment_metadata.json"
    if meta_path.exists():
      payload = json.loads(meta_path.read_text(encoding="utf-8"))
      summary[item["name"]] = {
        "description": item["description"],
        "balanced": payload.get("metrics_balanced_test", {}),
        "natural": payload.get("metrics_natural_test", {}),
      }

  summary_path = output_root / "ablation_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
  print(f"\nAblation summary saved to: {summary_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
