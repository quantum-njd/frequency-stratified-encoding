#!/usr/bin/env python3
"""Regenerate all numerical results in a fixed order."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "experiment_common.py",
    "generate_exact_population.py",
    "generate_resolution_persistence.py",
    "generate_internal_frequency_allocation.py",
    "generate_resolution_sensitivity.py",
    "generate_resolved_population.py",
    "generate_radial_profile_sensitivity.py",
    "generate_coefficient_realization.py",
    "generate_training_check.py",
)


def main() -> None:
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", str(ROOT / "mplconfig"))
    for script in SCRIPTS:
        print(f"\n=== {script} ===", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / script)],
                       check=True, cwd=ROOT, env=environment)


if __name__ == "__main__":
    main()
