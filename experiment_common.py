#!/usr/bin/env python3
"""Shared numerical definitions for the frequency-support experiments.

The code implements the one-dimensional candidate-support identity

    Omega = Delta^(oplus L0) oplus alpha_1 Delta ... oplus alpha_p Delta,

where ``p=L-L0``.  Positive schedules at fixed ``(L0,rho_R)`` obey

    sum(alpha_j) = rho_R * L - L0,

so their outer candidate-support radius is identical.  Only the allocation of
that common scale budget changes.

Toy example
-----------
For Delta={-1,0,1}, L=2, L0=1, and alpha_1=2,
Omega={-3,-2,-1,0,1,2,3}.  Run this file directly to verify the example and
the finite-resolution packing rule.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"
CACHE_DIR = ROOT / "cache"
MPL_DIR = ROOT / "mplconfig"

FIGURE_DPI = 600
L = 6
L0_VALUES = tuple(range(L + 1))
REACH_VALUES = (1.0, 2.0, 5.0, 10.0)
PRIMARY_EPSILON = 0.15
RESOLUTION_VALUES = (0.0, 0.05, 0.10, 0.15, 0.25)
RANDOM_SEED = 20260716
N_RANDOM = 24
SUPPORT_DECIMALS = 10
SUPPORT_REL_TOL = 2.0e-9
ALGORITHM_VERSION = 7


@dataclass(frozen=True)
class HamiltonianSpec:
    key: str
    label: str
    qubits: int
    eigenvalues: tuple[float, ...]

    @property
    def difference_set(self) -> np.ndarray:
        values = np.asarray(self.eigenvalues, dtype=float)
        differences = (values[:, None] - values[None, :]).ravel()
        return np.unique(np.round(differences, 14))

    @property
    def r0(self) -> float:
        return float(np.max(np.abs(self.difference_set)))


HAMILTONIANS = (
    HamiltonianSpec("HP", r"$H_{\mathrm{P}}$, $d=1$", 1, (-0.5, 0.5)),
    HamiltonianSpec("HC", r"$H_{\mathrm{C}}$, $d=2$", 2, (-0.5, 0.0, 0.5)),
    HamiltonianSpec("HI", r"$H_{\mathrm{I}}$, $d=2$", 2,
                    (-0.5, -1.0 / 3.0, 0.0, 0.5)),
)
H_BY_KEY = {item.key: item for item in HAMILTONIANS}

STRUCTURED = ("constant", "linear", "exponential")
LABELS = {"constant": "constant", "linear": "linear",
          "exponential": "exponential", "random": "random reference"}
COLORS = {"constant": "#2878B5", "linear": "#F28E2B",
          "exponential": "#D84A3A", "random": "#202020"}


def ensure_dirs() -> None:
    for path in (DATA_DIR, FIGURE_DIR, LOG_DIR, CACHE_DIR, MPL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_output_directories() -> None:
    """Compatibility alias used by the circuit-level scripts."""
    ensure_dirs()


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.2,
        "axes.labelsize": 12.6, "axes.titlesize": 13.0,
        "xtick.labelsize": 10.4, "ytick.labelsize": 10.4,
        "legend.fontsize": 11.2, "axes.linewidth": 1.0,
        "lines.linewidth": 2.0, "lines.markersize": 4.5,
        "savefig.dpi": FIGURE_DPI,
    })


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(0.02, 0.98, label, transform=axis.transAxes, ha="left",
              va="top", fontsize=13.2, fontweight="bold")


def active_raw_profile(name: str, p: int) -> np.ndarray:
    """Return a profile on active-layer indices j=1,...,p.

    No normalized depth or contrast matching is introduced.  Changing L0
    changes p, which is part of the mechanism being tested.
    """
    if p < 1:
        return np.empty(0)
    j = np.arange(1, p + 1, dtype=float)
    if name == "constant":
        return np.ones(p)
    if name == "linear":
        return j
    if name == "exponential":
        return 2.0 ** (j - 1.0)
    raise KeyError(name)


def random_weights_by_p() -> dict[int, list[np.ndarray]]:
    """Sample fixed positive allocations independently for each active depth.

    Samples are drawn uniformly from the positive simplex using Dirichlet(1).
    We sort each vector because permuting scaled copies of the same difference
    set leaves the candidate Minkowski sum unchanged.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    result: dict[int, list[np.ndarray]] = {}
    for p in range(1, L + 1):
        draws = rng.dirichlet(np.ones(p), size=N_RANDOM)
        result[p] = [np.sort(row) for row in draws]
    return result


RANDOM_WEIGHTS = random_weights_by_p()


def scale_budget(l0: int, rho: float) -> float:
    return rho * L - l0


def structured_scales(name: str, l0: int, rho: float) -> np.ndarray:
    p = L - l0
    if p == 0:
        return np.empty(0)
    raw = active_raw_profile(name, p)
    return scale_budget(l0, rho) * raw / np.sum(raw)


def random_scales(index: int, l0: int, rho: float) -> np.ndarray:
    p = L - l0
    if p == 0:
        return np.empty(0)
    return scale_budget(l0, rho) * RANDOM_WEIGHTS[p][index]


def _unique_normalized(values: np.ndarray,
                       relative_tolerance: float = SUPPORT_REL_TOL) -> np.ndarray:
    """Merge numerically equal values in final-radius-normalized units.

    Adjacent-value merging avoids bin-edge artifacts of decimal rounding.  The
    tolerance is far below every finite resolution used in the paper and is
    audited separately from the physical threshold epsilon_omega.
    """
    if values.size == 0:
        return values
    ordered = np.sort(np.asarray(values, dtype=float))
    keep = np.ones(len(ordered), dtype=bool)
    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be nonnegative")
    keep[1:] = np.diff(ordered) > relative_tolerance
    return ordered[keep]


def candidate_support(delta: np.ndarray, boundary: int,
                      scales: Sequence[float],
                      relative_tolerance: float = SUPPORT_REL_TOL) -> np.ndarray:
    """Compute a sorted distinct one-dimensional candidate support.

    ``relative_tolerance`` acts only on numerical equality after all
    frequencies have been normalized by the final analytic radius.  It is
    distinct from the physical resolution ``epsilon_omega`` used later.
    """
    support_normalized = np.asarray([0.0])
    layer_scales = sorted([1.0] * boundary + list(np.asarray(scales, dtype=float)))
    # Use one final analytic radius as the normalization scale at every
    # intermediate de-duplication.  This makes numerical equality invariant
    # under a common rescaling of all layers, as required when L0=0.
    final_radius = float(np.sum(np.abs(layer_scales))) * float(np.max(np.abs(delta)))
    for scale in layer_scales:
        contribution = scale * delta / final_radius
        support_normalized = _unique_normalized(
            (support_normalized[:, None] + contribution[None, :]).reshape(-1),
            relative_tolerance,
        )
    return support_normalized * final_radius


def _support_cache_key(ham: HamiltonianSpec, l0: int, rho: float,
                       profile: str, scales: np.ndarray) -> str:
    payload = json.dumps({
        "algorithm": ALGORITHM_VERSION, "ham": ham.key,
        "delta": ham.difference_set.tolist(), "L": L, "L0": l0,
        "rho": rho, "profile": profile,
        "scales": np.asarray(scales).round(14).tolist(),
        "decimals": SUPPORT_DECIMALS,
        "relative_tolerance": SUPPORT_REL_TOL,
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def support_cached(ham: HamiltonianSpec, l0: int, rho: float,
                   profile: str, scales: np.ndarray) -> np.ndarray:
    ensure_dirs()
    key = _support_cache_key(ham, l0, rho, profile, scales)
    path = CACHE_DIR / f"support_{key}.npy"
    if path.exists():
        return np.load(path)
    support = candidate_support(ham.difference_set, l0, scales)
    np.save(path, support)
    return support


def analytic_radius(ham: HamiltonianSpec, l0: int,
                    scales: Sequence[float]) -> float:
    return (l0 + float(np.sum(np.abs(scales)))) * ham.r0


def packing_number(values: np.ndarray, epsilon: float) -> int:
    """Return the exact one-dimensional epsilon-packing number.

    The packing number is the largest cardinality of a subset whose distinct
    elements are separated by more than ``epsilon``.  For sorted points on the
    real line, the left-to-right greedy selection is optimal: selecting the
    smallest available point leaves at least as much room for all later
    selections as choosing any larger first point.
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    if ordered.size == 0:
        return 0
    count = 1
    last = float(ordered[0])
    boundary_tolerance = 1.0e-12 * max(
        1.0, float(np.max(np.abs(ordered))), abs(float(epsilon))
    )
    for value in ordered[1:]:
        if float(value) - last > epsilon + boundary_tolerance:
            count += 1
            last = float(value)
    return count


def radial_cdf(representatives: np.ndarray, radius: float,
               r_grid: np.ndarray) -> np.ndarray:
    absolute = np.sort(np.abs(np.asarray(representatives)))
    counts = np.searchsorted(absolute, r_grid * radius + 1e-12, side="right")
    return counts / len(absolute)


def radial_packing_cdf(values: np.ndarray, radius: float, epsilon: float,
                       r_grid: np.ndarray) -> np.ndarray:
    """Cumulative radial packing fraction without choosing representatives.

    At each normalized radius ``r``, the numerator is the packing number of
    the original frequencies inside ``[-r*radius, r*radius]``.  Dividing by
    the packing number of the full support gives a unique finite-resolution
    radial statistic.
    """
    values = np.asarray(values, dtype=float)
    denominator = packing_number(values, epsilon)
    if denominator == 0:
        raise ValueError("radial_packing_cdf requires a nonempty support")
    result = []
    tolerance = 1.0e-12 * max(1.0, abs(float(radius)))
    for r in np.asarray(r_grid, dtype=float):
        subset = values[np.abs(values) <= r * radius + tolerance]
        result.append(packing_number(subset, epsilon) / denominator)
    return np.asarray(result, dtype=float)


def quantile_summary(values: Sequence[float]) -> tuple[float, float, float]:
    q25, median, q75 = np.quantile(np.asarray(values, dtype=float),
                                   [0.25, 0.5, 0.75])
    return float(q25), float(median), float(q75)


def write_csv(path: Path, rows: Iterable[dict],
              fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames) if fieldnames is not None else list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def toy_examples() -> None:
    delta = np.asarray([-1.0, 0.0, 1.0])
    support = candidate_support(delta, 1, [2.0])
    expected = np.arange(-3.0, 4.0)
    if not np.allclose(support, expected):
        raise AssertionError(f"Toy support mismatch: {support}")
    packing = packing_number(np.asarray([0.0, 0.04, 0.11]), 0.10)
    if packing != 2:
        raise AssertionError(f"Toy packing mismatch: {packing}")
    chain_packing = packing_number(np.asarray([0.0, 0.10, 0.20]), 0.15)
    if chain_packing != 2:
        raise AssertionError(f"Chain toy packing mismatch: {chain_packing}")
    print("Toy support:", support.tolist())
    print("Toy finite-resolution packing number:", packing)
    print("Nontransitive-chain packing number:", chain_packing)


if __name__ == "__main__":
    toy_examples()
