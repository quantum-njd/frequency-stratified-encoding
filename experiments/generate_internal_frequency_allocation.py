#!/usr/bin/env python3
"""Generate exact and finite-resolution internal frequency allocation.

The controlled slice L0=3 and rho_R=2 gives every schedule the same outer
radius.  Columns compare Hamiltonians.  The top row uses exact frequencies;
the bottom row uses cumulative packing fractions at the fixed absolute
resolution epsilon=0.15.

Toy interpretation
------------------
If Omega={-4,-1,0,1,4}, then F(0.25)=3/5 because the central quarter-radius
interval [-1,1] contains three of the five frequencies.  A plateau in F means
that the corresponding radial interval contains no frequency representative.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiment_common import (COLORS, DATA_DIR, FIGURE_DIR, FIGURE_DPI,
                       HAMILTONIANS, LABELS, N_RANDOM, PRIMARY_EPSILON,
                       STRUCTURED, add_panel_label,
                       configure_plot_style, ensure_dirs, packing_number,
                       radial_cdf, radial_packing_cdf, random_scales,
                       structured_scales, support_cached, write_csv, write_json)

L0 = 3
RHO = 2.0
R_GRID = np.linspace(0.0, 1.0, 201)


def curve_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ham in HAMILTONIANS:
        for kind, names in (("structured", STRUCTURED),
                            ("random", tuple(f"random_{i:02d}" for i in range(N_RANDOM)))):
            for name in names:
                if kind == "structured":
                    scales = structured_scales(name, L0, RHO)
                else:
                    sample = int(name.split("_")[1])
                    scales = random_scales(sample, L0, RHO)
                support = support_cached(ham, L0, RHO, name, scales)
                radius = float(np.max(np.abs(support)))
                exact = radial_cdf(support, radius, R_GRID)
                resolved_count = packing_number(support, PRIMARY_EPSILON)
                finite = radial_packing_cdf(
                    support, radius, PRIMARY_EPSILON, R_GRID
                )
                for resolution, curve, count in (
                    (0.0, exact, len(support)),
                    (PRIMARY_EPSILON, finite, resolved_count),
                ):
                    for r, value in zip(R_GRID, curve):
                        rows.append({"hamiltonian": ham.key, "rho_R": RHO,
                                     "L0": L0, "kind": kind, "profile": name,
                                     "epsilon": resolution, "support_count": count,
                                     "radius": radius, "r": r, "F": value})
    return rows


def extract(rows, ham, profile, epsilon):
    chosen = [row for row in rows if row["hamiltonian"] == ham
              and row["profile"] == profile
              and np.isclose(float(row["epsilon"]), epsilon)]
    return np.asarray([float(row["F"]) for row in chosen])


def plot(rows: list[dict[str, object]]) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), sharex=True, sharey=True)
    for col, ham in enumerate(HAMILTONIANS):
        for row_index, epsilon in enumerate((0.0, PRIMARY_EPSILON)):
            ax = axes[row_index, col]
            random_matrix = np.asarray([
                extract(rows, ham.key, f"random_{sample:02d}", epsilon)
                for sample in range(N_RANDOM)
            ])
            q25, median, q75 = np.quantile(random_matrix, [0.25, 0.5, 0.75], axis=0)
            ax.fill_between(R_GRID, q25, q75, step="post", color="0.62",
                            alpha=0.38, label="random IQR")
            ax.step(R_GRID, median, where="post", color=COLORS["random"],
                    ls="--", lw=2.3, label=LABELS["random"])
            for name in STRUCTURED:
                ax.step(R_GRID, extract(rows, ham.key, name, epsilon),
                        where="post", color=COLORS[name], lw=2.2,
                        label=LABELS[name])
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.015)
            ax.grid(alpha=0.20)
            add_panel_label(ax, f"({chr(97 + 3 * row_index + col)})")
            if row_index == 0:
                ax.set_title(ham.label)
            if row_index == 1:
                ax.set_xlabel(r"normalized position $r=|\omega|/R_{\rm FS}$")
    axes[0, 0].set_ylabel(r"exact cumulative fraction $F_{\Omega}^{(0)}(r)$")
    axes[1, 0].set_ylabel(r"resolved cumulative fraction $F_{\Omega}^{(0.15)}(r)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.008))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIGURE_DIR / "fig_internal_frequency_allocation.png",
                dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / "fig_internal_frequency_allocation.pdf",
                bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rows = curve_rows()
    write_csv(DATA_DIR / "internal_frequency_allocation_raw.csv", rows)
    write_json(DATA_DIR / "internal_frequency_allocation_config.json", {
        "L0": L0, "rho_R": RHO, "absolute_epsilon": PRIMARY_EPSILON,
        "structured_schedules": STRUCTURED, "random_samples": N_RANDOM,
        "r_grid_points": len(R_GRID), "figure_dpi": FIGURE_DPI,
        "finite_resolution_metric": "cumulative one-dimensional epsilon-packing fraction",
        "claim_scope": "radial distribution of exact and finite-resolution candidate supports",
    })
    plot(rows)
    print(FIGURE_DIR / "fig_internal_frequency_allocation.png")


if __name__ == "__main__":
    main()
