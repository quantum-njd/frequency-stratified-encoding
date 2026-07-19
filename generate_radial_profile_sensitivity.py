#!/usr/bin/env python3
"""Generate radial-profile sensitivity at two controlled slices.

The top row keeps rho_R=2 but moves the boundary to L0=0.  The bottom row
keeps L0=3 but increases the reach to rho_R=10. This checks that the reference
radial profile is not interpreted as a universal ordering of schedule families.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiment_common import (COLORS, DATA_DIR, FIGURE_DIR, FIGURE_DPI,
                       HAMILTONIANS, LABELS, N_RANDOM, STRUCTURED,
                       add_panel_label, configure_plot_style, ensure_dirs,
                       radial_cdf, random_scales, structured_scales,
                       support_cached, write_csv)

R_GRID = np.linspace(0.0, 1.0, 201)
SLICES = ((0, 2.0, "all layers scaled"),
          (3, 10.0, "high-reach test"))


def calculate():
    rows = []
    for l0, rho, description in SLICES:
        for ham in HAMILTONIANS:
            names = list(STRUCTURED) + [f"random_{i:02d}" for i in range(N_RANDOM)]
            for name in names:
                if name.startswith("random"):
                    sample = int(name.split("_")[1])
                    scales = random_scales(sample, l0, rho)
                    kind = "random"
                else:
                    scales = structured_scales(name, l0, rho)
                    kind = "structured"
                support = support_cached(ham, l0, rho, name, scales)
                radius = float(np.max(np.abs(support)))
                curve = radial_cdf(support, radius, R_GRID)
                for r, value in zip(R_GRID, curve):
                    rows.append({"description": description, "L0": l0,
                                 "rho_R": rho, "hamiltonian": ham.key,
                                 "kind": kind, "profile": name,
                                 "r": r, "F_exact": value})
    return rows


def extract(rows, description, ham, profile):
    return np.asarray([float(row["F_exact"]) for row in rows
                       if row["description"] == description
                       and row["hamiltonian"] == ham
                       and row["profile"] == profile])


def plot(rows):
    configure_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), sharex=True, sharey=True)
    for row_index, (l0, rho, description) in enumerate(SLICES):
        for col, ham in enumerate(HAMILTONIANS):
            ax = axes[row_index, col]
            matrix = np.asarray([extract(rows, description, ham.key,
                                         f"random_{i:02d}")
                                 for i in range(N_RANDOM)])
            q25, median, q75 = np.quantile(matrix, [0.25, 0.5, 0.75], axis=0)
            ax.fill_between(R_GRID, q25, q75, step="post", color="0.62",
                            alpha=0.38, label="random IQR")
            ax.step(R_GRID, median, where="post", color=COLORS["random"],
                    ls="--", lw=2.3, label=LABELS["random"])
            for name in STRUCTURED:
                ax.step(R_GRID, extract(rows, description, ham.key, name),
                        where="post", color=COLORS[name], lw=2.2,
                        label=LABELS[name])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.015)
            ax.grid(alpha=0.20)
            add_panel_label(ax, f"({chr(97 + 3 * row_index + col)})")
            if row_index == 0:
                ax.set_title(ham.label)
            if col == 0:
                ax.set_ylabel(description + "\n" + r"$F_{\Omega}^{(0)}(r)$")
            if row_index == 1:
                ax.set_xlabel(r"normalized position $r=|\omega|/R_{\rm FS}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.008))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIGURE_DIR / "fig_radial_profile_sensitivity.png",
                dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def main():
    ensure_dirs()
    rows = calculate()
    write_csv(DATA_DIR / "radial_profile_sensitivity.csv", rows)
    plot(rows)
    print(FIGURE_DIR / "fig_radial_profile_sensitivity.png")


if __name__ == "__main__":
    main()
