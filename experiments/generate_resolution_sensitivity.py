#!/usr/bin/env python3
"""Generate sensitivity of S_res to the absolute resolution threshold.

The controlled setting L0=3, rho_R=2 is reused.  This plot checks that the
main cross-section epsilon=0.15 is not the only threshold examined.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiment_common import (COLORS, DATA_DIR, FIGURE_DIR, FIGURE_DPI,
                       HAMILTONIANS, LABELS, N_RANDOM, RESOLUTION_VALUES,
                       STRUCTURED, add_panel_label,
                       configure_plot_style, packing_number,
                       ensure_dirs, random_scales, structured_scales,
                       support_cached, write_csv)

L0 = 3
RHO = 2.0


def persistence(support, epsilon):
    return packing_number(support, epsilon) / len(support)


def calculate():
    rows = []
    for ham in HAMILTONIANS:
        for name in STRUCTURED:
            support = support_cached(ham, L0, RHO, name,
                                     structured_scales(name, L0, RHO))
            for epsilon in RESOLUTION_VALUES:
                rows.append({"hamiltonian": ham.key, "kind": "structured",
                             "profile": name, "sample": -1,
                             "epsilon": epsilon,
                             "S_res": persistence(support, epsilon)})
        for sample in range(N_RANDOM):
            name = f"random_{sample:02d}"
            support = support_cached(ham, L0, RHO, name,
                                     random_scales(sample, L0, RHO))
            for epsilon in RESOLUTION_VALUES:
                rows.append({"hamiltonian": ham.key, "kind": "random",
                             "profile": name, "sample": sample,
                             "epsilon": epsilon,
                             "S_res": persistence(support, epsilon)})
    return rows


def plot(rows):
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.25), sharex=True, sharey=True)
    for col, ham in enumerate(HAMILTONIANS):
        ax = axes[col]
        matrix = np.asarray([
            [next(float(row["S_res"]) for row in rows
                  if row["hamiltonian"] == ham.key
                  and row["profile"] == f"random_{sample:02d}"
                  and np.isclose(float(row["epsilon"]), epsilon))
             for epsilon in RESOLUTION_VALUES]
            for sample in range(N_RANDOM)
        ])
        q25, median, q75 = np.quantile(matrix, [0.25, 0.5, 0.75], axis=0)
        ax.fill_between(RESOLUTION_VALUES, q25, q75, color="0.62", alpha=0.38,
                        label="random IQR")
        ax.plot(RESOLUTION_VALUES, median, color=COLORS["random"], ls="--",
                marker="o", label=LABELS["random"])
        for name in STRUCTURED:
            y = [next(float(row["S_res"]) for row in rows
                      if row["hamiltonian"] == ham.key and row["profile"] == name
                      and np.isclose(float(row["epsilon"]), epsilon))
                 for epsilon in RESOLUTION_VALUES]
            ax.plot(RESOLUTION_VALUES, y, color=COLORS[name], marker="o",
                    label=LABELS[name])
        ax.axvline(0.15, color="0.35", ls=":", lw=1.0)
        ax.set_title(ham.label)
        ax.set_xlabel(r"absolute resolution $\varepsilon_\omega$")
        ax.set_ylim(-0.02, 1.03)
        ax.grid(alpha=0.20)
        add_panel_label(ax, f"({chr(97 + col)})")
    axes[0].set_ylabel(r"persistence $S_{\rm res}(\varepsilon_\omega)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(FIGURE_DIR / "fig_resolution_sensitivity.png",
                dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def main():
    ensure_dirs()
    rows = calculate()
    write_csv(DATA_DIR / "resolution_sensitivity.csv", rows)
    plot(rows)
    print(FIGURE_DIR / "fig_resolution_sensitivity.png")


if __name__ == "__main__":
    main()
