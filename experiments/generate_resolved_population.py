#!/usr/bin/env python3
"""Generate the absolute resolved population.

The persistence calculation reports the retained fraction

    S_res(epsilon) = N_epsilon / N0.

This calculation reports ``N_epsilon`` itself for exactly the same
Hamiltonians, reaches, boundaries, schedules, random samples, and absolute
threshold epsilon=0.15.  It therefore adds no experimental variable; it only
prevents a high retained fraction from being confused with a large resolved
support.

Toy interpretation
------------------
Schedule A has N0=10 and S_res=1, so N_epsilon=10.  Schedule B has N0=100 and
S_res=0.4, so N_epsilon=40.  Although B retains a smaller fraction, it still
contains more distinguishable frequencies at the stated resolution.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from generate_resolution_persistence import calculate, selected
from experiment_common import (COLORS, DATA_DIR, FIGURE_DIR, FIGURE_DPI,
                       HAMILTONIANS, L, N_RANDOM, REACH_VALUES, STRUCTURED,
                       add_panel_label, configure_plot_style, ensure_dirs,
                       write_csv, write_json)


def plot(rows: list[dict[str, object]]) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 12.6),
                             sharex=True, sharey=True)
    all_counts = np.asarray([float(row["N_epsilon"]) for row in rows])
    common_limits = (0.75 * float(np.min(all_counts)),
                     1.35 * float(np.max(all_counts)))

    for col, ham in enumerate(HAMILTONIANS):
        standard = float(selected(rows, ham.key, 1.0, L,
                                  kind="standard")[0]["N_epsilon"])
        for row_index, rho in enumerate(REACH_VALUES):
            ax = axes[row_index, col]
            x = np.arange(L)
            random_matrix = np.asarray([
                [float(selected(rows, ham.key, rho, l0,
                                profile=f"random_{sample:02d}")[0]["N_epsilon"])
                 for l0 in range(L)]
                for sample in range(N_RANDOM)
            ])
            q25, median, q75 = np.quantile(
                random_matrix, [0.25, 0.5, 0.75], axis=0
            )
            ax.errorbar(x, median,
                        yerr=np.vstack((median - q25, q75 - median)),
                        fmt="none", ecolor="0.55", elinewidth=2.2,
                        capsize=4.0, capthick=1.6, zorder=1,
                        label="random IQR" if (row_index, col) == (0, 0)
                        else None)
            ax.plot(x, median, color=COLORS["random"], ls="--", marker="o",
                    label="random reference")
            for name in STRUCTURED:
                y = [float(selected(rows, ham.key, rho, l0,
                                    profile=name)[0]["N_epsilon"])
                     for l0 in range(L)]
                ax.plot(x, y, color=COLORS[name], marker="o", label=name)
            ax.scatter([L], [standard], marker="*", s=105, color="black",
                       zorder=5,
                       label="standard" if (row_index, col) == (0, 0) else None)
            ax.set_yscale("log")
            ax.set_ylim(*common_limits)
            ax.set_xlim(-0.2, L + 0.2)
            ax.set_xticks(range(L + 1))
            ax.grid(alpha=0.20)
            add_panel_label(ax, f"({chr(97 + 3 * row_index + col)})")
            if row_index == 0:
                ax.set_title(ham.label)
            if col == 0:
                ax.set_ylabel(fr"$\rho_R={rho:g}$" + "\n"
                              + r"resolved population $N_{0.15}$")
            if row_index == len(REACH_VALUES) - 1:
                ax.set_xlabel(r"boundary $L_0$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.008))
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    output = FIGURE_DIR / "fig_resolved_population.png"
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / "fig_resolved_population.pdf",
                bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rows = calculate()
    write_csv(DATA_DIR / "resolved_population_raw.csv", rows)
    write_json(DATA_DIR / "resolved_population_config.json", {
        "source_design": "same settings as the persistence calculation",
        "quantity": "N_epsilon",
        "absolute_epsilon": 0.15,
        "figure_dpi": FIGURE_DPI,
    })
    plot(rows)
    print(FIGURE_DIR / "fig_resolved_population.png")


if __name__ == "__main__":
    main()
