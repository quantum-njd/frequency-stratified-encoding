#!/usr/bin/env python3
"""Generate the exact candidate-support population results.

Each panel fixes one Hamiltonian and one reach ratio.  The horizontal axis is
the integer stratification boundary L0.  Colored curves are explicit schedule
rules; the black reference summarizes 24 prespecified random simplex
allocations.  The script saves a 600-dpi PNG and the plotted values, then calls
``plt.show()``.

Toy interpretation
------------------
For Delta={-1,0,1}, standard depth two gives {-2,-1,0,1,2}, so N0=5.
Using scales (1,2) gives {-3,-2,-1,0,1,2,3}, so N0=7.  The radius and the
number of distinct frequencies answer different questions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiment_common import (COLORS, DATA_DIR, FIGURE_DIR, FIGURE_DPI,
                       HAMILTONIANS, L, L0_VALUES, LABELS, N_RANDOM,
                       REACH_VALUES, STRUCTURED, add_panel_label,
                       analytic_radius, configure_plot_style, ensure_dirs,
                       random_scales, structured_scales, support_cached,
                       write_csv, write_json)


def calculate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ham in HAMILTONIANS:
        standard = support_cached(ham, L, 1.0, "standard", np.empty(0))
        for rho in REACH_VALUES:
            for l0 in L0_VALUES:
                if l0 == L:
                    rows.append({"hamiltonian": ham.key, "rho_R": rho,
                                 "L0": l0, "kind": "standard",
                                 "profile": "standard", "sample": -1,
                                 "N0": len(standard),
                                 "radius_numeric": float(np.max(np.abs(standard))),
                                 "radius_analytic": L * ham.r0})
                    continue
                for name in STRUCTURED:
                    scales = structured_scales(name, l0, rho)
                    support = support_cached(ham, l0, rho, name, scales)
                    rows.append({"hamiltonian": ham.key, "rho_R": rho,
                                 "L0": l0, "kind": "structured",
                                 "profile": name, "sample": -1,
                                 "N0": len(support),
                                 "radius_numeric": float(np.max(np.abs(support))),
                                 "radius_analytic": analytic_radius(ham, l0, scales)})
                for sample in range(N_RANDOM):
                    scales = random_scales(sample, l0, rho)
                    profile = f"random_{sample:02d}"
                    support = support_cached(ham, l0, rho, profile, scales)
                    rows.append({"hamiltonian": ham.key, "rho_R": rho,
                                 "L0": l0, "kind": "random",
                                 "profile": profile, "sample": sample,
                                 "N0": len(support),
                                 "radius_numeric": float(np.max(np.abs(support))),
                                 "radius_analytic": analytic_radius(ham, l0, scales)})
    return rows


def selected(rows, ham, rho, l0, profile=None, kind=None):
    return [row for row in rows if row["hamiltonian"] == ham
            and float(row["rho_R"]) == rho and int(row["L0"]) == l0
            and (profile is None or row["profile"] == profile)
            and (kind is None or row["kind"] == kind)]


def plot(rows: list[dict[str, object]]) -> None:
    configure_plot_style()
    # Use Hamiltonians as columns throughout the support-geometry figures; the
    # consistent column semantics make vertical comparisons isolate
    # the reach ratio for one fixed generator spectrum.
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 12.6), sharex=True, sharey=True)
    all_counts = np.asarray([float(row["N0"]) for row in rows])
    common_limits = (0.75 * float(np.min(all_counts)),
                     1.35 * float(np.max(all_counts)))
    for j, ham in enumerate(HAMILTONIANS):
        standard = float(selected(rows, ham.key, 1.0, L, kind="standard")[0]["N0"])
        for i, rho in enumerate(REACH_VALUES):
            ax = axes[i, j]
            x = np.arange(L)
            random_matrix = np.asarray([
                [float(selected(rows, ham.key, rho, l0,
                                profile=f"random_{sample:02d}")[0]["N0"])
                 for l0 in range(L)]
                for sample in range(N_RANDOM)
            ])
            q25, median, q75 = np.quantile(random_matrix, [0.25, 0.5, 0.75], axis=0)
            # Exact counts coincide for most generic allocations.  Draw an
            # interval only where the IQR is genuinely nonzero; otherwise the
            # black median marker is the complete summary.
            nonzero_iqr = q75 > q25
            if np.any(nonzero_iqr):
                ax.errorbar(x[nonzero_iqr], median[nonzero_iqr],
                            yerr=np.vstack((median[nonzero_iqr] - q25[nonzero_iqr],
                                            q75[nonzero_iqr] - median[nonzero_iqr])),
                            fmt="none", ecolor="0.55", elinewidth=2.2,
                            capsize=4.0, capthick=1.6, zorder=1)
            ax.plot(x, median, color=COLORS["random"], ls="--",
                    marker="o", label=LABELS["random"])
            for name in STRUCTURED:
                y = [float(selected(rows, ham.key, rho, l0, profile=name)[0]["N0"])
                     for l0 in range(L)]
                ax.plot(x, y, color=COLORS[name], marker="o",
                        label=LABELS[name])
            ax.scatter([L], [standard], marker="*", s=105, color="black",
                       zorder=5, label="standard" if (i, j) == (0, 0) else None)
            ax.set_yscale("log")
            ax.set_ylim(*common_limits)
            ax.set_xlim(-0.2, L + 0.2)
            ax.set_xticks(range(L + 1))
            ax.grid(alpha=0.20)
            add_panel_label(ax, f"({chr(97 + 3 * i + j)})")
            if i == 0:
                ax.set_title(ham.label)
            if j == 0:
                ax.set_ylabel(fr"$\rho_R={rho:g}$" + "\n" + r"exact population $N_0$")
            if i == len(REACH_VALUES) - 1:
                ax.set_xlabel(r"boundary $L_0$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.008))
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(FIGURE_DIR / "fig_exact_population.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / "fig_exact_population.pdf",
                bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rows = calculate()
    write_csv(DATA_DIR / "exact_population_raw.csv", rows)
    write_json(DATA_DIR / "exact_population_config.json", {
        "L": L, "L0_values": L0_VALUES, "rho_R": REACH_VALUES,
        "structured_schedules": STRUCTURED, "random_samples": N_RANDOM,
        "figure_dpi": FIGURE_DPI,
        "claim_scope": "generator-induced exact candidate support",
    })
    plot(rows)
    print(FIGURE_DIR / "fig_exact_population.png")


if __name__ == "__main__":
    main()
