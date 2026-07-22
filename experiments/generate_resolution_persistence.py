#!/usr/bin/env python3
"""Generate support persistence at fixed absolute resolution.

The panel layout matches the exact-population calculation. The changed quantity is

    S_res(epsilon) = N_epsilon / N0,

with the same absolute threshold epsilon=0.15 for every schedule and reach.

Toy interpretation
------------------
If exact frequencies are {0,0.04,0.11} and epsilon=0.10, the largest
pairwise-separated subset has two elements.  Hence N0=3, N_epsilon=2, and
S_res=2/3.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiment_common import (COLORS, DATA_DIR, FIGURE_DIR, FIGURE_DPI,
                       HAMILTONIANS, L, L0_VALUES, LABELS, N_RANDOM,
                       PRIMARY_EPSILON, REACH_VALUES, STRUCTURED,
                       add_panel_label, packing_number,
                       configure_plot_style, ensure_dirs, random_scales,
                       structured_scales, support_cached, write_csv, write_json)


def metrics(support: np.ndarray) -> tuple[int, int, float]:
    resolved = packing_number(support, PRIMARY_EPSILON)
    return len(support), resolved, resolved / len(support)


def calculate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ham in HAMILTONIANS:
        standard = support_cached(ham, L, 1.0, "standard", np.empty(0))
        for rho in REACH_VALUES:
            for l0 in L0_VALUES:
                if l0 == L:
                    n0, neps, persistence = metrics(standard)
                    rows.append({"hamiltonian": ham.key, "rho_R": rho,
                                 "L0": l0, "kind": "standard",
                                 "profile": "standard", "sample": -1,
                                 "epsilon": PRIMARY_EPSILON, "N0": n0,
                                 "N_epsilon": neps, "S_res": persistence})
                    continue
                for name in STRUCTURED:
                    scales = structured_scales(name, l0, rho)
                    support = support_cached(ham, l0, rho, name, scales)
                    n0, neps, persistence = metrics(support)
                    rows.append({"hamiltonian": ham.key, "rho_R": rho,
                                 "L0": l0, "kind": "structured",
                                 "profile": name, "sample": -1,
                                 "epsilon": PRIMARY_EPSILON, "N0": n0,
                                 "N_epsilon": neps, "S_res": persistence})
                for sample in range(N_RANDOM):
                    scales = random_scales(sample, l0, rho)
                    profile = f"random_{sample:02d}"
                    support = support_cached(ham, l0, rho, profile, scales)
                    n0, neps, persistence = metrics(support)
                    rows.append({"hamiltonian": ham.key, "rho_R": rho,
                                 "L0": l0, "kind": "random",
                                 "profile": profile, "sample": sample,
                                 "epsilon": PRIMARY_EPSILON, "N0": n0,
                                 "N_epsilon": neps, "S_res": persistence})
    return rows


def selected(rows, ham, rho, l0, profile=None, kind=None):
    return [row for row in rows if row["hamiltonian"] == ham
            and float(row["rho_R"]) == rho and int(row["L0"]) == l0
            and (profile is None or row["profile"] == profile)
            and (kind is None or row["kind"] == kind)]


def plot(rows: list[dict[str, object]]) -> None:
    configure_plot_style()
    # Keep the same panel organization as the exact-population figure: Hamiltonians are
    # columns and reach ratios are rows.
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 12.6), sharex=True, sharey=True)
    for j, ham in enumerate(HAMILTONIANS):
        standard = float(selected(rows, ham.key, 1.0, L, kind="standard")[0]["S_res"])
        for i, rho in enumerate(REACH_VALUES):
            ax = axes[i, j]
            x = np.arange(L)
            matrix = np.asarray([
                [float(selected(rows, ham.key, rho, l0,
                                profile=f"random_{sample:02d}")[0]["S_res"])
                 for l0 in range(L)]
                for sample in range(N_RANDOM)
            ])
            q25, median, q75 = np.quantile(matrix, [0.25, 0.5, 0.75], axis=0)
            ax.errorbar(x, median,
                        yerr=np.vstack((median - q25, q75 - median)),
                        fmt="none", ecolor="0.55", elinewidth=2.2,
                        capsize=4.0, capthick=1.6, zorder=1,
                        label="random IQR" if (i, j) == (0, 0) else None)
            ax.plot(x, median, color=COLORS["random"], ls="--",
                    marker="o", label=LABELS["random"])
            for name in STRUCTURED:
                y = [float(selected(rows, ham.key, rho, l0, profile=name)[0]["S_res"])
                     for l0 in range(L)]
                ax.plot(x, y, color=COLORS[name], marker="o",
                        label=LABELS[name])
            ax.scatter([L], [standard], marker="*", s=105, color="black",
                       zorder=5, label="standard" if (i, j) == (0, 0) else None)
            ax.set_xlim(-0.2, L + 0.2)
            ax.set_ylim(-0.02, 1.03)
            ax.set_xticks(range(L + 1))
            ax.grid(alpha=0.20)
            add_panel_label(ax, f"({chr(97 + 3 * i + j)})")
            if i == 0:
                ax.set_title(ham.label)
            if j == 0:
                ax.set_ylabel(fr"$\rho_R={rho:g}$" + "\n" + r"persistence $S_{\rm res}(0.15)$")
            if i == len(REACH_VALUES) - 1:
                ax.set_xlabel(r"boundary $L_0$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.008))
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(FIGURE_DIR / "fig_resolution_persistence.png",
                dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / "fig_resolution_persistence.pdf",
                bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rows = calculate()
    write_csv(DATA_DIR / "resolution_persistence_raw.csv", rows)
    write_json(DATA_DIR / "resolution_persistence_config.json", {
        "L": L, "L0_values": L0_VALUES, "rho_R": REACH_VALUES,
        "absolute_epsilon": PRIMARY_EPSILON,
        "structured_schedules": STRUCTURED, "random_samples": N_RANDOM,
        "finite_resolution_metric": "one-dimensional epsilon-packing number",
        "figure_dpi": FIGURE_DPI,
        "claim_scope": "finite-frequency-resolution distinguishability; no noise model",
    })
    plot(rows)
    print(FIGURE_DIR / "fig_resolution_persistence.png")


if __name__ == "__main__":
    main()
