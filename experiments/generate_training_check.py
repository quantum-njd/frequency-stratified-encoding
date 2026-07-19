#!/usr/bin/env python3
"""Generate the matched-resource training check.

This auxiliary experiment trains the same two-qubit, six-layer, 24-parameter
ansatz under the three schedules used in the coefficient-realization
calculation. Layer scales are fixed; only the variational Ry/Rz angles are
optimized.

Target families are selected from candidate supports before training:
- Shared: frequencies available to all three schedules;
- Matched-only: frequencies available to the matched-reach schedule but not
  to the standard schedule, within the standard radius;
- Expanded-only: frequencies above the standard radius that are available to
  the expanded-reach schedule.

Each family contains four randomly sampled single-frequency targets.  Sampling
uses one fixed seed, is uniform without replacement within each declared
frequency pool, and takes place before training.  All schedules use the same
inputs, targets, initial parameters, optimizer, and step count.
The experiment tests the tested ansatz and budget only; it does not establish
universal trainability.

Toy check
---------
Before training, one parameter-shift derivative is compared with a centered
finite-difference derivative on a small batch.  This checks the gradient used
by the optimizer without adding a second paper-level circuit.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from generate_coefficient_realization import (
    COMMON_PERIOD,
    DEPTH,
    PARAMETERS_PER_LAYER,
    SCHEDULES,
    candidate_for_schedule,
    circuit_values,
)
from experiment_common import (
    DATA_DIR,
    FIGURE_DIR,
    FIGURE_DPI,
    add_panel_label,
    configure_plot_style,
    ensure_output_directories,
    quantile_summary,
    write_csv,
    write_json,
)


TARGET_SEED = 20262001
INITIALIZATION_SEED = 20262003
INITIALIZATIONS = 4
TARGETS_PER_FAMILY = 4
TRAIN_POINTS = 96
TEST_POINTS = 512
TRAINING_STEPS = 160
LEARNING_RATE = 0.05
DECAY_STEP = 100
DECAY_FACTOR = 0.3
TARGET_AMPLITUDE = 0.70
CURVE_STRIDE = 10

RESULT_CSV = DATA_DIR / "training_results.csv"
CURVE_CSV = DATA_DIR / "training_curves.csv"
TARGET_CSV = DATA_DIR / "training_targets.csv"
EXAMPLE_CSV = DATA_DIR / "training_example_predictions.csv"
SUMMARY_CSV = DATA_DIR / "training_summary.csv"
CONFIG_JSON = DATA_DIR / "training_config.json"
FIGURE_PATH = FIGURE_DIR / "fig_training_check.png"


def sample_without_replacement(
    values: list[float], number: int, rng: np.random.Generator
) -> tuple[float, ...]:
    """Uniformly sample an ordered target list without replacement."""
    if len(values) < number:
        raise ValueError("The candidate pool is smaller than the requested target count.")
    indices = rng.choice(len(values), size=number, replace=False)
    return tuple(float(values[int(index)]) for index in indices)


def build_targets() -> list[dict[str, object]]:
    """Randomly sample targets from support-defined pools before training.

    The expanded-only pool is restricted to the first unit-width band beyond
    the standard radius, 6 < omega <= 7.  This predeclared band tests a
    moderate extension of spectral reach and is not based on coefficient or
    optimization outcomes.
    """

    supports = {
        name: set(float(x) for x in candidate_for_schedule(scales))
        for name, scales in SCHEDULES.items()
    }
    shared_pool = sorted(x for x in set.intersection(*supports.values()) if x > 0.0)
    matched_pool = sorted(
        x
        for x in supports["Matched reach"] - supports["Standard"]
        if 0.0 < x <= 6.0
    )
    expanded_pool = sorted(
        x
        for x in supports["Expanded reach"]
        - (supports["Standard"] | supports["Matched reach"])
        if 6.0 < x <= 7.0
    )
    rng = np.random.default_rng(TARGET_SEED)
    selected = {
        "Shared": sample_without_replacement(shared_pool, TARGETS_PER_FAMILY, rng),
        "Matched-only": sample_without_replacement(matched_pool, TARGETS_PER_FAMILY, rng),
        "Expanded-only": sample_without_replacement(expanded_pool, TARGETS_PER_FAMILY, rng),
    }
    rows: list[dict[str, object]] = []
    for family, frequencies in selected.items():
        for target_index, frequency in enumerate(frequencies):
            rows.append(
                {
                    "family": family,
                    "target_index": target_index,
                    "frequency": frequency,
                    "amplitude": TARGET_AMPLITUDE,
                    "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                }
            )
    return rows


def target_values(x_values: np.ndarray, target: dict[str, object]) -> np.ndarray:
    return float(target["amplitude"]) * np.cos(
        float(target["frequency"]) * x_values + float(target["phase"])
    )


def model_parameter_shift_gradient(
    parameters: np.ndarray,
    x_values: np.ndarray,
    scales: np.ndarray,
    parameter_index: int,
) -> np.ndarray:
    plus = parameters.copy().reshape(-1)
    minus = parameters.copy().reshape(-1)
    plus[parameter_index] += np.pi / 2.0
    minus[parameter_index] -= np.pi / 2.0
    return 0.5 * (
        circuit_values(x_values, scales, plus.reshape(parameters.shape))
        - circuit_values(x_values, scales, minus.reshape(parameters.shape))
    )


def gradient_sanity_check() -> None:
    """Compare one parameter-shift derivative with finite differences."""

    rng = np.random.default_rng(314159)
    parameters = rng.normal(0.0, 0.2, size=(DEPTH, PARAMETERS_PER_LAYER))
    x_values = np.linspace(0.0, 1.0, 7)
    parameter_index = 5
    shifted = model_parameter_shift_gradient(
        parameters, x_values, SCHEDULES["Standard"], parameter_index
    )
    step = 1.0e-6
    plus = parameters.copy().reshape(-1)
    minus = parameters.copy().reshape(-1)
    plus[parameter_index] += step
    minus[parameter_index] -= step
    finite = (
        circuit_values(x_values, SCHEDULES["Standard"], plus.reshape(parameters.shape))
        - circuit_values(x_values, SCHEDULES["Standard"], minus.reshape(parameters.shape))
    ) / (2.0 * step)
    error = float(np.max(np.abs(shifted - finite)))
    if error > 2.0e-6:
        raise AssertionError(f"Parameter-shift sanity check failed: {error}")
    print(f"Gradient toy check passed; max derivative error={error:.3e}.")


def loss_and_gradient(
    parameters: np.ndarray,
    x_values: np.ndarray,
    targets: np.ndarray,
    scales: np.ndarray,
) -> tuple[float, np.ndarray]:
    predictions = circuit_values(x_values, scales, parameters)
    residual = predictions - targets
    loss = float(np.mean(residual**2))
    gradient = np.empty(parameters.size, dtype=float)
    for index in range(parameters.size):
        derivative = model_parameter_shift_gradient(
            parameters, x_values, scales, index
        )
        gradient[index] = 2.0 * float(np.mean(residual * derivative))
    return loss, gradient.reshape(parameters.shape)


def train_one(
    initial_parameters: np.ndarray,
    scales: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Run a fixed-step Adam optimization and return the final parameters."""

    parameters = initial_parameters.copy()
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    beta1, beta2 = 0.9, 0.999
    epsilon = 1.0e-8
    curve: list[dict[str, float]] = []
    for step in range(1, TRAINING_STEPS + 1):
        loss, gradient = loss_and_gradient(parameters, x_train, y_train, scales)
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        corrected_first = first_moment / (1.0 - beta1**step)
        corrected_second = second_moment / (1.0 - beta2**step)
        learning_rate = LEARNING_RATE * (
            DECAY_FACTOR if step > DECAY_STEP else 1.0
        )
        parameters -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )
        if step == 1 or step % CURVE_STRIDE == 0 or step == TRAINING_STEPS:
            curve.append({"step": float(step), "train_mse": loss})
    return parameters, curve


def normalized_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.var(target))
    return float(np.mean((prediction - target) ** 2) / denominator)


def target_coefficient_error(
    prediction: np.ndarray,
    x_values: np.ndarray,
    target: dict[str, object],
) -> float:
    frequency = float(target["frequency"])
    estimated = np.mean(prediction * np.exp(-1j * frequency * x_values))
    exact = 0.5 * float(target["amplitude"]) * np.exp(1j * float(target["phase"]))
    return float(np.abs(estimated - exact) ** 2 / np.abs(exact) ** 2)


def run_training() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    targets = build_targets()
    x_train = np.linspace(0.0, COMMON_PERIOD, TRAIN_POINTS, endpoint=False)
    x_test = np.linspace(0.0, COMMON_PERIOD, TEST_POINTS, endpoint=False)
    rng = np.random.default_rng(INITIALIZATION_SEED)
    initializations = rng.normal(
        0.0,
        0.30,
        size=(INITIALIZATIONS, DEPTH, PARAMETERS_PER_LAYER),
    )
    result_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    for target in targets:
        y_train = target_values(x_train, target)
        y_test = target_values(x_test, target)
        for initialization in range(INITIALIZATIONS):
            for schedule_name, scales in SCHEDULES.items():
                trained, curve = train_one(
                    initializations[initialization], scales, x_train, y_train
                )
                prediction = circuit_values(x_test, scales, trained)
                nmse = normalized_mse(prediction, y_test)
                coefficient_error = target_coefficient_error(
                    prediction, x_test, target
                )
                result_rows.append(
                    {
                        "family": target["family"],
                        "target_index": target["target_index"],
                        "frequency": target["frequency"],
                        "phase": target["phase"],
                        "initialization": initialization,
                        "schedule": schedule_name,
                        "test_nmse": nmse,
                        "target_coefficient_error": coefficient_error,
                    }
                )
                for point in curve:
                    curve_rows.append(
                        {
                            "family": target["family"],
                            "target_index": target["target_index"],
                            "initialization": initialization,
                            "schedule": schedule_name,
                            **point,
                        }
                    )
                # Predeclared examples: the first randomly sampled target and
                # first shared initialization in the matched-only and
                # expanded-only families. Neither is replaced after training.
                if (
                    target["family"] in ("Matched-only", "Expanded-only")
                    and int(target["target_index"]) == 0
                    and initialization == 0
                ):
                    for x_value, target_value, predicted_value in zip(
                        x_test, y_test, prediction
                    ):
                        example_rows.append(
                            {
                                "family": target["family"],
                                "frequency": target["frequency"],
                                "phase": target["phase"],
                                "initialization": initialization,
                                "x": float(x_value),
                                "target": float(target_value),
                                "schedule": schedule_name,
                                "prediction": float(predicted_value),
                            }
                        )

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in result_rows:
        for metric in ("test_nmse", "target_coefficient_error"):
            grouped[(str(row["family"]), str(row["schedule"]), metric)].append(float(row[metric]))
    summary_rows: list[dict[str, object]] = []
    for (family, schedule, metric), values in sorted(grouped.items()):
        q25, median, q75 = quantile_summary(values)
        summary_rows.append(
            {
                "family": family,
                "schedule": schedule,
                "metric": metric,
                "q25": q25,
                "median": median,
                "q75": q75,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "sample_count": len(values),
            }
        )
    return result_rows, curve_rows, example_rows, summary_rows


def plot(
    summary_rows: list[dict[str, object]],
    example_rows: list[dict[str, object]],
) -> None:
    configure_plot_style()
    families = ("Shared", "Matched-only", "Expanded-only")
    schedules = list(SCHEDULES)
    colors = {"Standard": "#4C78A8", "Matched reach": "#59A14F", "Expanded reach": "#E45756"}
    offsets = {"Standard": -0.22, "Matched reach": 0.0, "Expanded reach": 0.22}
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.8))

    for schedule in schedules:
        x_values, medians, lower, upper = [], [], [], []
        for family_index, family in enumerate(families):
            row = next(
                row
                for row in summary_rows
                if row["family"] == family
                and row["schedule"] == schedule
                and row["metric"] == "test_nmse"
            )
            x_values.append(family_index + offsets[schedule])
            median = float(row["median"])
            medians.append(median)
            lower.append(median - float(row["q25"]))
            upper.append(float(row["q75"]) - median)
        axes[0].errorbar(
            x_values,
            medians,
            yerr=np.asarray([lower, upper]),
            marker="o",
            linestyle="none",
            markersize=6.2,
            elinewidth=1.8,
            capsize=4,
            color=colors[schedule],
            label=schedule,
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(
        range(len(families)),
        ["Shared", "Matched-only", "Expanded-only"],
        rotation=12,
    )
    axes[0].set_ylabel("normalized test MSE")
    axes[0].set_title("Random target ensembles")
    axes[0].grid(color="0.89", linewidth=0.65)
    axes[0].set_axisbelow(True)

    for axis, family, title in (
        (axes[1], "Matched-only", "Internal reorganization"),
        (axes[2], "Expanded-only", "Spectral-range extension"),
    ):
        target_curve = sorted(
            (
                row
                for row in example_rows
                if row["family"] == family and row["schedule"] == "Standard"
            ),
            key=lambda row: float(row["x"]),
        )
        if not target_curve:
            raise AssertionError(f"Missing predeclared example for {family}.")
        mask = np.asarray([float(row["x"]) <= 2.0 * np.pi for row in target_curve])
        x = np.asarray([float(row["x"]) for row in target_curve])[mask]
        target = np.asarray([float(row["target"]) for row in target_curve])[mask]
        frequency = float(target_curve[0]["frequency"])
        axis.plot(x, target, color="black", linewidth=2.2, label="Target")
        for schedule in schedules:
            rows = sorted(
                (
                    row
                    for row in example_rows
                    if row["family"] == family and row["schedule"] == schedule
                ),
                key=lambda row: float(row["x"]),
            )
            prediction = np.asarray([float(row["prediction"]) for row in rows])[mask]
            axis.plot(x, prediction, color=colors[schedule], linewidth=1.7, label=schedule)
        axis.set_xlabel(r"input $x$")
        axis.set_ylabel("model output")
        axis.set_title(fr"{title}, $\omega={frequency:g}$")
        axis.grid(color="0.89", linewidth=0.65)
        axis.set_axisbelow(True)

    for label, axis in zip(("(a)", "(b)", "(c)"), axes):
        add_panel_label(axis, label)
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    handles_c, labels_c = axes[1].get_legend_handles_labels()
    combined = dict(zip(labels_a + labels_c, handles_a + handles_c))
    figure.legend(combined.values(), combined.keys(), loc="lower center",
                  ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0, 0.15, 1, 1), w_pad=1.5)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(figure)


def main() -> None:
    ensure_output_directories()
    gradient_sanity_check()
    target_rows = build_targets()
    result_rows, curve_rows, example_rows, summary_rows = run_training()
    write_csv(TARGET_CSV, target_rows)
    write_csv(RESULT_CSV, result_rows)
    write_csv(CURVE_CSV, curve_rows)
    write_csv(EXAMPLE_CSV, example_rows)
    write_csv(SUMMARY_CSV, summary_rows)
    write_json(
        CONFIG_JSON,
        {
            "depth": DEPTH,
            "parameters": DEPTH * PARAMETERS_PER_LAYER,
            "schedules": {name: values.tolist() for name, values in SCHEDULES.items()},
            "target_seed": TARGET_SEED,
            "target_sampling": "uniform without replacement in each declared pool",
            "expanded_only_pool": "candidate frequencies satisfying 6 < omega <= 7",
            "display_rule": "first sampled target and first shared initialization; no replacement after training",
            "initialization_seed": INITIALIZATION_SEED,
            "initializations": INITIALIZATIONS,
            "targets_per_family": TARGETS_PER_FAMILY,
            "train_points": TRAIN_POINTS,
            "test_points": TEST_POINTS,
            "training_steps": TRAINING_STEPS,
            "learning_rate": LEARNING_RATE,
            "decay_step": DECAY_STEP,
            "decay_factor": DECAY_FACTOR,
            "target_amplitude": TARGET_AMPLITUDE,
            "figure_dpi": FIGURE_DPI,
            "claim_scope": "tested two-qubit ansatz, target ensemble, and optimization budget only",
        },
    )
    plot(summary_rows, example_rows)
    print(f"Saved: {TARGET_CSV}")
    print(f"Saved: {RESULT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Saved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
