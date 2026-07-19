# Numerical Experiment Code

This repository contains the Python scripts used to generate numerical results
associated with frequency-stratified quantum encoding.

## Requirements

- Python 3.10 or later
- NumPy
- Matplotlib
- Pillow

Create the local environment with:

```bash
sh setup_python_env.sh
```

## Run

Run the complete numerical suite with:

```bash
sh run_experiments.sh
```

The scripts create the following directories locally:

- `data/` for generated numerical outputs;
- `figures/` for generated figures;
- `cache/` for regenerable intermediate arrays;
- `logs/` and `mplconfig/` for local runtime files.

The complete run includes the population, persistence, radial-profile,
coefficient-realization, training, and sensitivity calculations.

## Code organization

- `experiments/experiment_common.py`: shared numerical and plotting functions;
- `experiments/generate_*.py`: numerical result and sensitivity generators;
- `experiments/run_all.py`: fixed execution order for the complete suite.

Random seeds, thresholds, schedules, and shared comparison settings are defined
in the source files, with key settings also recorded in the generated configuration files.

No external data sets or network services are required after the Python
dependencies have been installed.
