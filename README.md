# Clustering_by_Spark

PySpark clustering project for preprocessing taxi data and running K-Means, Elkan K-Means, and DBSCAN experiments.

## Environment

- Python: `3.10`
- Package manager: `uv`

This project uses `uv` to manage the Python version, virtual environment, and dependencies. The required Python version is pinned in `.python-version`, and dependencies are tracked in `pyproject.toml` and `uv.lock`.

## Setup with uv

Install `uv` first if it is not already available on your machine.

Then, from the project root, run:

```bash
uv sync
```

This will:

- create the local virtual environment in `.venv`
- install Python `3.10` if needed
- install all required packages for the project

## Run with uv

From the project root:

```bash
uv run python src/data_preprocessing.py
uv run python src/k_means.py
uv run python src/elkan_k_means.py
uv run python src/dbscan_grid.py
```

Using `uv run` is recommended so everyone on the team runs the project with the same Python version and dependencies.

## Structure

- `src/`: Python source files
- `scripts/`: SLURM job scripts
- `data/`: input dataset such as `train.csv` and generated outputs such as `preprocessed_data_test`

## Run locally

From the project root:

```bash
uv run python src/data_preprocessing.py
uv run python src/k_means.py
uv run python src/elkan_k_means.py
uv run python src/dbscan_grid.py
```

Expected data layout:

```text
Clustering_by_Spark/
  data/
    train.csv
    preprocessed_data_test/
    preprocessed_data_full/
```

## Run with SLURM

From the project root:

```bash
sbatch scripts/data_preprocessing.slurm
sbatch scripts/k_means_pseudo_distributed.slurm
sbatch scripts/elkan_k_means_pseudo_distributed.slurm
sbatch scripts/dbscan_pseudo_distributed.slurm
```

## Main dependencies

- `numpy`
- `pyspark`

## Path handling

All Python scripts resolve input and output paths relative to the project root. The raw dataset is read from `data/`, and preprocessed Spark outputs are also written to `data/`, so the project can be cloned and run on different machines without editing absolute paths.