# Clustering_by_Spark

PySpark clustering project for preprocessing taxi data and running K-Means, Elkan K-Means, and DBSCAN experiments. We did not invoke the existing built-in algorithms of PySpark, but instead implemented the above algorithm from the very basics.

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

All scripts also accept explicit input and output paths, which is useful when running on Databricks or other shared storage setups.

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

Example with explicit local paths:

```bash
uv run python src/data_preprocessing.py --input data/train.csv --output data/preprocessed_data_full
uv run python src/k_means.py --input data/preprocessed_data_full --output data/results/kmeans_centroids_json
uv run python src/elkan_k_means.py --input data/preprocessed_data_full --output data/results/elkan_kmeans_centroids_json
uv run python src/dbscan_grid.py --input data/preprocessed_data_full --output data/results/dbscan_clusters_parquet
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

## Run on Databricks

Recommended layout:

```text
Workspace code:
  /Workspace/Users/theochengworkinginbox@gmail.com/Clustering_by_Spark

Unity Catalog volume data:
  /Volumes/workspace/default/msbd5003_data/raw/train.csv
  /Volumes/workspace/default/msbd5003_data/processed/preprocessed_data_full
  /Volumes/workspace/default/msbd5003_data/results/
```

The scripts now default to Databricks-friendly volume paths, so they can be used directly in Databricks Jobs:

```bash
python src/data_preprocessing.py \
  --input /Volumes/workspace/default/msbd5003_data/raw/train.csv \
  --output /Volumes/workspace/default/msbd5003_data/processed/preprocessed_data_full

python src/k_means.py \
  --input /Volumes/workspace/default/msbd5003_data/processed/preprocessed_data_full \
  --output /Volumes/workspace/default/msbd5003_data/results/kmeans_centroids_json

python src/elkan_k_means.py \
  --input /Volumes/workspace/default/msbd5003_data/processed/preprocessed_data_full \
  --output /Volumes/workspace/default/msbd5003_data/results/elkan_kmeans_centroids_json

python src/dbscan_grid.py \
  --input /Volumes/workspace/default/msbd5003_data/processed/preprocessed_data_full \
  --output /Volumes/workspace/default/msbd5003_data/results/dbscan_clusters_parquet
```

Notes:

- Do not set `master("local[*]")` on Databricks; the cluster configuration should be used as-is.
- Store large inputs and outputs in Unity Catalog volumes instead of the workspace filesystem.
- The preprocessing output directory and clustering result directories will be created by Spark when the jobs run.

## Main dependencies

- `numpy`
- `pyspark`

## Path handling

All Python scripts resolve input and output paths relative to the project root. The raw dataset is read from `data/`, and preprocessed Spark outputs are also written to `data/`, so the project can be cloned and run on different machines without editing absolute paths.
