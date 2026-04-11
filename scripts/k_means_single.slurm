#!/bin/bash
#SBATCH --job-name=ztanak_kmeans
#SBATCH --output=ztanak_kmeans_output.txt
#SBATCH --error=ztanak_kmeans_error.txt
#SBATCH --time=00:10:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=mscbdtsuperpod

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}" || exit 1
python "${PROJECT_ROOT}/src/k_means.py"
