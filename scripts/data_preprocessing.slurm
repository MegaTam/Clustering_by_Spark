#!/bin/bash
#SBATCH --job-name=ztanak_data_preprocessing
#SBATCH --output=ztanak_data_preprocessing_output.txt
#SBATCH --error=ztanak_data_preprocessing_error.txt
#SBATCH --time=00:03:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=mscbdtsuperpod

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}" || exit 1
python "${PROJECT_ROOT}/src/data_preprocessing.py"
