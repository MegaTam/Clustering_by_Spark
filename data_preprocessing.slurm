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

python /home/ztanak/MSBD5003/data_preprocessing.py