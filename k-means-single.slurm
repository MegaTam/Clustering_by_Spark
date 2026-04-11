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

python /home/ztanak/MSBD5003/k-means.py
