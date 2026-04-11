#!/bin/bash
#SBATCH --job-name=ztanak_kmeans
#SBATCH --output=ztanak_kmeans_output.txt
#SBATCH --error=ztanak_kmeans_error.txt
#SBATCH --time=00:15:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=mscbdtsuperpod

module load Anaconda3/2023.09-0
source activate env_5003

export SPARK_HOME=/home/ztanak/spark-standalone
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

export PYSPARK_PYTHON=${CONDA_PREFIX}/bin/python
export PYSPARK_DRIVER_PYTHON=${CONDA_PREFIX}/bin/python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}" || exit 1

MASTER_NODE=$(hostname)
SPARK_MASTER_PORT=7077
SPARK_MASTER_URL="spark://${MASTER_NODE}:${SPARK_MASTER_PORT}"

echo ">>> [系统提示] SLURM 分配的节点为: ${MASTER_NODE} <<<"
echo ">>> [系统提示] 开始启动单机伪分布式 Spark 集群... <<<"

start-master.sh -h "$MASTER_NODE" -p "$SPARK_MASTER_PORT"
sleep 5

start-worker.sh ${SPARK_MASTER_URL} -c ${SLURM_CPUS_PER_TASK} -m 24G
sleep 5
echo ">>> [系统提示] 集群启动完毕。准备提交任务..."

echo ">>> [系统提示] 正在将 k_means.py 提交到独立集群执行... <<<"
spark-submit \
    --master ${SPARK_MASTER_URL} \
    --executor-cores 8 \
    --executor-memory 16G \
    --driver-memory 4G \
    "${PROJECT_ROOT}/src/k_means.py"

echo ">>> [系统提示] 计算完成，正在关闭 Spark 集群... <<<"
stop-worker.sh
stop-master.sh
