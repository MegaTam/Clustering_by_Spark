#!/bin/bash
#SBATCH --job-name=ztanak_kmeans_dist
#SBATCH --output=ztanak_kmeans_dist_output.txt
#SBATCH --error=ztanak_kmeans_dist_error.txt
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=mscbdtsuperpod

# 1. 激活隔离的虚拟环境
module load Anaconda3/2023.09-0
source activate env_5003

# 2. 指向完整版 Spark 目录
export SPARK_HOME=/home/ztanak/spark-standalone
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# 确保 Spark 使用 Conda 环境中的 Python
export PYSPARK_PYTHON=${CONDA_PREFIX}/bin/python
export PYSPARK_DRIVER_PYTHON=${CONDA_PREFIX}/bin/python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}" || exit 1

# 3. 自动解析 SLURM 分配的节点列表
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# 默认将分配到的第一个节点作为 Master 节点
MASTER_NODE=${NODES[0]}
SPARK_MASTER_PORT=7077
SPARK_MASTER_URL="spark://${MASTER_NODE}:${SPARK_MASTER_PORT}"

echo "=========================================================="
echo ">>> [系统提示] 共分配到 ${#NODES[@]} 个计算节点"
echo ">>> [系统提示] 节点列表: ${NODES[*]}"
echo ">>> [系统提示] Master 节点定为: ${MASTER_NODE}"
echo "=========================================================="

# 4. 启动 Spark Master
echo ">>> [启动] 正在 ${MASTER_NODE} 启动 Master..."
start-master.sh -h "$MASTER_NODE" -p "$SPARK_MASTER_PORT"
sleep 5

# 5. 跨节点启动 Spark Worker
export SPARK_WORKER_CORES=${SLURM_CPUS_PER_TASK}
export SPARK_WORKER_MEMORY=24G

for node in "${NODES[@]}"; do
    echo ">>> [启动] 正在节点 ${node} 启动 Worker..."
    ssh $node "export SPARK_HOME=$SPARK_HOME; export PATH=\$SPARK_HOME/bin:\$SPARK_HOME/sbin:\$PATH; start-worker.sh $SPARK_MASTER_URL -c $SPARK_WORKER_CORES -m $SPARK_WORKER_MEMORY"
done

sleep 10
echo ">>> [系统提示] 多机集群启动完毕。准备提交分布式任务..."

# 6. 使用 spark-submit 提交任务
spark-submit \
    --master ${SPARK_MASTER_URL} \
    --executor-cores 8 \
    --executor-memory 16G \
    --driver-memory 4G \
    "${PROJECT_ROOT}/src/k_means.py"

echo ">>> [系统提示] 任务执行完毕，正在清理跨节点 Spark 进程..."

# 7. 安全关闭所有节点的 Worker 和 Master
for node in "${NODES[@]}"; do
    echo ">>> [关闭] 正在节点 ${node} 关闭 Worker..."
    ssh $node "$SPARK_HOME/sbin/stop-worker.sh"
done

echo ">>> [关闭] 正在 ${MASTER_NODE} 关闭 Master..."
stop-master.sh
