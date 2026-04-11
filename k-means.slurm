#!/bin/bash
#SBATCH --job-name=ztanak_kmeans_dist
#SBATCH --output=ztanak_kmeans_dist_output.txt
#SBATCH --error=ztanak_kmeans_dist_error.txt
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --nodes=3              # 申请 3 个物理节点 (可根据需要调整)
#SBATCH --ntasks-per-node=1    # 每个节点启动 1 个总任务控制 Worker
#SBATCH --cpus-per-task=8      # 每个节点分配 8 个核心
#SBATCH --mem=32G              # 每个节点分配 32G 内存
#SBATCH --account=mscbdtsuperpod

# 1. 激活隔离的虚拟环境
# 激活conda 
module load Anaconda3/2023.09-0
# 激活环境
source activate env_5003

# 2. 指向完整版 Spark 目录
export SPARK_HOME=/home/ztanak/spark-standalone
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# 确保 Spark 使用 Conda 环境中的 Python
export PYSPARK_PYTHON=${CONDA_PREFIX}/bin/python
export PYSPARK_DRIVER_PYTHON=${CONDA_PREFIX}/bin/python


# 3. 自动解析 SLURM 分配的节点列表
# scontrol show hostnames 会把 "intel-[01-03]" 这样的缩写展开为 "intel-01 intel-02 intel-03"
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

# 4. 启动 Spark Master (直接在当前运行脚本的首个节点启动)
echo ">>> [启动] 正在 ${MASTER_NODE} 启动 Master..."
start-master.sh -h "$MASTER_NODE" -p "$SPARK_MASTER_PORT"
sleep 5

# 5. 跨节点启动 Spark Worker
# 通过循环遍历所有分配到的节点，使用 ssh 远程启动 Worker 并连接到 Master
export SPARK_WORKER_CORES=${SLURM_CPUS_PER_TASK}
export SPARK_WORKER_MEMORY=24G

for node in "${NODES[@]}"; do
    echo ">>> [启动] 正在节点 ${node} 启动 Worker..."
    # 注意：HPC 集群通常允许在分配的节点间免密 ssh。
    # 我们需要将环境变量一并传给目标节点，否则目标节点可能找不到 start-worker.sh
    ssh $node "export SPARK_HOME=$SPARK_HOME; export PATH=\$SPARK_HOME/bin:\$SPARK_HOME/sbin:\$PATH; start-worker.sh $SPARK_MASTER_URL -c $SPARK_WORKER_CORES -m $SPARK_WORKER_MEMORY"
done

# 给所有 Worker 足够的时间注册到 Master
sleep 10
echo ">>> [系统提示] 多机集群启动完毕。准备提交分布式任务..."

# 6. 使用 spark-submit 提交任务
# --executor-cores 指定每个 Executor 的核心数（通常等于或小于单机分配的核心数）
# 在此架构下，如果有 3 个节点，将会有 3 个 Worker 处理任务
spark-submit \
    --master ${SPARK_MASTER_URL} \
    --executor-cores 8 \
    --executor-memory 16G \
    --driver-memory 4G \
    /home/ztanak/MSBD5003/k-means.py


echo ">>> [系统提示] 任务执行完毕，正在清理跨节点 Spark 进程..."


# 7. 安全关闭所有节点的 Worker 和 Master
for node in "${NODES[@]}"; do
    echo ">>> [关闭] 正在节点 ${node} 关闭 Worker..."
    ssh $node "$SPARK_HOME/sbin/stop-worker.sh"
done

echo ">>> [关闭] 正在 ${MASTER_NODE} 关闭 Master..."
stop-master.sh