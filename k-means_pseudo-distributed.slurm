#!/bin/bash
#SBATCH --job-name=ztanak_kmeans
#SBATCH --output=ztanak_kmeans_output.txt
#SBATCH --error=ztanak_kmeans_error.txt
#SBATCH --time=00:15:00
#SBATCH --partition=cpu
#SBATCH --nodes=1              # 只申请1个节点
#SBATCH --ntasks=1             # 1个总任务
#SBATCH --cpus-per-task=8      # 把单机核心数提高到8核
#SBATCH --mem=32G              # 提高单机内存到32G
#SBATCH --account=mscbdtsuperpod


# 激活隔离的虚拟环境
# 激活conda 
module load Anaconda3/2023.09-0
# 激活环境
source activate env_5003


# 指向完整版 Spark 目录
export SPARK_HOME=/home/ztanak/spark-standalone
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# 确保 Spark 使用 Conda 环境中的 Python
export PYSPARK_PYTHON=${CONDA_PREFIX}/bin/python
export PYSPARK_DRIVER_PYTHON=${CONDA_PREFIX}/bin/python


# 1. 获取当前节点名称并构建 Master URL
MASTER_NODE=$(hostname)
SPARK_MASTER_PORT=7077
SPARK_MASTER_URL="spark://${MASTER_NODE}:${SPARK_MASTER_PORT}"

echo ">>> [系统提示] SLURM 分配的节点为: ${MASTER_NODE} <<<"
echo ">>> [系统提示] 开始启动单机伪分布式 Spark 集群... <<<"

# 2. 启动 Spark Master
start-master.sh -h "$MASTER_NODE" -p "$SPARK_MASTER_PORT"
sleep 5  # 给予 Master 充分的启动时间

# 3. 启动 Spark Worker 并连接到 Master
# 分配 SLURM 申请到的 8 个核心和 24G 内存给 Worker
start-worker.sh ${SPARK_MASTER_URL} -c ${SLURM_CPUS_PER_TASK} -m 24G
sleep 5  # 给予 Worker 注册到 Master 的时间
echo ">>> [系统提示] 集群启动完毕。准备提交任务..."

# 4. 使用 spark-submit 提交任务(与多机提交方式保持一致)
echo ">>> [系统提示] 正在将 k-means.py 提交到独立集群执行... <<<"
spark-submit \
    --master ${SPARK_MASTER_URL} \
    --executor-cores 8 \
    --executor-memory 16G \
    --driver-memory 4G \
    /home/ztanak/MSBD5003/k-means.py


# 5. 强制使用绝对路径关闭集群
echo ">>> [系统提示] 计算完成，正在关闭 Spark 集群... <<<"
stop-worker.sh
stop-master.sh
