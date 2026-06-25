import argparse
from collections import deque
from pathlib import Path
import time

from pyspark import StorageLevel
from pyspark.sql import Row
from pyspark.sql import SparkSession


def get_project_root():
    file_path = globals().get("__file__")
    if file_path:
        return Path(file_path).resolve().parent.parent
    return Path("/Workspace/Users/theochengworkinginbox@gmail.com/Clustering_by_Spark")


PROJECT_ROOT = get_project_root()
DEFAULT_INPUT_PATH = "/Volumes/workspace/default/msbd5003_data/processed/preprocessed_data_full"
DEFAULT_OUTPUT_PATH = "/Volumes/workspace/default/msbd5003_data/results/dbscan_clusters_parquet"


def is_uri_path(path_value):
    return "://" in str(path_value)


def resolve_local_path(path_value):
    path_str = str(path_value)
    if is_uri_path(path_str):
        return None

    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_spark_path(path_value):
    path_str = str(path_value)
    if is_uri_path(path_str):
        return path_str

    local_path = resolve_local_path(path_str)
    return local_path.as_uri()


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Run grid-based distributed DBSCAN on preprocessed data.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to the preprocessed Pickle RDD.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Directory for DBSCAN cluster assignments.")
    parser.add_argument("--eps", type=float, default=0.02, help="Neighborhood radius in normalized space.")
    parser.add_argument("--min-pts", type=int, default=10, help="Minimum points required to form a dense region.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N preprocessed records for clustering.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Only use the first 1000 preprocessed records for a quick validation run.",
    )
    return parser


def limit_preprocessed_rdd(sc, data_rdd, limit=None, test_mode=False):
    effective_limit = 1000 if test_mode and limit is None else limit
    if effective_limit is None:
        return data_rdd
    if effective_limit <= 0:
        raise ValueError("--limit must be a positive integer")

    limited_records = data_rdd.take(effective_limit)
    num_slices = max(1, min(sc.defaultParallelism, max(1, len(limited_records) // 1000)))
    print(f">>> [数据] 当前仅使用前 {len(limited_records)} 条预处理记录进行 DBSCAN 聚类。")
    return sc.parallelize(limited_records, numSlices=num_slices)


# ==========================================
# 1. 单机版基础 DBSCAN (用于网格内部的局部聚类)
# ==========================================
def local_dbscan(cell_id, points_data, eps, min_pts):
    """
    points_data 包含: [(point_id, features_array, is_core_cell), ...]
    is_core_cell: True 表示该点真正属于这个网格，False 表示它是隔壁网格借过来的"幽灵点"
    """
    points = list(points_data)
    n = len(points)
    if n == 0:
        return []

    # 局部簇分配，初始化为 0 (未访问)
    labels = [0] * n
    cluster_id = 0
    eps_sq = eps * eps
    coordinates = [(float(point[1][0]), float(point[1][1])) for point in points]
    neighbor_cache = [None] * n

    # 预计算距离矩阵 (因为单网格内点数较少，直接计算是可行的)
    # 为了避免依赖 scipy，这里使用缓存和平方距离减少重复计算与开方开销。
    def get_neighbors(p_idx):
        cached_neighbors = neighbor_cache[p_idx]
        if cached_neighbors is not None:
            return cached_neighbors

        neighbors = []
        px, py = coordinates[p_idx]
        for i, (x, y) in enumerate(coordinates):
            dx = px - x
            if dx > eps or dx < -eps:
                continue

            dy = py - y
            if dy > eps or dy < -eps:
                continue

            if dx * dx + dy * dy <= eps_sq:
                neighbors.append(i)

        neighbor_cache[p_idx] = neighbors
        return neighbors

    for i in range(n):
        if labels[i] != 0:
            continue

        neighbors = get_neighbors(i)
        if len(neighbors) < min_pts:
            labels[i] = -1  # 标记为噪声
        else:
            cluster_id += 1
            labels[i] = cluster_id

            # 扩展簇
            seed_queue = deque()
            queued = [False] * n
            for neighbor_idx in neighbors:
                if neighbor_idx == i or labels[neighbor_idx] > 0 or queued[neighbor_idx]:
                    continue
                seed_queue.append(neighbor_idx)
                queued[neighbor_idx] = True

            while seed_queue:
                current_p = seed_queue.popleft()
                if labels[current_p] == -1:
                    labels[current_p] = cluster_id  # 噪声点变为边界点
                    continue
                if labels[current_p] != 0:
                    continue

                labels[current_p] = cluster_id
                current_neighbors = get_neighbors(current_p)
                if len(current_neighbors) >= min_pts:
                    for neighbor_idx in current_neighbors:
                        if labels[neighbor_idx] > 0 or queued[neighbor_idx]:
                            continue
                        seed_queue.append(neighbor_idx)
                        queued[neighbor_idx] = True

    # 格式化输出: 过滤掉噪声(-1)，并生成全局唯一的局部簇ID
    # 只返回真正属于该网格的点的结果，幽灵点只用来辅助计算邻域
    results = []
    for i in range(n):
        pid, _, is_core = points[i]
        # 只要该点在局部成簇了，不管是不是幽灵点，都要输出，用于后续的连接合并
        if labels[i] > 0:
            global_local_cluster = f"{cell_id[0]}_{cell_id[1]}_C{labels[i]}"
            results.append((pid, global_local_cluster))

    return results


# ==========================================
# 2. 并查集 (Union-Find) 逻辑，用于合并跨网格的簇
# ==========================================
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  # 路径压缩
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # 简单合并，这里直接用字符串排序来决定主从，保证一致性
            if root_i < root_j:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j


# ==========================================
# 3. 分布式 DBSCAN 主逻辑
# ==========================================
def run_distributed_dbscan(spark, data_path, eps=0.02, min_pts=10, limit=None, test_mode=False):
    sc = spark.sparkContext
    data_path = resolve_spark_path(data_path)
    num_partitions = max(sc.defaultParallelism * 2, 64)
    print(">>> 正在加载数据并准备网格划分...")
    print(f">>> [配置] eps={eps}, min_pts={min_pts}, shuffle partitions={num_partitions}")

    # data_rdd 格式: (id, np.array([lon, lat]))
    data_rdd = sc.pickleFile(data_path).cache()
    data_rdd = limit_preprocessed_rdd(sc, data_rdd, limit=limit, test_mode=test_mode).cache()

    # 1. 计算全局边界以建立网格坐标系
    lon_min = data_rdd.map(lambda x: x[1][0]).min()
    lat_min = data_rdd.map(lambda x: x[1][1]).min()

    bc_eps = sc.broadcast(eps)
    bc_lon_min = sc.broadcast(lon_min)
    bc_lat_min = sc.broadcast(lat_min)

    # 2. 映射阶段 (Map/FlatMap): 分配网格与幽灵点
    def assign_to_grids(record):
        pid, coords = record
        lon, lat = coords
        e = bc_eps.value
        l_min = bc_lon_min.value
        la_min = bc_lat_min.value

        # 计算主网格坐标
        cx = int((lon - l_min) / e)
        cy = int((lat - la_min) / e)

        emissions = []
        # 主网格记录 (is_core = True)
        emissions.append(((cx, cy), (pid, coords, True)))

        # 幽灵点判断逻辑 (检查距离上下左右边界是否小于 eps)
        # 注意：为了简化，这里使用正方形网格。如果是对角线范围，还需要发送到四个角网格
        x_offset = (lon - l_min) % e
        y_offset = (lat - la_min) % e

        send_left = x_offset < e
        send_right = (e - x_offset) < e
        send_bottom = y_offset < e
        send_top = (e - y_offset) < e

        # 防止局部变量未使用告警，同时保留原作者的思考痕迹。
        _ = (send_left, send_right, send_bottom, send_top)

        # 为了保证绝对连通，将点发送到周围 8 个邻居网格 (极度严谨的做法)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                # 简化实现：所有点都作为幽灵点发向周围网格。由于距离计算过滤，只有真正的邻居会被利用
                emissions.append(((cx + dx, cy + dy), (pid, coords, False)))

        return emissions

    # 包含主点和幽灵点的全量 RDD
    # 格式: ((cell_x, cell_y), (pid, coords, is_core))
    grid_rdd = data_rdd.flatMap(assign_to_grids)

    # 3. 局部聚类阶段 (GroupByKey)
    print(">>> 正在执行局部 DBSCAN 聚类...")
    local_cluster_start = time.time()
    local_clusters_rdd = grid_rdd.groupByKey(numPartitions=num_partitions).flatMap(
        lambda x: local_dbscan(x[0], x[1], bc_eps.value, min_pts)
    )
    local_clusters_rdd = local_clusters_rdd.persist(StorageLevel.MEMORY_AND_DISK)

    # 4. 全局合并阶段 (Driver 端计算并查集)
    print(">>> 正在抽取跨网格簇连通图...")
    def create_cluster_set(cluster_id):
        return {cluster_id}

    def add_cluster_to_set(cluster_ids, cluster_id):
        cluster_ids.add(cluster_id)
        return cluster_ids

    def merge_cluster_sets(left_cluster_ids, right_cluster_ids):
        if len(left_cluster_ids) < len(right_cluster_ids):
            left_cluster_ids, right_cluster_ids = right_cluster_ids, left_cluster_ids
        left_cluster_ids.update(right_cluster_ids)
        return left_cluster_ids

    overlapping_clusters = local_clusters_rdd.combineByKey(
        create_cluster_set,
        add_cluster_to_set,
        merge_cluster_sets,
        numPartitions=num_partitions,
    ).filter(lambda x: len(x[1]) > 1).collect()
    print(
        f">>> [阶段完成] 局部 DBSCAN 与连通图抽取完成，用时: {time.time() - local_cluster_start:.2f}s，"
        f"跨网格重叠点数: {len(overlapping_clusters)}"
    )

    print(">>> 正在构建并查集进行全局合并...")
    uf = UnionFind()
    for pid, cluster_list in overlapping_clusters:
        unique_clusters = sorted(cluster_list)
        for i in range(1, len(unique_clusters)):
            uf.union(unique_clusters[0], unique_clusters[i])

    # 构建全局映射字典并广播
    cluster_mapping = {}
    for pid, cluster_list in overlapping_clusters:
        for cid in cluster_list:
            if cid not in cluster_mapping:
                cluster_mapping[cid] = uf.find(cid)

    bc_cluster_mapping = sc.broadcast(cluster_mapping)

    # 5. 映射最终结果
    print(">>> 正在生成最终的全局聚类结果...")

    def map_to_global_cluster(record):
        pid, local_cid = record
        mapping = bc_cluster_mapping.value
        global_cid = mapping.get(local_cid, local_cid)
        return (pid, global_cid)

    # 最终结果去重 (因为一个点可能有主记录和幽灵记录，但最终都会映射到同一个全局 ID)
    final_clusters_rdd = local_clusters_rdd.map(map_to_global_cluster).distinct()

    return final_clusters_rdd


def save_clusters(spark, cluster_rdd, output_path):
    output_path = resolve_spark_path(output_path)
    cluster_df = spark.createDataFrame(
        cluster_rdd.map(lambda x: Row(point_id=str(x[0]), cluster_id=str(x[1])))
    )
    cluster_df.write.mode("overwrite").parquet(output_path)
    print(f">>> 已将 DBSCAN 聚类结果保存到: {output_path}")


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    spark = SparkSession.builder.appName("Grid_Distributed_DBSCAN").getOrCreate()

    # 这里的 eps 需要极其谨慎地调整，因为你的数据做了 Min-Max Scaling (范围在 0~1 之间)
    final_rdd = run_distributed_dbscan(
        spark,
        data_path=args.input,
        eps=args.eps,
        min_pts=args.min_pts,
        limit=args.limit,
        test_mode=args.test_mode,
    )

    cluster_counts = final_rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b).collect()
    cluster_counts.sort(key=lambda x: x[1], reverse=True)

    print("\n>>> 分布式 DBSCAN 聚类结果统计 (Top 10 簇):")
    for cid, count in cluster_counts[:10]:
        print(f"Cluster ID: {cid}, 包含数据点: {count}")

    save_clusters(spark, final_rdd, args.output)
    spark.stop()
