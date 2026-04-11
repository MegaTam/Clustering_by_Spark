from pyspark.sql import SparkSession
import numpy as np
import math

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
    
    # 局部簇分配，初始化为 0 (未访问)
    labels = [0] * n 
    cluster_id = 0
    
    # 预计算距离矩阵 (因为单网格内点数较少，直接计算是可行的)
    # 为了避免依赖 scipy，这里手写欧氏距离计算
    def get_neighbors(p_idx):
        neighbors = []
        p_coords = points[p_idx][1]
        for i in range(n):
            if np.linalg.norm(p_coords - points[i][1]) <= eps:
                neighbors.append(i)
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
            seed_set = list(neighbors)
            seed_set.remove(i) if i in seed_set else None
            
            while len(seed_set) > 0:
                current_p = seed_set.pop(0)
                if labels[current_p] == -1:
                    labels[current_p] = cluster_id # 噪声点变为边界点
                if labels[current_p] != 0:
                    continue
                    
                labels[current_p] = cluster_id
                current_neighbors = get_neighbors(current_p)
                if len(current_neighbors) >= min_pts:
                    seed_set.extend(current_neighbors)
                    
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
        self.parent[i] = self.find(self.parent[i]) # 路径压缩
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
def run_distributed_dbscan(spark, data_path, eps=0.02, min_pts=10):
    sc = spark.sparkContext
    print(">>> 正在加载数据并准备网格划分...")
    
    # data_rdd 格式: (id, np.array([lon, lat]))
    data_rdd = sc.pickleFile(data_path).cache()
    
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
        
        send_left = x_offset < e  # 其实只需检查 x_offset < eps 即可，但严格说应该查更小的阈值，这里简化处理直接发送相邻网格
        send_right = (e - x_offset) < e
        send_bottom = y_offset < e
        send_top = (e - y_offset) < e
        
        # 为了保证绝对连通，将点发送到周围 8 个邻居网格 (极度严谨的做法)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                # 如果点在边界 eps 范围内，发送为 False (非核心点)
                # 简化实现：所有点都作为幽灵点发向周围网格。由于距离计算过滤，只有真正的邻居会被利用
                emissions.append(((cx + dx, cy + dy), (pid, coords, False)))
                
        return emissions

    # 包含主点和幽灵点的全量 RDD
    # 格式: ((cell_x, cell_y), (pid, coords, is_core))
    grid_rdd = data_rdd.flatMap(assign_to_grids)
    
    # 3. 局部聚类阶段 (GroupByKey)
    # 按网格聚合，触发 Shuffle，然后调用 local_dbscan
    print(">>> 正在执行局部 DBSCAN 聚类...")
    local_clusters_rdd = grid_rdd.groupByKey().flatMap(
        lambda x: local_dbscan(x[0], x[1], bc_eps.value, min_pts)
    )
    # 输出格式: (point_id, local_cluster_id)
    local_clusters_rdd.cache()
    
    # 4. 全局合并阶段 (Driver 端计算并查集)
    print(">>> 正在抽取跨网格簇连通图...")
    # 将属于多个簇的 point_id 聚合并提取出来
    # 格式: (point_id, [Cluster_0_0_C1, Cluster_0_1_C3, ...])
    overlapping_clusters = local_clusters_rdd.groupByKey() \
                                             .mapValues(list) \
                                             .filter(lambda x: len(set(x[1])) > 1) \
                                             .collect()
                                             
    print(">>> 正在构建并查集进行全局合并...")
    uf = UnionFind()
    for pid, cluster_list in overlapping_clusters:
        unique_clusters = list(set(cluster_list))
        # 将列表中的所有簇通过并查集连接在一起
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
        # 如果字典里有，说明被合并了；如果没有，就用原本的局部 ID 作为全局 ID
        mapping = bc_cluster_mapping.value
        global_cid = mapping.get(local_cid, local_cid)
        return (pid, global_cid)

    # 最终结果去重 (因为一个点可能有主记录和幽灵记录，但最终都会映射到同一个全局 ID)
    final_clusters_rdd = local_clusters_rdd.map(map_to_global_cluster).distinct()
    
    return final_clusters_rdd

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Grid_Distributed_DBSCAN").getOrCreate()
    
    # 这里的 eps 需要极其谨慎地调整，因为你的数据做了 Min-Max Scaling (范围在 0~1 之间)
    # 如果标准化后的距离过短，可能把所有点聚成一团。0.02 只是个占位符。
    final_rdd = run_distributed_dbscan(
        spark, 
        data_path="preprocessed_data_test", 
        eps=0.02, 
        min_pts=10
    )
    
    cluster_counts = final_rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b).collect()
    cluster_counts.sort(key=lambda x: x[1], reverse=True)
    
    print("\n>>> 分布式 DBSCAN 聚类结果统计 (Top 10 簇):")
    for cid, count in cluster_counts[:10]:
        print(f"Cluster ID: {cid}, 包含数据点: {count}")
        
    spark.stop()
