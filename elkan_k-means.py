from pyspark.sql import SparkSession
import numpy as np
import time

def compute_centroid_distances(centroids):
    """Driver端计算所有质心两两之间的距离矩阵"""
    k = len(centroids)
    dist_matrix = np.zeros((k, k))
    # s_array 记录每个质心到其他质心距离的一半的最小值 (Elkan的引理1)
    s_array = np.zeros(k)
    for i in range(k):
        min_dist = np.inf
        for j in range(k):
            if i != j:
                dist = np.linalg.norm(centroids[i] - centroids[j])
                dist_matrix[i, j] = dist
                if dist < min_dist:
                    min_dist = dist
        s_array[i] = min_dist / 2.0
    return dist_matrix, s_array

def initialize_state(record, centroids):
    """第 0 轮：计算所有准确距离，初始化上下界"""
    id_val, features = record
    k = len(centroids)
    lower_bounds = np.zeros(k)
    
    min_dist = np.inf
    assignment = -1
    
    for j in range(k):
        dist = np.linalg.norm(features - centroids[j])
        lower_bounds[j] = dist
        if dist < min_dist:
            min_dist = dist
            assignment = j
            
    upper_bound = min_dist
    return (id_val, features, assignment, upper_bound, lower_bounds)

def elkan_map_logic(row, bc_centroids, bc_dist_matrix, bc_s_array, dist_calc_acc):
    """Elkan 核心剪枝与分配逻辑"""
    id_val, x, c_x, u_x, l_x = row
    centroids = bc_centroids.value
    dist_matrix = bc_dist_matrix.value
    s_array = bc_s_array.value
    k = len(centroids)
    
    # 剪枝条件 1: 如果当前上界 <= 到最近的其他质心距离的一半，绝对不会改变所属簇
    if u_x <= s_array[c_x]:
        return (id_val, x, c_x, u_x, l_x)
        
    u_x_is_stale = True  # 标记上界是否准确（是否等于真实的距离）
    
    for j in range(k):
        if j == c_x:
            continue
            
        # 剪枝条件 2: 上界 <= 下界，或 上界 <= 质心间距的一半
        if u_x <= l_x[j] or u_x <= dist_matrix[c_x, j] / 2.0:
            continue
            
        # 上述条件不满足，需要收紧上界：计算真实 d(x, c_x)
        if u_x_is_stale:
            actual_dist_cx = np.linalg.norm(x - centroids[c_x])
            dist_calc_acc += 1
            u_x = actual_dist_cx
            l_x[c_x] = actual_dist_cx
            u_x_is_stale = False
            
            # 收紧上界后再次检查条件
            if u_x <= l_x[j] or u_x <= dist_matrix[c_x, j] / 2.0:
                continue
                
        # 终极情况：必须计算当前点到其他质心 j 的真实距离
        actual_dist_j = np.linalg.norm(x - centroids[j])
        dist_calc_acc += 1
        l_x[j] = actual_dist_j
        
        # 如果距离更小，重新分配质心
        if actual_dist_j < u_x:
            c_x = j
            u_x = actual_dist_j
            u_x_is_stale = False # 新的上界现在是准确的
            
    return (id_val, x, c_x, u_x, l_x)

def update_bounds_logic(row, bc_centroid_shifts):
    """Driver更新质心后，Worker更新点的上下界"""
    id_val, x, c_x, u_x, l_x = row
    shifts = bc_centroid_shifts.value
    k = len(shifts)
    
    # 上界随着所属质心的移动而放宽
    u_x = u_x + shifts[c_x]
    
    # 下界随着其他质心的移动而放宽(减小)
    for j in range(k):
        l_x[j] = max(0.0, l_x[j] - shifts[j])
        
    return (id_val, x, c_x, u_x, l_x)

def run_elkan_kmeans(spark, data_path, k=5, max_iterations=20, tol=1e-4):
    sc = spark.sparkContext
    print(">>> 正在加载数据并初始化 Elkan 状态...")
    data_rdd = sc.pickleFile(data_path).cache()
    
    centroids = data_rdd.map(lambda x: x[1]).takeSample(False, k, seed=42)
    
    # 距离计算次数累加器（用于报告里的量化对比）
    dist_calc_acc = sc.accumulator(0)
    
    # 第 0 轮：初始化完整状态 RDD
    bc_centroids = sc.broadcast(centroids)
    state_rdd = data_rdd.map(lambda row: initialize_state(row, bc_centroids.value)).cache()
    # 强制触发行动操作，确保初始化完成
    state_rdd.count()
    
    for i in range(max_iterations):
        start_time = time.time()
        dist_calc_acc.value = 0 # 重置本轮计算器
        
        # 1. Driver端预计算并广播质心距离矩阵
        dist_matrix, s_array = compute_centroid_distances(centroids)
        bc_centroids = sc.broadcast(centroids)
        bc_dist_matrix = sc.broadcast(dist_matrix)
        bc_s_array = sc.broadcast(s_array)
        
        # 2. Worker 执行 Elkan Map 剪枝和分配逻辑
        assigned_rdd = state_rdd.map(
            lambda row: elkan_map_logic(row, bc_centroids, bc_dist_matrix, bc_s_array, dist_calc_acc)
        )
        
        # 3. Reduce 阶段：提取用于重新计算中心点的数据 (c_x, (features, 1))
        reduced_rdd = assigned_rdd.map(lambda row: (row[2], (row[1], 1))) \
                                  .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        
        new_centroids_data = reduced_rdd.mapValues(lambda val: val[0] / val[1]).collect()
        
        # 生成新的质心列表
        new_centroids = np.zeros_like(centroids)
        for idx, vec in new_centroids_data:
            new_centroids[idx] = vec
            
        # 4. 计算质心偏移量 Shift
        centroid_shifts = [np.linalg.norm(centroids[j] - new_centroids[j]) for j in range(k)]
        shift_sum = sum(centroid_shifts)
        
        # 5. 更新状态 RDD (非常关键的 RDD Lineage 管理)
        bc_centroid_shifts = sc.broadcast(centroid_shifts)
        new_state_rdd = assigned_rdd.map(lambda row: update_bounds_logic(row, bc_centroid_shifts)).cache()
        
        # 触发新的缓存，释放旧的缓存
        new_state_rdd.count()
        state_rdd.unpersist()
        state_rdd = new_state_rdd
        centroids = new_centroids
        
        end_time = time.time()
        print(f"Iter {i+1}: Shift={shift_sum:.6f}, 距离计算次数: {dist_calc_acc.value}, 耗时: {end_time - start_time:.2f}s")
        
        if shift_sum < tol:
            print(">>> 模型已收敛！")
            break
            
    return centroids

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Elkan_KMeans_From_Scratch").getOrCreate()
    final_centroids = run_elkan_kmeans(spark, data_path="preprocessed_data_test", k=5)
    print("\n>>> Elkan 最终质心:")
    for idx, c in enumerate(final_centroids):
        print(f"Cluster {idx}: {c}")
    spark.stop()
    