from pathlib import Path
import time

import numpy as np
from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def euclidean_distance(point, centroid):
    """计算欧氏距离"""
    return np.linalg.norm(point - centroid)


def closest_centroid(point, centroids):
    """寻找距离当前点最近的质心索引"""
    distances = [euclidean_distance(point, c) for c in centroids]
    return np.argmin(distances)


def run_kmeans(spark, data_path, k=5, max_iterations=20, tol=1e-4):
    sc = spark.sparkContext
    data_path = resolve_project_path(data_path)
    # 1. 读取 Pickle 格式的 RDD
    # 格式为: (id, np.array([lon, lat]))
    print(" [数据] 正在加载预处理后的数据... [数据] ")
    data_rdd = sc.pickleFile(str(data_path))

    # 仅提取特征向量用于聚类计算，格式为: np.array([lon, lat])
    features_rdd = data_rdd.map(lambda x: x[1]).cache()

    # 2. 随机初始化质心 (在 Driver 端进行)
    # takeSample 动作会从集群中拉取样本到本地内存
    print(f" [初始化] 随机初始化 {k} 个质心... [初始化] ")
    centroids = features_rdd.takeSample(False, k, seed=42)

    for i in range(max_iterations):
        start_time = time.time()
        # 3. 广播当前质心到所有 Worker 节点
        bc_centroids = sc.broadcast(centroids)
        
        # 4. Map 阶段：计算每个点所属的质心
        mapped_rdd = features_rdd.map(
            lambda x: (closest_centroid(x, bc_centroids.value), (x, 1))
        )

        # 5. Reduce 阶段：按质心索引聚合
        reduced_rdd = mapped_rdd.reduceByKey(
            lambda a, b: (a[0] + b[0], a[1] + b[1])
        )

        # 6. 计算新的质心并拉取到 Driver 端
        new_centroids_data = reduced_rdd.mapValues(
            lambda val: val[0] / val[1]
        ).collect()

        # 更新质心列表 (需要按照 index 排序确保顺序不乱)
        new_centroids_data.sort(key=lambda x: x[0])
        new_centroids = [x[1] for x in new_centroids_data]

        # 7. 收敛检查
        shift = sum(euclidean_distance(centroids[j], new_centroids[j]) for j in range(k))
        centroids = new_centroids

        end_time = time.time()
        print(f" [迭代] Iteration {i + 1} completed in {end_time - start_time:.2f}s, centroid shift: {shift:.6f} [迭代] ")

        if shift < tol:
            print(" [收敛] 模型已收敛！ [收敛] ")
            break

    return centroids


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Standard_KMeans_From_Scratch") \
        .getOrCreate()

    final_centroids = run_kmeans(
        spark,
        data_path="data/preprocessed_data_test",
        k=5,
        max_iterations=20
    )

    print("\n [结果] 最终质心坐标:")
    for idx, c in enumerate(final_centroids):
        print(f" [结果] Cluster {idx}: {c} [结果] ")

    spark.stop()
