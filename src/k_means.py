import argparse
import json
from pathlib import Path
import time

import numpy as np
from pyspark.sql import Row
from pyspark.sql import SparkSession


def get_project_root():
    file_path = globals().get("__file__")
    if file_path:
        return Path(file_path).resolve().parent.parent
    return Path("/Workspace/Users/theochengworkinginbox@gmail.com/Clustering_by_Spark")


PROJECT_ROOT = get_project_root()
DEFAULT_INPUT_PATH = "/Volumes/workspace/default/msbd5003_data/processed/preprocessed_data_full"
DEFAULT_OUTPUT_PATH = "/Volumes/workspace/default/msbd5003_data/results/kmeans_centroids_json"


def resolve_project_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Run K-Means clustering on preprocessed data.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to the preprocessed Pickle RDD.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Directory for centroid outputs.")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters.")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum number of iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance.")
    return parser


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
    print(">>> 正在加载预处理后的数据...")
    data_rdd = sc.pickleFile(str(data_path))

    # 仅提取特征向量用于聚类计算，格式为: np.array([lon, lat])
    features_rdd = data_rdd.map(lambda x: x[1]).cache()

    # 2. 随机初始化质心 (在 Driver 端进行)
    # takeSample 动作会从集群中拉取样本到本地内存
    print(f">>> 随机初始化 {k} 个质心...")
    centroids = features_rdd.takeSample(False, k, seed=42)

    for i in range(max_iterations):
        start_time = time.time()

        # 3. 广播当前质心到所有 Worker 节点
        # 这一步极其重要，避免了大量的网络传输开销
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
        print(f"Iteration {i + 1} completed in {end_time - start_time:.2f}s, centroid shift: {shift:.6f}")

        if shift < tol:
            print(">>> 模型已收敛！")
            break

    return centroids


def save_centroids(spark, centroids, output_path):
    output_path = resolve_project_path(output_path)
    rows = [
        Row(cluster_id=int(idx), centroid=json.dumps(np.asarray(centroid).tolist()))
        for idx, centroid in enumerate(centroids)
    ]
    centroid_df = spark.createDataFrame(rows)
    centroid_df.write.mode("overwrite").json(str(output_path))
    print(f">>> 已将 K-Means 质心保存到: {output_path}")


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    spark = SparkSession.builder.appName("Standard_KMeans_From_Scratch").getOrCreate()

    final_centroids = run_kmeans(
        spark,
        data_path=args.input,
        k=args.k,
        max_iterations=args.max_iterations,
        tol=args.tol,
    )

    print("\n>>> 最终质心坐标:")
    for idx, c in enumerate(final_centroids):
        print(f"Cluster {idx}: {c}")

    save_centroids(spark, final_centroids, args.output)
    spark.stop()
