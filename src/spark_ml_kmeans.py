import argparse
import json
from pathlib import Path

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def get_project_root():
    file_path = globals().get("__file__")
    if file_path:
        return Path(file_path).resolve().parent.parent
    return Path("/Workspace/Users/theochengworkinginbox@gmail.com/Clustering_by_Spark")


PROJECT_ROOT = get_project_root()
DEFAULT_INPUT_PATH = "/Volumes/workspace/default/msbd5003_data/processed/preprocessed_data_full"
DEFAULT_OUTPUT_PATH = "/Volumes/workspace/default/msbd5003_data/results/spark_ml_kmeans"


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


def delete_output_path_if_exists(spark, path_value):
    spark_path = resolve_spark_path(path_value)
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    jvm = sc._jvm
    output_path = jvm.org.apache.hadoop.fs.Path(spark_path)
    fs = output_path.getFileSystem(hadoop_conf)

    if fs.exists(output_path):
        fs.delete(output_path, True)
        print(f">>> [清理] 发现同名旧目录，已删除: {spark_path}")


def join_output_path(base_path, child_name):
    base_str = str(base_path).rstrip("/")
    if is_uri_path(base_str):
        return f"{base_str}/{child_name}"
    return str(Path(base_str) / child_name)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Validate clustering results with pyspark.ml.clustering.KMeans on the preprocessed dataset."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to the preprocessed Pickle RDD.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Base directory where centroids, assignments, and metrics subdirectories will be written.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of clusters.")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum number of iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument(
        "--init-mode",
        choices=["random", "k-means||"],
        default="k-means||",
        help="Initialization mode used by Spark MLlib KMeans.",
    )
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
    parser.add_argument(
        "--compute-silhouette",
        action="store_true",
        help="Compute silhouette score with squared Euclidean distance for additional validation.",
    )
    return parser


def limit_preprocessed_rdd(sc, data_rdd, limit=None, test_mode=False):
    effective_limit = 1000 if test_mode and limit is None else limit
    if effective_limit is None:
        return data_rdd, None
    if effective_limit <= 0:
        raise ValueError("--limit must be a positive integer")

    limited_records = data_rdd.take(effective_limit)
    num_slices = max(1, min(sc.defaultParallelism, max(1, len(limited_records) // 1000)))
    print(f">>> [数据] 当前仅使用前 {len(limited_records)} 条预处理记录进行 Spark ML KMeans 聚类。")
    return sc.parallelize(limited_records, numSlices=num_slices), len(limited_records)


def load_preprocessed_dataframe(spark, data_path, limit=None, test_mode=False):
    sc = spark.sparkContext
    data_path = resolve_spark_path(data_path)
    print(">>> 正在加载预处理后的 Pickle RDD 数据...")
    data_rdd = sc.pickleFile(data_path)
    data_rdd, effective_count = limit_preprocessed_rdd(sc, data_rdd, limit=limit, test_mode=test_mode)

    dataset_df = spark.createDataFrame(
        data_rdd.map(
            lambda x: Row(
                point_id=str(x[0]),
                features=Vectors.dense([float(x[1][0]), float(x[1][1])]),
            )
        )
    )
    row_count = effective_count if effective_count is not None else dataset_df.count()
    print(f">>> [数据] Spark MLlib KMeans 输入样本数: {row_count}")
    return dataset_df


def fit_spark_ml_kmeans(
    spark,
    data_path,
    output_path,
    k=5,
    max_iterations=20,
    tol=1e-4,
    seed=42,
    init_mode="k-means||",
    limit=None,
    test_mode=False,
    compute_silhouette=False,
):
    dataset_df = load_preprocessed_dataframe(spark, data_path, limit=limit, test_mode=test_mode).cache()

    model = KMeans(
        featuresCol="features",
        predictionCol="prediction",
        k=k,
        maxIter=max_iterations,
        tol=tol,
        seed=seed,
        initMode=init_mode,
    ).fit(dataset_df)

    transformed_df = model.transform(dataset_df).cache()
    assignments_df = transformed_df.select(
        "point_id",
        F.col("prediction").cast("int").alias("cluster_id"),
    )

    cluster_sizes_df = assignments_df.groupBy("cluster_id").count().orderBy("cluster_id")
    cluster_sizes = [
        {"cluster_id": int(row["cluster_id"]), "count": int(row["count"])}
        for row in cluster_sizes_df.collect()
    ]

    silhouette_score = None
    if compute_silhouette:
        evaluator = ClusteringEvaluator(
            featuresCol="features",
            predictionCol="prediction",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean",
        )
        silhouette_score = float(evaluator.evaluate(transformed_df))

    summary = model.summary
    training_cost = float(summary.trainingCost)

    centroids_output = join_output_path(output_path, "centroids_json")
    assignments_output = join_output_path(output_path, "assignments_parquet")
    metrics_output = join_output_path(output_path, "metrics_json")

    delete_output_path_if_exists(spark, centroids_output)
    delete_output_path_if_exists(spark, assignments_output)
    delete_output_path_if_exists(spark, metrics_output)

    centers = model.clusterCenters()
    centroids_df = spark.createDataFrame(
        [
            Row(
                cluster_id=int(idx),
                centroid=json.dumps([float(value) for value in center]),
            )
            for idx, center in enumerate(centers)
        ]
    )
    centroids_df.write.mode("overwrite").json(resolve_spark_path(centroids_output))
    assignments_df.write.mode("overwrite").parquet(resolve_spark_path(assignments_output))

    metrics_df = spark.createDataFrame(
        [
            Row(
                algorithm="pyspark.ml.clustering.KMeans",
                k=int(k),
                max_iterations=int(max_iterations),
                tol=float(tol),
                seed=int(seed),
                init_mode=init_mode,
                limit=int(limit) if limit is not None else None,
                test_mode=bool(test_mode),
                training_cost=training_cost,
                silhouette_score=silhouette_score,
                cluster_sizes=json.dumps(cluster_sizes),
            )
        ]
    )
    metrics_df.write.mode("overwrite").json(resolve_spark_path(metrics_output))

    print(f">>> 已将 Spark MLlib KMeans 质心保存到: {resolve_spark_path(centroids_output)}")
    print(f">>> 已将 Spark MLlib KMeans 样本簇标签保存到: {resolve_spark_path(assignments_output)}")
    print(f">>> 已将 Spark MLlib KMeans 指标保存到: {resolve_spark_path(metrics_output)}")
    print(f">>> 训练代价 trainingCost: {training_cost:.6f}")
    if silhouette_score is not None:
        print(f">>> 轮廓系数 silhouette: {silhouette_score:.6f}")
    print(">>> 每个簇的样本数:")
    for item in cluster_sizes:
        print(f"Cluster {item['cluster_id']}: {item['count']}")


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    spark = SparkSession.builder.appName("Spark_MLlib_KMeans_Validation").getOrCreate()
    fit_spark_ml_kmeans(
        spark,
        data_path=args.input,
        output_path=args.output,
        k=args.k,
        max_iterations=args.max_iterations,
        tol=args.tol,
        seed=args.seed,
        init_mode=args.init_mode,
        limit=args.limit,
        test_mode=args.test_mode,
        compute_silhouette=args.compute_silhouette,
    )
    spark.stop()
