import argparse
import json
from pathlib import Path

import numpy as np
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def get_project_root():
    file_path = globals().get("__file__")
    if file_path:
        return Path(file_path).resolve().parent.parent
    return Path("/Workspace/Users/theochengworkinginbox@gmail.com/Clustering_by_Spark")


PROJECT_ROOT = get_project_root()
DEFAULT_INPUT_PATH = "data/test.csv"
DEFAULT_SCALER_PATH = "data/preprocessed_data_full_scaler_json"
DEFAULT_CENTROIDS_PATH = "data/results/kmeans_centroids_json"
DEFAULT_OUTPUT_PATH = "data/results/test_predictions_parquet"


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
    parser = argparse.ArgumentParser(
        description="Assign new points to the nearest saved centroids using training scaler metadata."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to the raw CSV input.")
    parser.add_argument("--scaler", default=DEFAULT_SCALER_PATH, help="Path to the saved scaler metadata.")
    parser.add_argument("--centroids", default=DEFAULT_CENTROIDS_PATH, help="Path to the saved centroid JSON output.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Directory for test-set predictions.")
    return parser


def load_scaler_metadata(spark, scaler_path):
    scaler_path = resolve_spark_path(scaler_path)
    row = spark.read.json(scaler_path).first()
    if row is None:
        raise ValueError(f"Scaler metadata not found at {scaler_path}")

    return np.array(row["min_vals"], dtype=float), np.array(row["max_vals"], dtype=float)


def load_centroids(spark, centroids_path):
    centroids_path = resolve_spark_path(centroids_path)
    centroid_rows = spark.read.json(centroids_path).collect()
    if not centroid_rows:
        raise ValueError(f"Centroids not found at {centroids_path}")

    centroid_rows.sort(key=lambda row: int(row["cluster_id"]))
    return [np.array(json.loads(row["centroid"]), dtype=float) for row in centroid_rows]


def closest_centroid(point, centroids):
    distances = [np.linalg.norm(point - centroid) for centroid in centroids]
    return int(np.argmin(distances))


def predict_assignments(spark, input_path, scaler_path, centroids_path, output_path):
    input_path = resolve_spark_path(input_path)
    output_path = resolve_spark_path(output_path)

    min_vals, max_vals = load_scaler_metadata(spark, scaler_path)
    centroids = load_centroids(spark, centroids_path)

    print(">>> 正在读取测试集并应用训练集归一化参数...")
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    cleaned_df = df.filter(
        (col("pickup_longitude") > -74.25) & (col("pickup_longitude") < -73.7) &
        (col("pickup_latitude") > 40.5) & (col("pickup_latitude") < 40.9) &
        (col("passenger_count") > 0)
    )

    bc_min = spark.sparkContext.broadcast(min_vals)
    bc_max = spark.sparkContext.broadcast(max_vals)
    bc_centroids = spark.sparkContext.broadcast(centroids)

    def scale_and_predict(row):
        point_id = row["id"]
        point = np.array([row["pickup_longitude"], row["pickup_latitude"]], dtype=float)

        diff = bc_max.value - bc_min.value
        diff[diff == 0] = 1e-9
        scaled_point = (point - bc_min.value) / diff
        cluster_id = closest_centroid(scaled_point, bc_centroids.value)
        return Row(
            point_id=str(point_id),
            cluster_id=int(cluster_id),
            scaled_longitude=float(scaled_point[0]),
            scaled_latitude=float(scaled_point[1]),
        )

    prediction_df = spark.createDataFrame(cleaned_df.rdd.map(scale_and_predict))
    prediction_df.write.mode("overwrite").parquet(output_path)
    print(f">>> 已将测试集预测结果保存到: {output_path}")
    return prediction_df


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    spark = SparkSession.builder.appName("Predict_With_Saved_Centroids").getOrCreate()
    prediction_df = predict_assignments(
        spark,
        input_path=args.input,
        scaler_path=args.scaler,
        centroids_path=args.centroids,
        output_path=args.output,
    )
    print(">>> 测试集预测结果示例:")
    prediction_df.show(10, truncate=False)
    spark.stop()
