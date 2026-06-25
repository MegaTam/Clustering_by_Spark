import argparse
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour


def get_project_root():
    file_path = globals().get("__file__")
    if file_path:
        return Path(file_path).resolve().parent.parent
    return Path("/Workspace/Users/theochengworkinginbox@gmail.com/Clustering_by_Spark")


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT_PATH = "/Volumes/workspace/default/msbd5003_data/raw/train.csv"
DEFAULT_OUTPUT_PATH = "/Volumes/workspace/default/msbd5003_data/processed/preprocessed_data_full"

# TEST_MODE = True  # True: 测试模式 (截取少量数据) / False: 全量模式
TEST_MODE = False


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


def default_scaler_output_path(output_path):
    output_str = str(output_path)
    if output_str.endswith("/"):
        output_str = output_str.rstrip("/")
    return f"{output_str}_scaler_json"


def save_scaler_metadata(spark, scaler_output_path, min_vals, max_vals):
    scaler_output_path = resolve_spark_path(scaler_output_path)
    delete_output_path_if_exists(spark, scaler_output_path)

    scaler_df = spark.createDataFrame([
        {
            "min_vals": [float(v) for v in np.asarray(min_vals).tolist()],
            "max_vals": [float(v) for v in np.asarray(max_vals).tolist()],
        }
    ])
    scaler_df.write.mode("overwrite").json(scaler_output_path)
    print(f">>> [持久化] 归一化参数已保存至目录: {scaler_output_path}")


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Preprocess NYC taxi data for clustering.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to the raw CSV input.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Directory where the preprocessed Pickle RDD should be written.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Limit the input to 1000 rows for quick validation runs.",
    )
    parser.add_argument(
        "--scaler-output",
        default=None,
        help="Directory where the Min-Max scaler metadata should be written.",
    )
    return parser


def preprocess_data(input_path, output_path=None, scaler_output_path=None, test_mode=TEST_MODE):
    input_path = resolve_spark_path(input_path)
    output_path = output_path or DEFAULT_OUTPUT_PATH
    spark_output_path = resolve_spark_path(output_path)
    local_output_path = resolve_local_path(output_path)
    scaler_output_path = scaler_output_path or default_scaler_output_path(output_path)

    if local_output_path is not None:
        local_output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 初始化 Spark Session
    spark = SparkSession.builder.appName("NYC_Taxi_Clustering_Preprocess").getOrCreate()

    if test_mode:
        print(">>> [环境] 当前为 TEST_MODE: 限制资源 (4核, 4G) <<<")
    else:
        print(">>> [环境] 当前为 FULL_MODE: 使用当前 Spark 集群配置处理全量数据 <<<")

    # 2. 读取原始数据
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # 根据测试开关截取数据
    if test_mode:
        df = df.limit(1000)
        print(">>> [数据] TEST_MODE 开启: 仅截取前 1000 条原始数据 <<<")

    # 3. 数据清洗 (过滤异常经纬度)
    cleaned_df = df.filter(
        (col("pickup_longitude") > -74.25) & (col("pickup_longitude") < -73.7) &
        (col("pickup_latitude") > 40.5) & (col("pickup_latitude") < 40.9) &
        (col("passenger_count") > 0)
    )

    # 4. 特征提取
    cleaned_df = cleaned_df.withColumn("pickup_hour", hour(col("pickup_datetime")))
    feature_cols = ["pickup_longitude", "pickup_latitude"]

    # 5. 转换为 RDD 格式: (id, np.array([lon, lat]))
    data_rdd = cleaned_df.rdd.map(lambda row: (
        row["id"],
        np.array([row[c] for c in feature_cols])
    ))

    # 6. 数据标准化 (Min-Max Scaling)
    stats = data_rdd.map(lambda x: x[1]).aggregate(
        (np.ones(len(feature_cols)) * np.inf, np.ones(len(feature_cols)) * -np.inf),
        lambda acc, x: (np.minimum(acc[0], x), np.maximum(acc[1], x)),
        lambda acc1, acc2: (np.minimum(acc1[0], acc2[0]), np.maximum(acc1[1], acc2[1]))
    )

    min_vals, max_vals = stats
    sc = spark.sparkContext
    broadcast_min = sc.broadcast(min_vals)
    broadcast_max = sc.broadcast(max_vals)

    def scale_features(point):
        diff = broadcast_max.value - broadcast_min.value
        diff[diff == 0] = 1e-9
        return (point - broadcast_min.value) / diff

    final_rdd = data_rdd.map(lambda x: (x[0], scale_features(x[1])))
    final_rdd.cache()

    valid_count = final_rdd.count()
    print(f">>> [完成] 预处理结束！有效数据记录数: {valid_count} 条。 <<<")

    # 7. 动态落盘保存 (为 K-Means 算法做准备)
    # Spark 保存文件时要求目标目录必须不存在,否则报错。因此先进行清理。
    delete_output_path_if_exists(spark, output_path)

    # 将 RDD 序列化为 Pickle 格式保存 (支持 Numpy 数组)
    final_rdd.saveAsPickleFile(spark_output_path)
    print(f">>> [持久化] 数据已成功保存至目录: {spark_output_path}")
    save_scaler_metadata(spark, scaler_output_path, min_vals, max_vals)

    return final_rdd


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    preprocess_data(
        args.input,
        output_path=args.output,
        scaler_output_path=args.scaler_output,
        test_mode=args.test_mode,
    )
