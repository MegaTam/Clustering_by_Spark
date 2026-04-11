import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour
import numpy as np


# TEST_MODE = True  # True: 测试模式 (截取少量数据) / False: 全量模式
TEST_MODE = False


def preprocess_data(input_path):
    # 1. 初始化 Spark Session
    builder = SparkSession.builder.appName("NYC_Taxi_Clustering_Preprocess")
    
    if TEST_MODE:
        builder = builder.master("local[4]") \
                         .config("spark.driver.memory", "4g")
        print(">>> [环境] 当前为 TEST_MODE: 限制资源 (4核, 4G) <<<")
    else:
        # 全量单机模式：调用服务器的强大算力
        # 使用 16 个核心，分配 16GB 内存（对这台 400GB 的机器来说依然很安全）
        builder = builder.master("local[16]") \
                         .config("spark.driver.memory", "16g") \
                         .config("spark.executor.memory", "16g") \
                         .config("spark.memory.offHeap.enabled", "true") \
                         .config("spark.memory.offHeap.size", "4g")
        print(">>> [环境] 当前为 FULL_MODE: 分配较高资源 (16核, 16G) 处理全量数据 <<<")

    spark = builder.getOrCreate()

    # 2. 读取原始数据
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # 根据测试开关截取数据
    if TEST_MODE:
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
    # 根据全局变量决定输出的文件夹名称
    output_dir = "preprocessed_data_test" if TEST_MODE else "preprocessed_data_full"
    
    # Spark 保存文件时要求目标目录必须不存在,否则报错。因此先进行清理。
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f">>> [清理] 发现同名旧目录，已删除: {output_dir}")

    # 将 RDD 序列化为 Pickle 格式保存 (支持 Numpy 数组)
    final_rdd.saveAsPickleFile(output_dir)
    print(f">>> [持久化] 数据已成功保存至目录: {output_dir}")

    return final_rdd

if __name__ == "__main__":
    train_rdd = preprocess_data("train.csv")
