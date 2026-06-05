from pyspark.sql import SparkSession
import os, sys

def get_spark(app_name: str = "movie-recsys") -> SparkSession:
    python_exe = os.environ.get("PYSPARK_PYTHON") or sys.executable
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[2]")
        .config("spark.pyspark.python", python_exe)
        .config("spark.pyspark.driver.python", python_exe)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
