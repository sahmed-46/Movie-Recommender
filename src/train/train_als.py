import argparse
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from src.utils.spark import get_spark

from pyspark.sql import functions as F

def train_test_split_by_time(ratings, test_ratio=0.2):
    # Split by timestamp quantile: newest interactions go to test
    cutoff = ratings.approxQuantile("timestamp", [1.0 - test_ratio], 0.001)[0]
    train = ratings.filter(F.col("timestamp") <= F.lit(cutoff))
    test  = ratings.filter(F.col("timestamp") >  F.lit(cutoff))
    return train, test


def main(processed_dir: str, model_dir: str):
    spark = get_spark("train-als")

    ratings = spark.read.parquet(f"{processed_dir}/ratings").cache()

    train, test = train_test_split_by_time(ratings, test_ratio=0.2)
    train.cache(); test.cache()

    als = ALS(
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        rank=40,
        regParam=0.08,
        maxIter=10,
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=False,
    )

    model = als.fit(train)

    preds = model.transform(test)
    rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction").evaluate(preds)

    print(f"ALS RMSE: {rmse:.4f}")

    model.write().overwrite().save(model_dir)
    spark.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--model_dir", required=True)
    args = p.parse_args()
    main(args.processed_dir, args.model_dir)
