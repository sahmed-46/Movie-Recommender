import argparse
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType, LongType
from src.utils.spark import get_spark

def main(raw_dir: str, out_dir: str):
    spark = get_spark("etl-prepare")

    ratings = (
        spark.read.option("header", True).csv(f"{raw_dir}/ratings.csv")
        .select(
            F.col("userId").cast(IntegerType()).alias("user_id"),
            F.col("movieId").cast(IntegerType()).alias("movie_id"),
            F.col("rating").cast(FloatType()).alias("rating"),
            F.col("timestamp").cast(LongType()).alias("timestamp"),
        )
        .dropna()
        .dropDuplicates(["user_id", "movie_id", "timestamp"])
    )

    movies = (
        spark.read.option("header", True).csv(f"{raw_dir}/movies.csv")
        .select(
            F.col("movieId").cast(IntegerType()).alias("movie_id"),
            F.col("title").alias("title"),
            F.col("genres").alias("genres"),
        )
        .dropna(subset=["movie_id", "title"])
        .dropDuplicates(["movie_id"])
    )

    # optional tags
    tags_path = f"{raw_dir}/tags.csv"
    try:
        tags = (
            spark.read.option("header", True).csv(tags_path)
            .select(
                F.col("userId").cast(IntegerType()).alias("user_id"),
                F.col("movieId").cast(IntegerType()).alias("movie_id"),
                F.col("tag").alias("tag"),
                F.col("timestamp").cast(LongType()).alias("timestamp"),
            )
            .dropna()
        )
        tags.write.mode("overwrite").parquet(f"{out_dir}/tags")
    except Exception:
        tags = None

    ratings.write.mode("overwrite").parquet(f"{out_dir}/ratings")
    movies.write.mode("overwrite").parquet(f"{out_dir}/movies")

    spark.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    main(args.raw_dir, args.out_dir)
