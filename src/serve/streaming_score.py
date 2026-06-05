import argparse
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType
from pyspark.ml.recommendation import ALSModel
from src.utils.spark import get_spark

def main(processed_dir: str, als_dir: str, in_dir: str, out_dir: str, k: int):
    spark = get_spark("streaming-score")

    als = ALSModel.load(als_dir)

    schema = StructType([
        StructField("user_id", IntegerType()),
        StructField("movie_id", IntegerType()),
        StructField("rating", FloatType()),
        StructField("timestamp", LongType()),
    ])

    events = (
        spark.readStream.schema(schema)
        .json(in_dir)  # drop JSON files into this folder
    )

    # Keep a small “active users” table from events; recommend for them each micro-batch.
    active_users = events.select("user_id").distinct()

    def foreach_batch(batch_df, batch_id: int):
        users = batch_df.select("user_id").distinct()
        if users.rdd.isEmpty():
            return
        recs = als.recommendForUserSubset(users, k)
        recs.write.mode("append").parquet(out_dir)

    query = (
        active_users.writeStream
        .foreachBatch(foreach_batch)
        .outputMode("update")
        .option("checkpointLocation", f"{out_dir}/_chk")
        .start()
    )

    query.awaitTermination()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=False, default="data/processed")
    p.add_argument("--als_dir", required=True)
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k", type=int, default=20)
    args = p.parse_args()
    main(args.processed_dir, args.als_dir, args.in_dir, args.out_dir, args.k)
