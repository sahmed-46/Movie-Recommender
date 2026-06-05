import argparse
from pyspark.sql import functions as F, Window
from src.utils.spark import get_spark
from src.eval.metrics import precision_at_k, recall_at_k, ndcg_at_k

def main(processed_dir: str, recs_dir: str, k: int):
    spark = get_spark("evaluate")

    ratings = spark.read.parquet(f"{processed_dir}/ratings")

    # Ground truth: latest movie per user (or latest N)
    w = Window.partitionBy("user_id").orderBy(F.col("timestamp").desc())
    truth = ratings.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") <= 1) \
                   .select("user_id", F.collect_set("movie_id").alias("truth"))

    recs = spark.read.parquet(recs_dir) \
            .groupBy("user_id").agg(F.collect_list("movie_id").alias("recs"))

    joined = truth.join(recs, "user_id", "inner").select("user_id", "truth", "recs")

    # evaluate on driver (sample if huge)
    rows = joined.limit(200000).collect()

    p_sum = r_sum = n_sum = 0.0
    n = 0
    for row in rows:
        t = set(row["truth"])
        rec_list = row["recs"]
        p_sum += precision_at_k(rec_list, t, k)
        r_sum += recall_at_k(rec_list, t, k)
        n_sum += ndcg_at_k(rec_list, t, k)
        n += 1

    print(f"Users evaluated: {n}")
    print(f"precision@{k}: {p_sum/n:.4f}")
    print(f"recall@{k}:    {r_sum/n:.4f}")
    print(f"NDCG@{k}:      {n_sum/n:.4f}")

    spark.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--recs_dir", required=True)
    p.add_argument("--k", type=int, default=20)
    args = p.parse_args()
    main(args.processed_dir, args.recs_dir, args.k)
