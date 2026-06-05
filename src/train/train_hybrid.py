import argparse
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import DoubleType
from src.utils.spark import get_spark

def cosine_udf():
    # cosine between sparse vectors
    from pyspark.sql.functions import udf
    def cos(a, b):
        if a is None or b is None:
            return 0.0
        # sparse vector dot product
        dot = float(a.dot(b))
        na = float(a.norm(2))
        nb = float(b.norm(2))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)
    return udf(cos, DoubleType())

def main(processed_dir: str, als_dir: str, out_dir: str, k: int, w_cf: float, w_cb: float):
    spark = get_spark("build-hybrid")

    ratings = spark.read.parquet(f"{processed_dir}/ratings")
    movies = spark.read.parquet(f"{processed_dir}/movies")
    content = spark.read.parquet(f"{processed_dir}/movie_content_features").select("movie_id", "features")

    als = ALSModel.load(als_dir)

    # 1) candidates from ALS
    users = ratings.select("user_id").distinct()
    recs = als.recommendForUserSubset(users, k * 5)  # oversample candidates
    recs = recs.select("user_id", F.explode("recommendations").alias("rec")) \
               .select("user_id", F.col("rec.movie_id").alias("movie_id"), F.col("rec.rating").alias("als_score"))

    # 2) build user profile vector from liked movies (rating >= 4)
    liked = ratings.filter(F.col("rating") >= 4.0).select("user_id", "movie_id")
    liked_feats = liked.join(content, "movie_id", "inner")

    # average vectors: use aggregate by converting to arrays is heavy.
    # Approx: sum vectors via Spark SQL "aggregate" not available for ML vectors in vanilla.
    # Workaround: use Spark's Summarizer (best) if using DenseVector; but we have SparseVector.
    # Practical approach: sample + use UDF to sum sparse vectors.
    from pyspark.sql.functions import udf
    from pyspark.ml.linalg import Vectors

    def add_sparse(vs):
        if not vs:
            return None
        acc = None
        for v in vs:
            if v is None:
                continue
            acc = v if acc is None else Vectors.dense(acc.toArray() + v.toArray())
        return acc

    add_sparse_udf = udf(add_sparse, VectorUDT())

    user_profile = (
        liked_feats.groupBy("user_id")
        .agg(F.collect_list("features").alias("feat_list"))
        .withColumn("user_vec", add_sparse_udf("feat_list"))
        .select("user_id", "user_vec")
    )

    cos = cosine_udf()

    # 3) compute content similarity for candidates
    cand = recs.join(content, "movie_id", "left").join(user_profile, "user_id", "left")
    cand = cand.withColumn("content_score", cos(F.col("user_vec"), F.col("features")))

    # 4) normalize scores per user (min-max)
    w = Window.partitionBy("user_id")
    cand = cand.withColumn("als_min", F.min("als_score").over(w)) \
               .withColumn("als_max", F.max("als_score").over(w)) \
               .withColumn("cb_min", F.min("content_score").over(w)) \
               .withColumn("cb_max", F.max("content_score").over(w))

    cand = cand.withColumn(
        "als_norm",
        F.when(F.col("als_max") > F.col("als_min"),
               (F.col("als_score") - F.col("als_min")) / (F.col("als_max") - F.col("als_min"))
        ).otherwise(F.lit(0.0))
    ).withColumn(
        "cb_norm",
        F.when(F.col("cb_max") > F.col("cb_min"),
               (F.col("content_score") - F.col("cb_min")) / (F.col("cb_max") - F.col("cb_min"))
        ).otherwise(F.lit(0.0))
    )

    cand = cand.withColumn("hybrid_score", F.lit(w_cf) * F.col("als_norm") + F.lit(w_cb) * F.col("cb_norm"))

    # 5) top-K
    w2 = Window.partitionBy("user_id").orderBy(F.col("hybrid_score").desc())
    topk = cand.withColumn("rank", F.row_number().over(w2)).filter(F.col("rank") <= k)

    topk = topk.join(movies, "movie_id", "left") \
               .select("user_id", "movie_id", "title", "genres", "als_score", "content_score", "hybrid_score", "rank")

    topk.write.mode("overwrite").parquet(out_dir)
    spark.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--als_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--w_cf", type=float, default=0.75)
    p.add_argument("--w_cb", type=float, default=0.25)
    args = p.parse_args()
    main(args.processed_dir, args.als_dir, args.out_dir, args.k, args.w_cf, args.w_cb)
