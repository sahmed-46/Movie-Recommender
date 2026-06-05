import argparse
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from src.utils.spark import get_spark

def main(processed_dir: str, out_dir: str):
    spark = get_spark("build-content-features")

    movies = spark.read.parquet(f"{processed_dir}/movies")
    ratings = spark.read.parquet(f"{processed_dir}/ratings")

    # optional tags -> aggregate per movie
    try:
        tags = spark.read.parquet(f"{processed_dir}/tags")
        tags_agg = tags.groupBy("movie_id").agg(F.concat_ws(" ", F.collect_list("tag")).alias("tags_text"))
        movies = movies.join(tags_agg, "movie_id", "left")
    except Exception:
        movies = movies.withColumn("tags_text", F.lit(""))

    movies = movies.withColumn(
        "text",
        F.concat_ws(
            " ",
            F.coalesce(F.col("title"), F.lit("")),
            F.regexp_replace(F.coalesce(F.col("genres"), F.lit("")), r"\|", " "),
            F.coalesce(F.col("tags_text"), F.lit("")),
        )
    )

    tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W+")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    tf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=1 << 18)
    idf = IDF(inputCol="tf", outputCol="features")

    pipe = Pipeline(stages=[tokenizer, remover, tf, idf])
    model = pipe.fit(movies)
    feats = model.transform(movies).select("movie_id", "title", "genres", "features")

    feats.write.mode("overwrite").parquet(out_dir)
    # Optionally save pipeline for reuse
    model.write().overwrite().save(f"{out_dir}_pipeline")

    spark.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    main(args.processed_dir, args.out_dir)
