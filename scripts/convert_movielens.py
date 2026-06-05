import pandas as pd
from pathlib import Path

RAW = Path("data/raw/ml-1m")
OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

def read_dat(path: Path, names):
    # MovieLens 1M sometimes includes non-UTF8 characters in titles.
    # latin-1/cp1252 safely decodes bytes 0x80-0xFF without crashing.
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=names,
        encoding="latin-1",   # <-- key fix
    )

ratings = read_dat(RAW / "ratings.dat", ["userId", "movieId", "rating", "timestamp"])
ratings.to_csv(OUT / "ratings.csv", index=False)

movies = read_dat(RAW / "movies.dat", ["movieId", "title", "genres"])
movies.to_csv(OUT / "movies.csv", index=False)

users = read_dat(RAW / "users.dat", ["userId", "gender", "age", "occupation", "zip"])
users.to_csv(OUT / "users.csv", index=False)

print("Conversion complete:")
print(" - data/raw/ratings.csv")
print(" - data/raw/movies.csv")
print(" - data/raw/users.csv")
