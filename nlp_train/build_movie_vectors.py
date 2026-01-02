import os
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMDB = os.path.join(ROOT, "module2", "tmdb_5000_movies.csv")
OUT_DIR = os.path.join(ROOT, "artifacts")
MODEL_PATH = os.path.join(OUT_DIR, "doc2vec.model")

VEC_PATH = os.path.join(OUT_DIR, "movie_vectors.npy")
META_PATH = os.path.join(OUT_DIR, "movie_meta.csv")

def main():
    df = pd.read_csv(TMDB)
    df = df.dropna(subset=["overview"]).copy()

    model = Doc2Vec.load(MODEL_PATH)

    ids = []
    titles = []
    vecs = []

    for _, row in df.iterrows():
        mid = int(row["id"])
        tag = str(mid)
        if tag in model.dv:
            ids.append(mid)
            titles.append(str(row.get("title", "")))
            vecs.append(model.dv[tag])

    M = np.vstack(vecs).astype(np.float32)

    np.save(VEC_PATH, M)
    pd.DataFrame({"id": ids, "title": titles}).to_csv(META_PATH, index=False)

    print("Saved:", VEC_PATH)
    print("Saved:", META_PATH)
    print("Shape:", M.shape)

if __name__ == "__main__":
    main()
