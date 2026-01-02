import os
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(ROOT, "module2", "final_movie_emotions_rt_enriched.csv")

OUT_DIR = os.path.join(ROOT, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "doc2vec.model")
META_OUT = os.path.join(OUT_DIR, "movie_meta.csv")
VEC_OUT = os.path.join(OUT_DIR, "movie_vectors.npy")

def main():
    df = pd.read_csv(IN_PATH)
    df["id"] = df["id"].astype(str)
    df["title"] = df["title"].astype(str)

    meta = df[["id", "title"]].drop_duplicates("id").reset_index(drop=True)

    model = Doc2Vec.load(MODEL_PATH)

    vecs = []
    keep = []
    dropped = 0

    for mid in meta["id"].tolist():
        if mid in model.dv:
            vecs.append(model.dv[mid])
            keep.append(True)
        else:
            keep.append(False)
            dropped += 1

    meta = meta[keep].reset_index(drop=True)
    M = np.vstack(vecs).astype(np.float32)

    meta.to_csv(META_OUT, index=False)
    np.save(VEC_OUT, M)
    print("Saved:", META_OUT)
    print("Saved:", VEC_OUT, "shape=", M.shape)
    if dropped:
        print("Dropped (no vectors):", dropped)

if __name__ == "__main__":
    main()
