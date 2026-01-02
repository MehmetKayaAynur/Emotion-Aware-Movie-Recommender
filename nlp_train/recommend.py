import os
import re
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "artifacts")
MODEL_PATH = os.path.join(OUT_DIR, "doc2vec.model")
VEC_PATH = os.path.join(OUT_DIR, "movie_vectors.npy")
META_PATH = os.path.join(OUT_DIR, "movie_meta.csv")

_token_re = re.compile(r"[A-Za-z0-9']+")

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return _token_re.findall(text.lower())

def cosine_topk(q: np.ndarray, M: np.ndarray, k: int = 10):
    q = q.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    sims = Mn @ q
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

class Doc2VecRecommender:
    def __init__(self):
        self.model = Doc2Vec.load(MODEL_PATH)
        self.M = np.load(VEC_PATH).astype(np.float32)
        self.meta = pd.read_csv(META_PATH)

    def recommend(self, user_text: str, top_k: int = 5):
        tokens = tokenize(user_text)
        if len(tokens) < 3:
            return []

        q = self.model.infer_vector(tokens, epochs=30)
        idxs, sims = cosine_topk(q, self.M, k=top_k)

        out = []
        for i, s in zip(idxs, sims):
            row = self.meta.iloc[int(i)]
            out.append({"id": int(row["id"]), "title": str(row["title"]), "score": float(s)})
        return out
