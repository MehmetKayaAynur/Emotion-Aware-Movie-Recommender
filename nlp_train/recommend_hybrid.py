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

# RT emotion dataset (enriched)
EMO_CSV = os.path.join(ROOT, "module2", "final_movie_emotions_rt_enriched.csv")

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())

def cosine_topk(q: np.ndarray, M: np.ndarray, k: int = 10):
    q = q.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    sims = Mn @ q
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

def cosine_pairwise(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

class HybridRecommender:
    """
    final = alpha * semantic_sim + (1-alpha) * emotion_sim
    semantic_sim: Doc2Vec cosine
    emotion_sim: cosine(user_emotion_vec, movie_emotion_vec)

    IMPORTANT: RT ids are strings (e.g., 'love_lies') so we use str everywhere.
    """

    def __init__(self):
        self.model = Doc2Vec.load(MODEL_PATH)
        self.M = np.load(VEC_PATH).astype(np.float32)

        # meta: id,title  (id can be slug => str)
        self.meta = pd.read_csv(META_PATH)
        self.meta["id"] = self.meta["id"].astype(str)
        self.meta["title"] = self.meta["title"].astype(str)

        # emotion profiles: id,title + emotion cols (id can be slug => str)
        emo = pd.read_csv(EMO_CSV)
        emo["id"] = emo["id"].astype(str)

        # emotion columns: everything except id,title (+ extra columns like rt_movie_id/genre/reviews varsa onları da çıkarmak gerekir)
        # enriched dosyada ek kolonlar olabileceği için whitelist yaklaşımı daha güvenli:
        base_exclude = {"id", "title", "rt_movie_id", "genre", "releaseDateTheaters", "reviews", "reviews_list"}
        self.emo_cols = [c for c in emo.columns if c not in base_exclude]

        # build map id -> emotion vector
        emo = emo[["id"] + self.emo_cols].copy()
        emo_by_id = emo.set_index("id")

        # emotion matrix aligned with meta order
        ids = self.meta["id"].tolist()
        emat = []
        for mid in ids:
            if mid in emo_by_id.index:
                v = emo_by_id.loc[mid, self.emo_cols].to_numpy(dtype=np.float32)
                emat.append(v)
            else:
                emat.append(np.zeros(len(self.emo_cols), dtype=np.float32))
        self.EM = np.vstack(emat).astype(np.float32)

    def recommend(self, user_text: str, user_emotions: dict, top_k: int = 10, alpha: float = 0.7):
        tokens = tokenize(user_text)
        if len(tokens) < 3:
            return []

        # 1) semantic
        q = self.model.infer_vector(tokens, epochs=30)
        cand_k = min(max(top_k * 8, 50), len(self.M))
        cand_idxs, cand_sem = cosine_topk(q, self.M, k=cand_k)

        # 2) user emotion vector aligned to emo_cols
        u = np.zeros(len(self.emo_cols), dtype=np.float32)
        for i, col in enumerate(self.emo_cols):
            u[i] = float(user_emotions.get(col, 0.0))

        # normalize
        s = float(u.sum())
        if s > 0:
            u = u / s

        # 3) rerank by hybrid score
        results = []
        for idx, sem in zip(cand_idxs, cand_sem):
            idx = int(idx)
            mid = str(self.meta.iloc[idx]["id"])   # ✅ str
            title = str(self.meta.iloc[idx]["title"])

            mv = self.EM[idx]
            emo_sim = cosine_pairwise(u, mv) if (u.sum() > 0 and mv.sum() > 0) else 0.0

            final = float(alpha * float(sem) + (1.0 - alpha) * float(emo_sim))
            results.append({
                "id": mid,
                "title": title,
                "final_score": final,
                "semantic_score": float(sem),
                "emotion_score": float(emo_sim),
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]
