import os
import re
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(ROOT, "module2", "final_movie_emotions_rt_enriched.csv")
OUT_DIR = os.path.join(ROOT, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "doc2vec.model")

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def tokenize(s: str):
    if not isinstance(s, str):
        return []
    return TOKEN_RE.findall(s.lower())

def main():
    df = pd.read_csv(IN_PATH)
    df["id"] = df["id"].astype(str)
    df["title"] = df["title"].astype(str)
    df["genre"] = df.get("genre", "").astype(str)
    df["reviews"] = df.get("reviews", "").astype(str)

    docs = []
    for _, r in df.iterrows():
        # training text: title + genre + reviews
        text = f"{r['title']} {r['genre']} {r['reviews']}"
        tokens = tokenize(text)
        if len(tokens) < 5:
            continue
        # tag MUST be string id
        docs.append(TaggedDocument(words=tokens, tags=[r["id"]]))

    model = Doc2Vec(
        vector_size=200,
        window=8,
        min_count=2,
        workers=4,
        epochs=30,
        dm=1,  # PV-DM
    )
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

    model.save(MODEL_PATH)
    print("Saved:", MODEL_PATH, "docs:", len(docs))

if __name__ == "__main__":
    main()
