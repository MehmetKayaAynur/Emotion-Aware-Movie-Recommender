# nlp_train/train_doc2vec.py
import os
import re
import ast
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMDB = os.path.join(ROOT, "module2", "tmdb_5000_movies.csv")
OUT_DIR = os.path.join(ROOT, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "doc2vec.model")


def safe_parse_jsonlike(s):
    # genres/keywords bazen "[{'id':..,'name':..}, ...]" gibi string gelir
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        x = ast.literal_eval(s)
        if isinstance(x, list):
            return x
    except Exception:
        pass
    return []


def extract_names(list_of_dicts):
    out = []
    for d in list_of_dicts:
        if isinstance(d, dict) and "name" in d:
            out.append(str(d["name"]))
    return out


_token_re = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return _token_re.findall(text.lower())


def s(x):
    """NaN/None -> '', diğerlerini stringe çevir."""
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def make_doc(row):
    # Bunlar bazen NaN(float) gelebiliyor -> s() ile temizle
    overview = s(row.get("overview", ""))
    tagline = s(row.get("tagline", ""))
    title = s(row.get("title", ""))

    genres = extract_names(safe_parse_jsonlike(row.get("genres", "")))
    keywords = extract_names(safe_parse_jsonlike(row.get("keywords", "")))

    # title/tagline'a biraz ağırlık (2x)
    text = " ".join(
        [
            title,
            title,
            tagline,
            tagline,
            " ".join(genres),
            " ".join(keywords),
            overview,
        ]
    )
    return tokenize(text)


def main():
    df = pd.read_csv(TMDB)

    # NaN’ları daha en başta boş stringe çevir (ekstra güvenlik)
    for col in ["overview", "tagline", "title", "genres", "keywords"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # boş/çok kısa overview olanları at
    df = df[df["overview"].astype(str).str.len() > 20].copy()

    documents = []
    for _, row in df.iterrows():
        movie_id = int(row["id"])
        tokens = make_doc(row)
        if len(tokens) >= 10:
            documents.append(TaggedDocument(words=tokens, tags=[str(movie_id)]))

    print("Docs:", len(documents))

    model = Doc2Vec(
        vector_size=200,
        window=8,
        min_count=2,
        workers=4,
        epochs=30,
        dm=1,  # PV-DM
        negative=10,
        sample=1e-4,
    )

    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    model.save(MODEL_PATH)
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
