import os
import sys
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from module1_emotion import predict_emotions
from nlp_train.recommend_hybrid import HybridRecommender

REVIEWS_CSV = os.path.join(ROOT, "module2", "rotten_tomatoes_movie_reviews.csv")
ENRICHED_CSV = os.path.join(ROOT, "module2", "final_movie_emotions_rt_enriched.csv")


@st.cache_resource
def load_rec():
    return HybridRecommender()


@st.cache_resource
def load_enriched_index():
    """
    final_movie_emotions_rt_enriched.csv:
    - id (string slug)
    - title
    - rt_movie_id (reviews id)
    - genre
    - releaseDateTheaters
    - reviews (optional)
    - emotion cols...
    """
    if not os.path.exists(ENRICHED_CSV):
        raise FileNotFoundError(f"Enriched CSV bulunamadı: {ENRICHED_CSV}")

    df = pd.read_csv(ENRICHED_CSV)
    if "id" not in df.columns or "title" not in df.columns:
        raise ValueError(f"Enriched CSV kolonları beklenmedik: {list(df.columns)}")

    df["id"] = df["id"].astype(str)
    # id unique değilse ilkini al (genelde unique olmalı)
    df = df.drop_duplicates(subset=["id"], keep="first").set_index("id")
    return df


@st.cache_resource
def load_reviews_index():
    df = pd.read_csv(REVIEWS_CSV)

    if "id" not in df.columns or "reviewText" not in df.columns:
        raise ValueError(f"Reviews CSV kolonları beklenmedik: {list(df.columns)}")

    df = df.dropna(subset=["id", "reviewText"]).copy()
    df["id"] = df["id"].astype(str)           # ✅ string
    df["reviewText"] = df["reviewText"].astype(str)

    # newest-first if creationDate exists
    if "creationDate" in df.columns:
        df = df.sort_values(["id", "creationDate"], ascending=[True, False], kind="mergesort")
    else:
        df = df.sort_values(["id"], ascending=[True], kind="mergesort")

    idx = {}
    for mid, g in df.groupby("id", sort=False):
        idx[str(mid)] = [t.strip() for t in g["reviewText"].head(5).tolist() if t.strip()]
    return idx


def safe_get(s, default=""):
    if s is None:
        return default
    if isinstance(s, float) and pd.isna(s):
        return default
    return str(s)


# ---------------- UI ----------------
st.set_page_config(page_title="Hybrid + Reviews", page_icon="🎬", layout="centered")
st.title("🎬 Emotion Based Movie Recommender")

rec = load_rec()

# Load indexes
try:
    enriched_idx = load_enriched_index()
except Exception as e:
    enriched_idx = None
    st.warning(f"Enriched index yüklenemedi: {e}")

try:
    reviews_idx = load_reviews_index()
except Exception as e:
    reviews_idx = {}
    st.warning(f"Review index yüklenemedi: {e}")

# Inputs
text = st.text_area("Bugün Nasıl Hissediyorsun ? ", height=120)

c1, c2 = st.columns(2)
with c1:
    top_k = st.number_input("Kaç öneri?", 1, 30, 10, 1)
with c2:
    alpha = st.slider("Anlam ağırlığı α", 0.0, 1.0, 0.7, 0.05)

if st.button("Öner", type="primary"):
    if not text.strip():
        st.warning("Bir şey yaz.")
    else:
        emo_out = predict_emotions(text.strip(), threshold=0.0, translate_tr=True, drop_neutral=True)
        user_emotions = emo_out.get("emotions", {})

        top3 = sorted(user_emotions.items(), key=lambda x: -x[1])[:3]
        st.session_state["top3"] = top3
        st.session_state["last_query"] = text.strip()

        results = rec.recommend(
            text.strip(),
            user_emotions=user_emotions,
            top_k=int(top_k),
            alpha=float(alpha),
        )
        st.session_state["results"] = results

# Persisted outputs
top3 = st.session_state.get("top3", [])
if top3:
    st.markdown("### Baskın duygular (Top 3)")
    for i, (lab, val) in enumerate(top3, 1):
        st.write(f"{i}) **{lab}** — {val:.4f}")

results = st.session_state.get("results", [])
if results:
    st.markdown("### Öneriler")

    for i, r in enumerate(results, 1):
        # recommender id is string
        rec_id = safe_get(r.get("id", ""))  # e.g. 'love_lies' or '1122759-bug'
        title = safe_get(r.get("title", "Unknown"))

        # pull meta from enriched (genre/date + rt_movie_id)
        rt_movie_id = ""
        genre = ""
        release_date = ""
        if enriched_idx is not None and rec_id in enriched_idx.index:
            row = enriched_idx.loc[rec_id]
            rt_movie_id = safe_get(row.get("rt_movie_id", ""))
            genre = safe_get(row.get("genre", ""))
            release_date = safe_get(row.get("releaseDateTheaters", ""))

        st.subheader(f"{i}. {title} ")
        if release_date or genre:
            st.caption(f"{release_date}  |  {genre}")

        st.caption(
            f"final: {float(r['final_score']):.4f} | semantic: {float(r['semantic_score']):.4f} | emotion: {float(r['emotion_score']):.4f}"
        )

        # Reviews: prefer rt_movie_id if present, else fallback to rec_id
        lookup_id = rt_movie_id if rt_movie_id else rec_id
        revs = reviews_idx.get(lookup_id, [])

        if revs:
            with st.expander("Yorumlar"):
                for j, txt in enumerate(revs, 1):
                    st.write(f"**{j})** {txt[:1200]}{'…' if len(txt) > 1200 else ''}")
        else:
            # helpful debugging hint
            if rt_movie_id:
                st.caption(f"Bu film için review yok.")
            else:
                st.caption("Bu film için review yok.")
