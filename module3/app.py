import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")


# root path (module1_emotion import için)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from module1_emotion import predict_emotions, LABELS

CSV_PATH = os.path.join(ROOT, "module2", "final_movie_emotions.csv")

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

def cosine_topk(user_vec: np.ndarray, M: np.ndarray, k: int = 10) -> np.ndarray:
    u = user_vec.astype(np.float32)
    u = u / (np.linalg.norm(u) + 1e-12)

    Mn = M.astype(np.float32)
    Mn = Mn / (np.linalg.norm(Mn, axis=1, keepdims=True) + 1e-12)

    sims = Mn @ u
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx

def get_poster_url(movie_id: int):
    """
    Cache YOK: API key sonradan değişince posterler gelsin diye.
    """
    if not TMDB_API_KEY:
        return None

    url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return None
        return TMDB_IMG_BASE + poster_path
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_movies():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"final_movie_emotions.csv bulunamadı: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    labels_no_neutral = [l for l in LABELS if l != "neutral"]
    missing = [l for l in labels_no_neutral if l not in df.columns]
    if missing:
        raise ValueError(f"CSV'de eksik duygu sütunları var: {missing}")

    # title col
    title_col = "title" if "title" in df.columns else ("original_title" if "original_title" in df.columns else None)
    if title_col is None:
        candidates = [c for c in df.columns if df[c].dtype == object]
        title_col = candidates[0] if candidates else None

    # id col (TMDb id)
    id_col = "id" if "id" in df.columns else ("movie_id" if "movie_id" in df.columns else None)
    if id_col is None:
        raise ValueError("CSV'de film id kolonu yok (id veya movie_id olmalı).")

    M = df[labels_no_neutral].to_numpy(dtype=np.float32)
    return df, M, labels_no_neutral, title_col, id_col


# ---------- UI ----------
st.set_page_config(page_title="Emotion-Aware Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Emotion-Aware Movie Recommender")

# input alanı BOŞ gelsin
text = st.text_area(
    "Kendini Nasıl hissediyorsun? Nasıl bir film izlemek istersin:)",
    value="",
    height=110,
    placeholder="Örn: Bugün çok stresliyim, her şey üst üste geldi."
)

top_k = st.number_input(
    "Kaç film önerilsin?",
    min_value=1,
    max_value=30,
    value=3,
    step=1
)

# UI'da GİZLİ ama sabit
threshold = 0.25
translate_tr = True

# TMDb key uyarısı (kibar)
if not TMDB_API_KEY:
    st.warning("TMDB_API_KEY bulunamadı. Posterler görünmeyecek. Terminalde export ettiğine emin ol.")

if st.button("Öneri getir", type="primary"):
    if not text.strip():
        st.info("Lütfen bir metin yaz ")
    else:
        df, M, labels_no_neutral, title_col, id_col = load_movies()

        with st.spinner("Duygular analiz ediliyor..."):
            out = predict_emotions(text, threshold=threshold, translate_tr=translate_tr)

        # sadece TOP 2 emotion göster
        top2 = sorted(out["emotions"].items(), key=lambda x: -x[1])[:2]

       # st.success("Tamam! Öneriler hazır ")

        st.markdown("### En yüksek 2 duygu")
        st.write(f"**1)** {top2[0][0]} : {top2[0][1]:.6f}")
        st.write(f"**2)** {top2[1][0]} : {top2[1][1]:.6f}")

        # kullanıcı vektörü (neutral hariç)
        user_vec = np.array([out["emotions"][l] for l in labels_no_neutral], dtype=np.float32)

        with st.spinner("Filmlerle eşleştiriliyor..."):
            idxs = cosine_topk(user_vec, M, k=int(top_k))

        st.markdown("### 🎬 Film önerileri")
        for rank, i in enumerate(idxs, start=1):
            row = df.iloc[int(i)]
            title = str(row[title_col]) if title_col else "UnknownTitle"
            movie_id = int(row[id_col])

            # similarity
            mv = M[int(i)]
            sim = float((mv / (np.linalg.norm(mv) + 1e-12)) @ (user_vec / (np.linalg.norm(user_vec) + 1e-12)))

            poster_url = get_poster_url(movie_id)

            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.write("🖼️ Poster yok")
                with cols[1]:
                    st.markdown(f"**{rank}. {title}**")
                    st.caption(f"Similarity: {sim:.4f}")
                    if "overview" in df.columns and isinstance(row.get("overview", None), str):
                        ov = row["overview"]
                        st.write(ov[:220] + ("..." if len(ov) > 220 else ""))

        st.caption("Posterler TMDb üzerinden alınır . (Attribution: TMDb)")
