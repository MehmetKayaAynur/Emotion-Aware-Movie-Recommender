# module3/dual_chat_app.py
import os
import sys
import re
import numpy as np
import pandas as pd
import streamlit as st
import requests

from dotenv import load_dotenv
load_dotenv()

# --- OpenAI ---
from openai import OpenAI

# ---- root path (module1_emotion import için) ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from module1_emotion import predict_emotions, LABELS  # noqa: E402


# --- Config ---
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
CSV_PATH = os.path.join(ROOT, "module2", "final_movie_emotions.csv")

client = OpenAI() if OPENAI_API_KEY else None


# ---------------- Utils ----------------
def cosine_topk(user_vec: np.ndarray, M: np.ndarray, k: int = 10) -> np.ndarray:
    u = user_vec.astype(np.float32)
    u = u / (np.linalg.norm(u) + 1e-12)

    Mn = M.astype(np.float32)
    Mn = Mn / (np.linalg.norm(Mn, axis=1, keepdims=True) + 1e-12)

    sims = Mn @ u
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx


@st.cache_data(show_spinner=False)
def load_movies():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV yok: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    labels_no_neutral = [l for l in LABELS if l != "neutral"]
    missing = [l for l in labels_no_neutral if l not in df.columns]
    if missing:
        raise ValueError(f"CSV'de eksik duygu sütunları var: {missing}")

    title_col = "title" if "title" in df.columns else None
    id_col = "id" if "id" in df.columns else None
    if not title_col or not id_col:
        raise ValueError("CSV'de 'id' ve 'title' kolonları olmalı.")

    M = df[labels_no_neutral].to_numpy(dtype=np.float32)
    return df, M, labels_no_neutral, title_col, id_col


def recommend_movies(mood_text: str, top_k: int):
    df, M, labels_no_neutral, title_col, id_col = load_movies()

    out = predict_emotions(mood_text, threshold=0.25, translate_tr=True, drop_neutral=True)
    user_vec = np.array([out["emotions"][l] for l in labels_no_neutral], dtype=np.float32)

    idxs = cosine_topk(user_vec, M, k=int(top_k))
    recs = []
    for i in idxs:
        row = df.iloc[int(i)]
        recs.append({"title": str(row[title_col]), "id": int(row[id_col])})

    top2 = sorted(out["emotions"].items(), key=lambda x: -x[1])[:2]
    return out, top2, recs


@st.cache_data(show_spinner=False)
def tmdb_get_movie(movie_id: int):
    """Film detayını TMDb'den çek (overview/genres/release vs.). Cache var: hız + rate limit dostu."""
    if not TMDB_API_KEY:
        return None
    url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def tmdb_poster_url(movie_id: int):
    data = tmdb_get_movie(movie_id)
    if not data:
        return None
    p = data.get("poster_path")
    if not p:
        return None
    return TMDB_IMG_BASE + p


def resolve_movie_ref(user_text: str, recs: list[dict]):
    """
    '2. film', 'ikinci film', '2nd movie' vb. referansları çöz.
    Dönerse: (movie_dict, reason_str), dönmezse: (None, None)
    """
    if not recs:
        return None, None

    t = user_text.lower().strip()

    # Sayısal: "2. film", "2 film", "film 2", "movie 2"
    m = re.search(r"\b(\d{1,2})\s*\.?\s*(film|movie)\b", t)
    if not m:
        m = re.search(r"\b(film|movie)\s*(\d{1,2})\b", t)
        if m:
            n = int(m.group(2))
            if 1 <= n <= len(recs):
                return recs[n - 1], f"{n}. film"
    else:
        n = int(m.group(1))
        if 1 <= n <= len(recs):
            return recs[n - 1], f"{n}. film"

    # Türkçe ordinal
    order_tr = {
        "birinci": 1,
        "ikinci": 2,
        "üçüncü": 3,
        "ucuncu": 3,
        "dördüncü": 4,
        "dorduncu": 4,
        "beşinci": 5,
        "besinci": 5,
    }
    if ("film" in t) or ("movie" in t):
        for k, v in order_tr.items():
            if k in t and 1 <= v <= len(recs):
                return recs[v - 1], f"{v}. film"

    # English ordinal
    order_en = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    if ("film" in t) or ("movie" in t):
        for k, v in order_en.items():
            if k in t and 1 <= v <= len(recs):
                return recs[v - 1], f"{v}. film"

    return None, None


def info_llm_answer(user_question: str, recommended_movies: list[dict]):
    """
    LLM'e sadece önerilen filmler üzerinden bilgi verdir.
    Context'i TMDb'den çekiyoruz (overview/genres/release/runtime).
    Ayrıca numaralı liste veriyoruz: '2. film' -> #2.
    """
    ctx_lines = []
    for rank, r in enumerate(recommended_movies[:10], start=1):
        mid = int(r["id"])
        title = r["title"]
        data = tmdb_get_movie(mid) if TMDB_API_KEY else None

        if data:
            overview = (data.get("overview") or "").strip()
            release = data.get("release_date") or ""
            genres = ", ".join([g["name"] for g in data.get("genres", [])]) if data.get("genres") else ""
            runtime = data.get("runtime")
            ctx_lines.append(
                f"{rank}) {title} (id={mid}) | release={release} | genres={genres} | runtime={runtime}\n"
                f"overview: {overview}"
            )
        else:
            ctx_lines.append(f"{rank}) {title} (id={mid}) | (TMDb yok, sadece başlık)")

    context = "\n\n".join(ctx_lines) if ctx_lines else "No recommendations available yet."

    system = (
        "You are a movie assistant. Answer ONLY using the provided CONTEXT about the recommended movies. "
        "The list is numbered. If the user says '2nd movie / second movie / 2. film / ikinci film', "
        "they mean item #2 in the numbered list. "
        "If the answer is not in the context, say you don't have that info and ask a clarifying question. "
        "Do NOT introduce movies outside the context unless the user explicitly asks for new recommendations."
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{user_question}"},
        ],
    )
    return resp.output_text


# ---------------- UI ----------------
st.set_page_config(page_title="Dual Chat Movie App", page_icon="🎬", layout="centered")
st.title("🎬 Duygu ile Öneri + Film Bilgi Chatbot")

# State init (ÖNERİLER KAYBOLMASIN diye her şeyi session_state'te tutuyoruz)
if "recs" not in st.session_state:
    st.session_state.recs = []
if "last_mood" not in st.session_state:
    st.session_state.last_mood = ""
if "last_top2" not in st.session_state:
    st.session_state.last_top2 = None
if "info_chat" not in st.session_state:
    st.session_state.info_chat = [
        {
            "role": "assistant",
            "content": "Önce 'Duygu → Film Öner' sekmesinden öneri al. Sonra burada önerilen filmler hakkında soru sor.",
        }
    ]

tab1, tab2 = st.tabs(["1) Duygu → Film Öner", "2) Önerilen Filmler Hakkında Sor (AI)"])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("Duygu Chat’i (Öneri)")
    mood = st.text_area(
        "Nasıl hissediyorsun? Ne izlemek istersin?",
        value=st.session_state.last_mood,
        height=110,
        placeholder="Örn: Bugün stresliyim, kafa dağıtmalık bir şey istiyorum.",
    )
    top_k = st.number_input("Kaç öneri?", min_value=1, max_value=30, value=5, step=1)

    if not TMDB_API_KEY:
        st.warning("TMDB_API_KEY yoksa poster/film detayları gelmez. (Ama öneri yine çalışır.)")

    if st.button("Öneri getir", type="primary"):
        if not mood.strip():
            st.warning("Bir şey yaz.")
        else:
            out, top2, recs = recommend_movies(mood.strip(), int(top_k))
            st.session_state.recs = recs
            st.session_state.last_mood = mood.strip()
            st.session_state.last_top2 = top2

    # BUTONA BASMASAN DA son öneriler ekranda kalsın:
    if st.session_state.last_top2:
        top2 = st.session_state.last_top2
        st.markdown("### Top 2 duygu (son analiz)")
        st.write(f"1) {top2[0][0]} : {top2[0][1]:.4f}")
        st.write(f"2) {top2[1][0]} : {top2[1][1]:.4f}")

    if st.session_state.recs:
        st.markdown("### Öneriler (hafızadan)")
        for i, r in enumerate(st.session_state.recs, 1):
            poster = tmdb_poster_url(r["id"]) if TMDB_API_KEY else None
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    if poster:
                        st.image(poster, use_container_width=True)
                    else:
                        st.write("🖼️ Poster yok")
                with cols[1]:
                    st.markdown(f"**{i}. {r['title']}**")
                    st.caption(f"TMDb id: {r['id']}")
    else:
        st.info("Henüz öneri yok. Üstten duygu yazıp 'Öneri getir'e bas.")

# ---------- TAB 2 ----------
with tab2:
    st.subheader("Bilgi Chat’i (AI)")

    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY yok. Bilgi chat çalışmaz.")
    if not TMDB_API_KEY:
        st.warning("TMDB_API_KEY yoksa film detayları zayıf olur (overview/genre çekemeyiz).")

    if not st.session_state.recs:
        st.info("Önce 'Duygu → Film Öner' sekmesinden öneri al.")

    # chat render
    for m in st.session_state.info_chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    q = st.chat_input("Önerilen filmler hakkında soru sor… (örn: 2. film ne anlatıyor?)")
    if q:
        st.session_state.info_chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)

        # 1) Önce deterministic çöz: "2. film" vb.
        picked, picked_reason = resolve_movie_ref(q, st.session_state.recs)
        if picked:
            data = tmdb_get_movie(picked["id"]) if TMDB_API_KEY else None
            if data and (data.get("overview") or "").strip():
                overview = data["overview"].strip()
                title = picked["title"]
                answer = f"**{picked_reason}: {title}**\n\n{overview}"
            else:
                # TMDb yoksa LLM'ye düşebilir ama context zayıf olur
                answer = f"**{picked_reason}: {picked['title']}**\n\nBu film için overview bilgisine şu an erişemiyorum."

            st.session_state.info_chat.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
            st.stop()

        # 2) Film adıyla sorarsa: LLM'e (context = önerilen filmler)
        if st.session_state.recs and client:
            answer = info_llm_answer(q, st.session_state.recs)
        else:
            answer = "Önce öneri al ve OPENAI_API_KEY ayarlı olsun."

        st.session_state.info_chat.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
