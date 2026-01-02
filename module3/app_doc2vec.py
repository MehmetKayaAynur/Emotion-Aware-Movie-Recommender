import os
import sys
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from module1_emotion import predict_emotions  # hazır model (user text)
from nlp_train.recommend_hybrid import HybridRecommender  # hibrit

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Hybrid Movie Recommender (Doc2Vec + Emotion Match)")

@st.cache_resource
def load_rec():
    return HybridRecommender()

rec = load_rec()

text = st.text_area("Ne izlemek istiyorsun? (duygu / vibe / tür)", value="", height=120)
k = st.number_input("Kaç öneri?", min_value=1, max_value=30, value=10, step=1)
alpha = st.slider("Anlam (Doc2Vec) ağırlığı α", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

if st.button("Öner", type="primary"):
    if not text.strip():
        st.warning("Bir şey yaz.")
    else:
        # 1) Kullanıcı metninden duygu dağılımı (hazır model)
        emo_out = predict_emotions(text.strip(), threshold=0.0, translate_tr=True, drop_neutral=True)
        user_emotions = emo_out.get("emotions", {})

        top3 = sorted(user_emotions.items(), key=lambda x: -x[1])[:3]
        st.markdown("### Yazdığın metindeki baskın duygular (Top 3) — hazır model")
        for i, (lab, val) in enumerate(top3, 1):
            st.write(f"{i}) **{lab}** — {val:.4f}")

        # 2) Hibrit öneri
        results = rec.recommend(text.strip(), user_emotions=user_emotions, top_k=int(k), alpha=float(alpha))

        st.markdown("### Öneriler (hibrit skor)")
        if not results:
            st.info("Metin çok kısa/boş. Biraz daha detay yaz.")
        else:
            for i, r in enumerate(results, 1):
                st.write(
                    f"**{i}. {r['title']}**  "
                    f"— final: {r['final_score']:.4f} | semantic: {r['semantic_score']:.4f} | emotion: {r['emotion_score']:.4f}"
                )
