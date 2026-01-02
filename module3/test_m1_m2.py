import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module1_emotion import predict_emotions, LABELS

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "module2",
    "final_movie_emotions.csv"
)

def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def main():
    df = pd.read_csv(CSV_PATH)
    labels_no_neutral = [l for l in LABELS if l != "neutral"]

    print("\n=== Module 1 + Module 2 Test ===")
    print("Çıkmak için Enter\n")

    while True:
        text = input("Metni gir: ").strip()
        if text == "":
            print("Çıkılıyor...")
            break

        out = predict_emotions(text, translate_tr=True)
        user_vec = np.array([out["emotions"][l] for l in labels_no_neutral])

        sims = []
        for _, row in df.iterrows():
            movie_vec = row[labels_no_neutral].to_numpy()
            sims.append(cosine_sim(user_vec, movie_vec))

        df["similarity"] = sims
        top = df.sort_values("similarity", ascending=False).head(5)

        print("\nACTIVE EMOTIONS:", out["active"])
        print("\nTOP 5 FILM ÖNERİSİ:")
        for i, r in enumerate(top.itertuples(), 1):
            print(f"{i}. {r.title}  | similarity={r.similarity:.4f}")

        print("-" * 50)

if __name__ == "__main__":
    main()
