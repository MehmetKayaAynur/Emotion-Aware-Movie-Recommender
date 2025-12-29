from module1_emotion import predict_emotions

print("=== MODÜL 1 | DUYGU / RUH HALİ ANALİZİ (neutral removed) ===")
print("Çıkmak için boş Enter'a bas.\n")

while True:
    text = input("Metni gir: ").strip()

    if text == "":
        print("\nÇıkılıyor...")
        break

    out = predict_emotions(text, translate_tr=True, drop_neutral=True)

    print("\n" + "-" * 50)
    print("TEXT:")
    print(out["text"])

    print("\nUSED TEXT (model input):")
    print(out["used_text"])

    print("\nACTIVE EMOTIONS:")
    for e in out["active"]:
        print(f"- {e}")

    print("\nTOP 10 EMOTIONS (neutral removed):")
    top10 = sorted(out["emotions"].items(), key=lambda x: -x[1])[:10]
    for k, v in top10:
        print(f"{k:15s} : {v:.6f}")

    print(f"\nMODEL: {out['model']}")
    print(f"THRESHOLD: {out['threshold']}")
    print("-" * 50 + "\n")
