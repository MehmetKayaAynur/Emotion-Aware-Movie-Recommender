from __future__ import annotations
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
NEUTRAL_LABEL = "neutral"

# Model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Label list (modelin label sırası!)
LABELS: List[str] = [model.config.id2label[i] for i in range(model.config.num_labels)]


def _drop_neutral(emotions: Dict[str, float]) -> Dict[str, float]:
    """Remove neutral from distribution (for better recommendation discrimination)."""
    if NEUTRAL_LABEL in emotions:
        emotions = dict(emotions)  # copy
        emotions.pop(NEUTRAL_LABEL, None)
    return emotions


def _pick_active(emotions_no_neutral: Dict[str, float], threshold: float, top_k: int) -> List[str]:
    """
    Active emotion selection (neutral removed):
    1) threshold üstündekileri seç
    2) hiç yoksa: top_k seç
    3) hala yoksa: boş liste (neutral yok!)
    """
    active = [k for k, v in emotions_no_neutral.items() if v >= threshold]

    if not active:
        active = [k for k, _ in sorted(emotions_no_neutral.items(), key=lambda x: -x[1])[:top_k]]

    return active


@torch.inference_mode()
def predict_emotions(
    text: str,
    threshold: float = 0.25,
    top_k: int = 5,
    translate_tr: bool = True,
    drop_neutral: bool = True
) -> Dict:
    """
    Input: user text
    Output: emotion probabilities + active labels (neutral removed by default)

    translate_tr=True ise:
      - Türkçe metni İngilizceye çevirir (GoEmotions İngilizce eğitildiği için)
      - Modeli çeviri metinle çalıştırır

    drop_neutral=True ise:
      - 'neutral' etiketi distribution'dan çıkarılır (M2 ile uyum + daha iyi ayrım)
    """
    original_text = text
    used_text = text

    if translate_tr:
        try:
            used_text = GoogleTranslator(source="tr", target="en").translate(original_text)
        except Exception:
            used_text = original_text  # çeviri fail olursa fallback

    inputs = tokenizer(
        used_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    logits = model(**inputs).logits[0]
    probs = torch.sigmoid(logits).cpu().tolist()

    raw_emotions = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    emotions = _drop_neutral(raw_emotions) if drop_neutral else raw_emotions
    active = _pick_active(emotions if drop_neutral else _drop_neutral(raw_emotions), threshold, top_k)

    return {
        "text": original_text,      # kullanıcı girdisi
        "used_text": used_text,     # modele verilen (çeviri olabilir)
        "emotions": emotions,       #  neutral yok (default)
        "active": active,
        "threshold": threshold,
        "model": MODEL_NAME,
        "translate_tr": translate_tr,
        "drop_neutral": drop_neutral
    }


if __name__ == "__main__":
    sample = "Bugün çok bunaldım ama yine de umutluyum."
    out = predict_emotions(sample, threshold=0.25, top_k=5, translate_tr=True, drop_neutral=True)

    top10 = sorted(out["emotions"].items(), key=lambda x: -x[1])[:10]
    print("TOP10 (neutral removed):", top10)
    print("ACTIVE:", out["active"])
    print("MODEL:", out["model"])
