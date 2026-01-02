# nlp_train/lexicon_emotion.py
# Rule/Lexicon-based emotion scoring (NO emojis). TR+EN, phrase patterns, intensifiers, negation scope,
# simple stemming/normalization. Explainable + fully hand-crafted.

import re
from collections import defaultdict

# ---------------------------
# Label set (match your CSV columns)
# ---------------------------
ALL_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
    "gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise"
]

# ---------------------------
# Normalization helpers
# ---------------------------
REPEAT_RE = re.compile(r"(.)\1{2,}", re.UNICODE)
TOKEN_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ']+", re.UNICODE)

TR_CHAR_MAP = str.maketrans({
    "ç":"c","ğ":"g","ı":"i","İ":"i","ö":"o","ş":"s","ü":"u",
    "Ç":"c","Ğ":"g","Ö":"o","Ş":"s","Ü":"u"
})

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = REPEAT_RE.sub(r"\1\1", t)         # "coooook" -> "coook"
    t = t.translate(TR_CHAR_MAP)          # "ş" -> "s" (phrase matching easier)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text: str):
    t = normalize_text(text)
    return TOKEN_RE.findall(t)

# simple stemming-ish (TR + EN) to catch common variants
TR_SUFFIXES = [
    "lar","ler",
    "im","um","iyim","yim","yum","sun","sin","siz","yiz","yuz","siniz","sunuz",
    "m","n","k","z",
    "de","da","den","dan","e","a","i","u","o","yi","yu","ya","ye",
    "dir","dır","dur","dür","tir","tır","tur","tür",
    "lik","lık","luk","luk","siz","suz","suz","suz",
    "ce","ca","cik","cuk","cok","cok",   # (note: "cok" is word too; kept for robustness)
    "mis","mus","müs","muş",
    "yor","yorum","yorsun","yoruz","yorlar",
]
EN_SUFFIXES = ["ing","ed","ly","ness","ment","ful","less","s","es"]

def stem_token(tok: str) -> str:
    if not tok:
        return tok
    t = tok
    # English stemming light
    for suf in EN_SUFFIXES:
        if len(t) > 5 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    # Turkish suffix stripping light (best-effort; not a real morphological analyzer)
    # Only strip if token stays reasonably long.
    for suf in TR_SUFFIXES:
        if len(t) > 5 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    return t

def normalize_token(tok: str) -> str:
    # already lower + tr chars mapped in tokenize()
    return stem_token(tok)

# ---------------------------
# Intensifiers / Diminishers / Negations
# ---------------------------
INTENSIFIERS = {
    # TR
    "cok","asiri","fazla","acayip","inanilmaz","muhtesem","muthis","efsane",
    "gercekten","cidden","hakikaten","tam","full","ultra","baya","bayagi","asla",  # asla as intensifier for emphasis
    # EN
    "very","really","so","extremely","super","insanely","highly","totally","absolutely",
    "literally","incredibly","hugely","massively",
}
DIMINISHERS = {
    # TR
    "biraz","az","hafif","pek","kismen","nispeten","ucundan","azicik",
    # EN
    "slightly","somewhat","kinda","kindof","sorta","a_bit","bit",
}
NEGATIONS = {
    # TR
    "degil","yok","asla","hic","hayir",
    # EN
    "not","no","never","without","neither","nor",
}

# Negation scope: number of following content tokens affected
NEG_SCOPE = 5

# Punctuation intensity
def punctuation_boost(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 1.0
    ex = text.count("!")
    qq = text.count("?")
    dots = text.count("...")
    # small bounded boost
    return 1.0 + min(0.6, 0.07 * ex + 0.04 * qq + 0.05 * dots)

# ---------------------------
# Phrase patterns (bigrams/trigrams/multi-word)
# We match on normalized_text (tr chars mapped, lower, collapsed spaces).
# Each phrase adds a weighted score to a label.
# ---------------------------
PHRASES = {
    # sadness / grief
    "icim sikildi": ("sadness", 1.6),
    "icim daraldi": ("sadness", 1.6),
    "modum dusuk": ("sadness", 1.4),
    "canim cok sikkin": ("sadness", 1.7),
    "yalniz hissediyorum": ("sadness", 1.8),
    "bogazim dugumlendi": ("grief", 1.7),
    "yikildim": ("grief", 1.4),
    "kahroldum": ("grief", 1.6),
    "basin sag olsun": ("grief", 1.8),
    "taziye": ("grief", 1.4),

    # nervousness / fear
    "kaygi yapiyorum": ("nervousness", 1.7),
    "panik atak": ("nervousness", 1.8),
    "strese girdim": ("nervousness", 1.6),
    "cok gerildim": ("nervousness", 1.6),
    "icim icimi yiyor": ("nervousness", 1.7),
    "korku filmi": ("fear", 1.1),
    "odum koptu": ("fear", 1.9),
    "tir tir titriyorum": ("fear", 1.9),
    "yerimden zipladim": ("fear", 1.4),

    # anger / annoyance
    "sinir oldum": ("anger", 1.6),
    "cok sinirlendim": ("anger", 1.8),
    "kafayi yedim": ("anger", 1.6),
    "delirdim": ("anger", 1.5),
    "tahammulum kalmadi": ("annoyance", 1.7),
    "canim sikildi": ("annoyance", 1.3),
    "gicik oldum": ("annoyance", 1.6),

    # amusement
    "kahkaha attim": ("amusement", 1.8),
    "gule gule": ("amusement", 1.2),
    "cok komik": ("amusement", 1.6),
    "guldum gec": ("amusement", 1.4),
    "lol": ("amusement", 1.3),

    # love
    "asik oldum": ("love", 1.9),
    "kalbim eridi": ("love", 1.7),
    "cok sevdim": ("love", 1.6),
    "romantik komedi": ("love", 1.2),

    # joy / excitement
    "icim acildi": ("joy", 1.7),
    "cok mutlu oldum": ("joy", 1.9),
    "havaya uctum": ("joy", 1.7),
    "asiri heyecanliyim": ("excitement", 1.9),
    "dort gozle bekliyorum": ("excitement", 1.6),
    "cant wait": ("excitement", 1.6),

    # relief
    "oh be": ("relief", 1.8),
    "rahat bir nefes": ("relief", 1.8),
    "oh finally": ("relief", 1.6),
    "finally": ("relief", 1.2),
    "icim rahat": ("relief", 1.6),
    "rahatladim": ("relief", 1.7),

    # gratitude / approval
    "tesekkur ederim": ("gratitude", 1.8),
    "cok tesekkur": ("gratitude", 1.9),
    "eline saglik": ("gratitude", 1.6),
    "sag ol": ("gratitude", 1.4),
    "helal olsun": ("approval", 1.6),
    "aferin": ("approval", 1.4),

    # pride
    "kendimle gurur duyuyorum": ("pride", 2.0),
    "gurur duydum": ("pride", 1.7),
    "basardim": ("pride", 1.7),
    "hak ettim": ("pride", 1.5),

    # realization / surprise
    "jeton dustu": ("realization", 1.9),
    "fark ettim": ("realization", 1.6),
    "meger": ("realization", 1.3),
    "inanamiyorum": ("surprise", 1.7),
    "sok oldum": ("surprise", 1.8),
    "yok artik": ("surprise", 1.6),

    # disappointment / disapproval / remorse
    "hayal kirikligi": ("disappointment", 1.8),
    "bekledigim gibi degil": ("disappointment", 1.6),
    "yazik oldu": ("disappointment", 1.5),
    "olmaz boyle": ("disapproval", 1.6),
    "kabul edilemez": ("disapproval", 1.7),
    "pismanim": ("remorse", 1.8),
    "ozur dilerim": ("remorse", 1.7),
    "benim hatam": ("remorse", 1.7),

    # curiosity / desire
    "acaba ne": ("curiosity", 1.2),
    "merak ediyorum": ("curiosity", 1.7),
    "bakmak istiyorum": ("desire", 1.4),
    "izlemek istiyorum": ("desire", 1.5),

    # confusion
    "ne oluyor": ("confusion", 1.6),
    "anlamadim": ("confusion", 1.6),
    "kafam karisik": ("confusion", 1.8),
}

# Precompile phrase regex (word-boundary-ish)
# We match phrases in normalized text directly; keep it simple and fast.
PHRASE_PATTERNS = [(p, re.compile(r"(^|\s)" + re.escape(p) + r"(\s|$)")) for p in PHRASES.keys()]

# ---------------------------
# Lexicon (TR + EN) - detailed per label
# NOTE: tokens are normalized (lower + tr chars -> ascii) then lightly stemmed.
# So keep lexicon mostly in normalized/ascii form.
# ---------------------------
LEXICON = {
    "admiration": {
        "admire","admiration","respect","respected","inspiring","inspiration","impressive","brilliant","genius",
        "legendary","iconic","masterpiece","phenomenal","excellent","greatness","marvelous","remarkable",
        "hayran","hayranlik","saygi","takdir","takdir_et","ilham","ilham_verici","etkileyici","mukemmel",
        "efsane","ustalik","sahane","harika","muthis","bayildim","cok_iyi",
    },
    "amusement": {
        "funny","hilarious","laugh","laughed","laughing","lol","lmao","rofl","comedy","joke","humor","witty","goofy","silly",
        "komik","kahkaha","gul","guldum","gulmek","mizah","espri","saka","eglenceli","matrak","caps","tiye",
    },
    "anger": {
        "angry","mad","furious","rage","outraged","pissed","hate","hated","hostile","enraged","fuming",
        "sinir","sinirli","ofke","ofkeli","kizgin","kizdim","nefret","kin","cildir","delir","delirdim","patladim","cileden",
    },
    "annoyance": {
        "annoy","annoyed","annoying","irritate","irritated","bother","bothered","frustrate","frustrated","tiring","exhausting","tedious",
        "rahatsiz","rahatsizim","gicik","biktim","bikkin","sikici","can_sikici","sinir_bo", "tahammul","bunaltici",
    },
    "approval": {
        "approve","approved","good","nice","well","well_done","solid","decent","fine","okay","ok","cool","valid",
        "onay","olur","iyi","guzel","yerinde","mantikli","dogru","basarili","guclu","tamam","super",
    },
    "caring": {
        "care","caring","concern","concerned","support","supportive","protect","protective","comfort","gentle","kind","kindness",
        "onemse","ilgilen","ilgi","sefkat","merhamet","destek","yardim","koru","sarip_sarmala","duyarlilik","incelik",
    },
    "confusion": {
        "confuse","confused","confusing","unclear","lost","puzzled","perplexed","huh","what","wtf","baffled",
        "kafam_karisik","karisik","anlamadim","belirsiz","ne_oluyor","sasirdim","anlamsiz","karmasik",
    },
    "curiosity": {
        "curious","curiosity","wonder","wondering","interested","intrigued","why","how","explore","discover","investigate","mystery",
        "merak","merakli","acaba","ilgimi","cek","kesfet","arastir","sorgula","gizem","merak_ediyorum",
    },
    "desire": {
        "want","desire","crave","need","wish","longing","yearn","hope_for","looking_for","seek",
        "istiyorum","isterim","arzu","arzuluyorum","ozlem","canim_cek","heves","hevesli","bakmak","izlemek","gormek",
    },
    "disappointment": {
        "disappointed","letdown","let_down","underwhelmed","meh","expected","expectation","worse","sadly","regretful",
        "hayal_kirikligi","umdugum","bekledigim","olmadi","tuh","yazik","bos_cikti","hevesim_kacti","kotu_cikti",
    },
    "disapproval": {
        "disapprove","wrong","bad","not_ok","unacceptable","shame","disgusting","terrible","awful",
        "kabul_edilemez","yanlis","uygunsuz","ayip","olmaz","tasvip","begenmedim","rezalet","kepaze",
    },
    "disgust": {
        "disgust","disgusted","gross","nasty","sickening","repulsive","ew","yuck","vile","filthy",
        "igrenc","tiksin","tiksinti","mide_bulandir","berbat","pis","kirli","rezil","iicim_kalkti","midem_bulandi",
    },
    "embarrassment": {
        "embarrassed","awkward","cringe","humiliated","shy","selfconscious","mortified",
        "utandim","utaniyorum","mahcup","rezil_oldum","yuzum_kizardi","kringe","siritkan","sakil",
    },
    "excitement": {
        "excited","exciting","thrilled","hyped","hype","pumped","stoked","cant_wait","eager","energized",
        "heyecan","heyecanli","cosku","sabirsiz","dort_gozle","hype","gaza_geldim","acayip_heyecan",
    },
    "fear": {
        "afraid","scared","terrified","horror","panic","frightened","danger","dread","nightmare",
        "korku","korktum","korkuyorum","dehset","panik","tehlike","odum_koptu","urk","kabus",
    },
    "gratitude": {
        "thank","thanks","grateful","appreciate","appreciation","blessed","thankful",
        "tesekkur","minnet","minnettar","sag_ol","eyvallah","cok_sagol","var_ol","eline_saglik",
    },
    "grief": {
        "grief","mourning","devastated","heartbroken","loss","funeral","bereaved","tragic","shattered",
        "yas","matem","yikildim","kahroldum","aci","kayip","olum","taziye","basin_sag_olsun","yureğim_yandi",
    },
    "joy": {
        "joy","happy","happiness","delighted","cheerful","glad","smile","wonderful","pleased","content",
        "mutlu","neseli","sevinc","keyif","keyifli","icim_acildi","gulumse","hos","cok_mutlu","sevin",
    },
    "love": {
        "love","romance","romantic","affection","adore","crush","sweetheart","beloved","passion",
        "ask","romantik","sevgi","sevmek","bayiliyorum","kalp","hoslaniyorum","asik","tutku","canim",
    },
    "nervousness": {
        "nervous","anxious","anxiety","stressed","stress","tense","uneasy","restless","worried","worry",
        "gergin","stres","kaygi","endise","huzursuz","telas","panik","strese","kaygiliyim","endiseliyim",
    },
    "optimism": {
        "hope","hopeful","optimistic","positive","better","bright","confident","itll_be_ok","encouraged",
        "umut","umutlu","iyimser","pozitif","daha_iyi","gececek","olacak","rahat_ol","hallederiz","cozeriz",
    },
    "pride": {
        "proud","pride","achievement","accomplished","earned","deserve","success","victory",
        "gurur","gururlu","basardim","basari","hak_ettim","kendimle_gurur","ovun","zafer","basardik",
    },
    "realization": {
        "realize","realized","suddenly","it_hit_me","understood","now_i_see","aha","epiphany",
        "fark_ettim","anladim","aydinlandim","jeton_dustu","meger","simdi_anladim","idrak",
    },
    "relief": {
        "relief","relieved","finally","phew","safe","calm","at_last","unburdened",
        "rahatladim","oh_be","sukur","nihayet","icim_rahat","kurtuldum","rahat_bir_nefes",
    },
    "remorse": {
        "remorse","regret","sorry","guilty","my_fault","apologize","apology","ashamed",
        "pisman","pismanim","ozur","ozur_dilerim","sucluyum","vicdan","vicdan_azabi","hatam",
    },
    "sadness": {
        "sad","down","depressed","blue","lonely","empty","cry","tears","miserable","hopeless","melancholy",
        "uzgun","mutsuz","yalniz","icim_sikildi","huzun","keder","aglamak","agladim","bosluk","bitkin",
    },
    "surprise": {
        "surprise","surprised","shocked","wow","unexpected","no_way","omg","unbelievable","stunned",
        "sasirdim","sok","vay","inanamiyorum","beklemezdim","nasil_ya","yok_artik","sasirtici",
    },
}

# Normalize lexicon entries into token forms (ascii+stem friendly).
# We'll treat underscores in lexicon as phrase-like single tokens only if present in text token stream,
# but primarily phrases handled by PHRASES. Still okay.
def _normalize_lexicon():
    out = {}
    for emo, words in LEXICON.items():
        s = set()
        for w in words:
            ww = normalize_text(w.replace("_", " "))
            # store token-level entries; if it's multiword, keep first token only won't help -> skip
            toks = TOKEN_RE.findall(ww)
            if len(toks) == 1:
                s.add(normalize_token(toks[0]))
            elif len(toks) > 1:
                # also keep each token to help partial matches a bit
                for t in toks:
                    s.add(normalize_token(t))
        out[emo] = s
    return out

LEX = _normalize_lexicon()

# ---------------------------
# Core scoring
# ---------------------------
def score_emotions(text: str) -> dict:
    """
    Returns normalized scores for ALL_LABELS. Sum=1 if any hits, else all zeros.
    Uses:
      - phrase hits (weighted)
      - token hits (lexicon)
      - intensifiers/diminishers (local)
      - negation scope (local)
      - punctuation boost (global)
    """
    norm = normalize_text(text)
    toks_raw = tokenize(text)
    toks = [normalize_token(t) for t in toks_raw if t]

    scores = defaultdict(float)

    # 1) phrase matches
    # We apply phrase matches on normalized text to catch multiword expressions reliably.
    for phrase, pat in PHRASE_PATTERNS:
        if pat.search(norm):
            emo, w = PHRASES[phrase]
            scores[emo] += float(w)

    # 2) token matches with rules
    p_boost = punctuation_boost(text)

    neg_left = 0
    weight_pending = 1.0

    for tok in toks:
        if tok in INTENSIFIERS:
            weight_pending *= 1.5
            continue
        if tok in DIMINISHERS:
            weight_pending *= 0.7
            continue
        if tok in NEGATIONS:
            neg_left = NEG_SCOPE
            continue

        # lexicon hits
        hit_any = False
        for emo in ALL_LABELS:
            if tok in LEX.get(emo, set()):
                w = 1.0 * weight_pending * p_boost
                if neg_left > 0:
                    # negation reduces emotional assertion; keep small residual rather than flipping labels
                    w *= 0.25
                scores[emo] += w
                hit_any = True

        # decay negation
        if neg_left > 0:
            neg_left -= 1

        # reset weight after a "content token" (hit or not)
        weight_pending = 1.0

    # 3) build full vector and normalize
    out = {k: float(scores.get(k, 0.0)) for k in ALL_LABELS}
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out

def top_k_emotions(text: str, k: int = 3):
    sc = score_emotions(text)
    ranked = sorted(sc.items(), key=lambda x: -x[1])
    return ranked[:k]
