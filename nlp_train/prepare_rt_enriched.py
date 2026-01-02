import os
import re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMO_PATH = os.path.join(ROOT, "module2", "final_movie_emotions_rt.csv")
MOV_PATH = os.path.join(ROOT, "module2", "rotten_tomatoes_movies.csv")
REV_PATH = os.path.join(ROOT, "module2", "rotten_tomatoes_movie_reviews.csv")

OUT_PATH = os.path.join(ROOT, "module2", "final_movie_emotions_rt_enriched.csv")

EMO_COLS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
    "gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise"
]

def norm_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.lower().strip()
    t = re.sub(r"&", " and ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def year_from_emotion_id(eid: str):
    # adrift_2018 -> 2018
    if not isinstance(eid, str):
        return None
    m = re.search(r"(19\d{2}|20\d{2})$", eid)
    return int(m.group(1)) if m else None

def year_from_date(s: str):
    if not isinstance(s, str) or len(s) < 4:
        return None
    m = re.match(r"(\d{4})", s.strip())
    return int(m.group(1)) if m else None

def main():
    emo = pd.read_csv(EMO_PATH)
    mov = pd.read_csv(MOV_PATH)
    rev = pd.read_csv(REV_PATH)

    # normalize ids/titles
    emo["id"] = emo["id"].astype(str)
    emo["title"] = emo["title"].astype(str)
    emo["title_norm"] = emo["title"].map(norm_title)
    emo["year_hint"] = emo["id"].map(year_from_emotion_id)

    mov["id"] = mov["id"].astype(str)  # RT movie id (for reviews)
    mov["title"] = mov["title"].astype(str)
    mov["title_norm"] = mov["title"].map(norm_title)
    mov["year_theaters"] = mov.get("releaseDateTheaters", "").astype(str).map(year_from_date)
    mov["year_stream"] = mov.get("releaseDateStreaming", "").astype(str).map(year_from_date)

    # reviews: group by RT movie id
    rev = rev.dropna(subset=["id", "reviewText"]).copy()
    rev["id"] = rev["id"].astype(str)
    rev["reviewText"] = rev["reviewText"].astype(str)

    if "creationDate" in rev.columns:
        rev = rev.sort_values(["id", "creationDate"], ascending=[True, False], kind="mergesort")

    rev5 = rev.groupby("id", sort=False)["reviewText"].head(5).groupby(rev["id"]).apply(list)
    reviews_by_movie_id = {k: v for k, v in rev5.items()}

    # --- Title-based matching: emo.title_norm -> mov.title_norm
    # If multiple matches, try year_hint vs theaters/streaming year; else pick first.
    mov_groups = mov.groupby("title_norm", sort=False)

    matched_movie_id = []
    matched_genre = []
    matched_release = []

    for _, row in emo.iterrows():
        tnorm = row["title_norm"]
        yh = row["year_hint"]

        if tnorm not in mov_groups.groups:
            matched_movie_id.append("")
            matched_genre.append("")
            matched_release.append("")
            continue

        cand = mov.loc[mov_groups.groups[tnorm]].copy()

        # if year hint exists, prefer closest match
        if yh is not None:
            cand["cand_year"] = cand["year_theaters"].fillna(cand["year_stream"])
            # exact year first
            exact = cand[cand["cand_year"] == yh]
            if len(exact) > 0:
                best = exact.iloc[0]
            else:
                # otherwise pick first candidate (could improve later)
                best = cand.iloc[0]
        else:
            best = cand.iloc[0]

        matched_movie_id.append(str(best["id"]))
        matched_genre.append(str(best.get("genre", "")) if pd.notna(best.get("genre", "")) else "")
        matched_release.append(str(best.get("releaseDateTheaters", "")) if pd.notna(best.get("releaseDateTheaters", "")) else "")

    emo["rt_movie_id"] = matched_movie_id
    emo["genre"] = matched_genre
    emo["releaseDateTheaters"] = matched_release

    # attach reviews list + joined text
    def get_reviews(mid: str):
        if not mid:
            return []
        return reviews_by_movie_id.get(mid, [])

    emo["reviews_list"] = emo["rt_movie_id"].map(get_reviews)
    emo["reviews"] = emo["reviews_list"].map(lambda xs: " ||| ".join(xs[:5]) if isinstance(xs, list) else "")

    # Keep columns
    keep_cols = ["id", "title", "rt_movie_id", "genre", "releaseDateTheaters", "reviews"] + EMO_COLS
    emo_out = emo[keep_cols].copy()

    emo_out.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)
    print("Matched RT movie_id ratio:", (emo_out["rt_movie_id"] != "").mean())

if __name__ == "__main__":
    main()
