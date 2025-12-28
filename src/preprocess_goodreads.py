import pandas as pd
from pathlib import Path
import ast
import re
import requests
from tqdm import tqdm
from urllib.parse import quote_plus

# ===============================
# PATH
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data/raw/goodreads_books.csv"
OUTPUT_PATH = BASE_DIR / "data/processed/items.csv"
IMAGE_DIR = BASE_DIR / "data/images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# REQUEST HEADERS (QUAN TRỌNG)
# ===============================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json"
}

# ===============================
# TEXT UTILS
# ===============================
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_genres(x):
    try:
        return " ".join(ast.literal_eval(x))
    except:
        return str(x)

# ===============================
# OPENLIBRARY SEARCH
# ===============================
def get_cover_id(title, author):
    try:
        q = quote_plus(f"{title} {author}")
        url = f"https://openlibrary.org/search.json?q={q}&limit=1"

        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()

        if "docs" in data and len(data["docs"]) > 0:
            return data["docs"][0].get("cover_i")
    except Exception as e:
        return None
    return None

def download_cover(cover_id, save_path):
    try:
        url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
        r = requests.get(url, headers=HEADERS, timeout=10)

        if r.status_code == 200 and len(r.content) > 5000:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except:
        pass
    return False

# ===============================
# MAIN
# ===============================
def main():
    print("📚 Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    items = pd.DataFrame({
        "item_id": df.index.astype(str),
        "title": df["Book"],
        "author": df["Author"],
        "description": df["Description"],
        "genres": df["Genres"].apply(parse_genres),
        "rating": df["Avg_Rating"],
        "ratings_count": df["Num_Ratings"],
    })

    items = items.dropna(subset=["title", "description", "genres"])
    items = items.drop_duplicates(subset=["title", "author"])
    items = items.sort_values("ratings_count", ascending=False).head(2000)
    items = items.reset_index(drop=True)

    # TEXT cho TF-IDF
    items["text"] = (
        items["title"].apply(clean_text) + " " +
        items["genres"].apply(clean_text) + " " +
        items["genres"].apply(clean_text) + " " +
        items["description"].apply(clean_text)
    )

    # ===============================
    # DOWNLOAD IMAGES
    # ===============================
    image_paths = []

    print("🖼️ Downloading book covers...")
    success = 0

    for idx, row in tqdm(items.iterrows(), total=len(items)):
        img_path = IMAGE_DIR / f"{idx}.jpg"

        if img_path.exists():
            image_paths.append(str(img_path))
            success += 1
            continue

        cover_id = get_cover_id(row["title"], row["author"])
        if cover_id and download_cover(cover_id, img_path):
            image_paths.append(str(img_path))
            success += 1
        else:
            image_paths.append("")

    items["image_path"] = image_paths
    items.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ DONE: {success}/{len(items)} images downloaded")

# ===============================
if __name__ == "__main__":
    main()
