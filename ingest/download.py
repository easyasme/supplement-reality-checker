import time
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"

# source list file (id|url per line)
SOURCES_FILE = ROOT / "ingest" / "sources.txt"

def read_sources():
    "Read sources.txt and return a list of (id, url)"
    lines = SOURCES_FILE.read_text(encoding="utf-8").splitlines()
    results = []
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue

        doc_id, url = line.split("|")
        results.append((doc_id.strip(), url.strip()))
    return results

def download_html(doc_id, url):
    "Download one page and save it as data/raw/<doc_id>.html"
    RAW_DIR.mkdir(parents=True, exist_ok=True) # Check folder exists
    out_path = RAW_DIR / f"{doc_id}.html"

    r = requests.get(url, timeout=30)

    # if server returns error, stop here
    r.raise_for_status()

    # save HTML content
    out_path.write_text(r.text, encoding="utf-8")
    return out_path

def main():
    sources = read_sources()
    print("How many sources?", len(sources))

    for doc_id, url in tqdm(sources):
        saved = download_html(doc_id, url)
        print("Saved:", saved)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
