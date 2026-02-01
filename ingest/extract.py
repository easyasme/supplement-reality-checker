import json
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "processed" / "docs.jsonl"

def html_to_text(html_str: str) -> str:
    # HTML string to plain text using BeautifulSoup and lxml parser
    soup = BeautifulSoup(html_str, "lxml")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # use main - main texts are in main
    main = soup.find("main")
    if main is None:
        main = soup

    text = main.get_text(separator="\n")

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # remove empty lines
    return "\n".join(lines)

def extract_title(html_str: str) -> str:
    # Try to pull good title from HTML
    soup = BeautifulSoup(html_str, "lxml")
    if soup.title and soup.title.text: # use <title> tag if present
        return soup.title.text.strip()
    return "Unknown Title"

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    html_files = sorted(RAW_DIR.glob("*.html"))
    if not html_files:
        print("No HTML files found in data/raw/")
        return

    count = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for path in html_files:
            doc_id = path.stem  # filename without .html
            html = path.read_text(encoding="utf-8")

            title = extract_title(html)
            text = html_to_text(html)

            record = {
                "doc_id": doc_id,
                "title": title,
                "source": "NIH ODS",
                "raw_path": str(path),
                "text": text,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")  # JSONL format
            count += 1

    print(f"Saved {count} docs to: {OUT_FILE}")

if __name__ == "__main__":
    main()
