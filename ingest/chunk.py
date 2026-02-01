import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_FILE = ROOT / "data" / "processed" / "docs.jsonl"
OUT_FILE = ROOT / "data" / "processed" / "chunks.jsonl"

def split_into_chunks(text: str, max_chars: int = 1200, overlap_chars: int = 200):
    """
    Split long text into chunks by character count.
    - max_chars: max size of one chunk
    - overlap_chars: overlap so we don't cut important context
    """
    chunks = []
    start = 0

    # sliding window chunking
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - overlap_chars

        # stop if overlap not move forward
        if start < 0:
            start = 0
        if start >= len(text):
            break

    return chunks

def main():
    if not DOCS_FILE.exists():
        print("docs.jsonl not found:", DOCS_FILE)
        return

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_chunks = 0

    with DOCS_FILE.open("r", encoding="utf-8") as f_in, OUT_FILE.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            doc_id = doc["doc_id"]
            title = doc.get("title", "")
            text = doc["text"]

            # Chunking
            chunks = split_into_chunks(text, max_chars=1200, overlap_chars=200)

            # write each chunk as one JSON line
            for i, chunk_text in enumerate(chunks):
                record = {
                    "chunk_id": f"{doc_id}::chunk_{i}",  # stable id
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_index": i,
                    "text": chunk_text,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

            total_docs += 1

    print("Docs processed:", total_docs)
    print("Total chunks:", total_chunks)
    print("Saved to:", OUT_FILE)

if __name__ == "__main__":
    main()
