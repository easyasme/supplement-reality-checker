import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "data" / "processed" / "faiss.index"
META_JSONL = ROOT / "data" / "processed" / "chunk_meta.jsonl"

def load_meta():
    # load meta lines into a list so "idx" maps to meta[idx]
    metas = []
    for line in META_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            metas.append(json.loads(line))
    return metas

def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    # load FAISS index
    index = faiss.read_index(str(INDEX_FILE))

    # load metadata (same order as embeddings were saved)
    metas = load_meta()

    query = input("Search query: ").strip()
    if not query:
        print("Empty query. Exiting.")
        return

    # embed query into vector
    q_vec = model.encode([query])
    q_vec = np.array(q_vec, dtype=np.float32)

    faiss.normalize_L2(q_vec)

    # top-k results
    k = 5
    scores, idxs = index.search(q_vec, k)

    print("\nTop results:\n")

    for rank in range(k):
        idx = int(idxs[0][rank])
        score = float(scores[0][rank])

        m = metas[idx]
        print(f"[{rank+1}] score={score:.4f}  doc_id={m['doc_id']}  chunk_id={m['chunk_id']}")
        print(m["text"][:400].replace("\n", " "))
        print("-" * 80)

if __name__ == "__main__":
    main()
