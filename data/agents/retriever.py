# Agent 1: Retrieval (FAISS-based)
import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "data" / "processed" / "faiss.index"
META_FILE = ROOT / "data" / "processed" / "chunk_meta.jsonl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _load_meta():
    #Load chunk metadata in the same order as embeddings
    metas = []
    for line in META_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            metas.append(json.loads(line))
    return metas

def retrieve(claim: str, k: int = 5):
    """Return top-k evidence chunks for a claim."""
    # load models + index
    embedder = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(str(INDEX_FILE))
    metas = _load_meta()

    # embed query and normalize
    vec = embedder.encode([claim])
    vec = np.array(vec, dtype=np.float32)
    faiss.normalize_L2(vec)

    scores, idxs = index.search(vec, k)

    results = []
    for rank in range(k):
        idx = int(idxs[0][rank])
        m = metas[idx]
        results.append({
            "rank": rank + 1,
            "score": float(scores[0][rank]),
            "chunk_id": m["chunk_id"],
            "doc_id": m["doc_id"],
            "text": m["text"],
        })
    return results
