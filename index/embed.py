import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"
EMB_NPY = ROOT / "data" / "processed" / "embeddings.npy"
META_JSONL = ROOT / "data" / "processed" / "chunk_meta.jsonl"

def main():
    # small, fast embedding model -> sentence-transformers
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # read all chunks into a list
    chunk_texts = []
    chunk_meta = []

    for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)

        # chunk text is embeded
        chunk_texts.append(obj["text"])

        chunk_meta.append({
            "chunk_id": obj["chunk_id"],
            "doc_id": obj["doc_id"],
            "title": obj.get("title", ""),
            "chunk_index": obj.get("chunk_index", -1),
            "text": obj["text"],
        })

    print("chunks number", len(chunk_texts))
    print("Embedding model:", model_name)

    # encode -> numpy array (N, D)
    vectors = model.encode(chunk_texts, show_progress_bar=True)

    vectors = np.array(vectors, dtype=np.float32)

    # save embeddings
    EMB_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_NPY, vectors)

    with META_JSONL.open("w", encoding="utf-8") as f:
        for m in chunk_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("Saved embeddings to:", EMB_NPY)
    print("Saved metadata to:", META_JSONL)
    print("Embeddings shape:", vectors.shape)

if __name__ == "__main__":
    main()
