from pathlib import Path
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parents[1]
EMB_NPY = ROOT / "data" / "processed" / "embeddings.npy"
INDEX_FILE = ROOT / "data" / "processed" / "faiss.index"

def main():
    vectors = np.load(EMB_NPY)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    n, d = vectors.shape
    print("Vectors shape:", vectors.shape)

    # cosine(a,b) == dot(norm(a), norm(b))
    faiss.normalize_L2(vectors)

    # simplest index
    index = faiss.IndexFlatIP(d)

    # add vectors into index
    index.add(vectors)
    print("Index size:", index.ntotal)

    faiss.write_index(index, str(INDEX_FILE))
    print("Saved FAISS index to:", INDEX_FILE)

if __name__ == "__main__":
    main()
