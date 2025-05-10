from config import (
    DEFAULT_EMBEDDING_MODEL, 
    DEFAULT_INDEX_PATH, 
    DEFAULT_METADATA_PATH, 
    CHUNKED_JSONL_FILE, 
    MODEL_DIR
)
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import argparse

def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def build_index(chunks, model_name):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=True)
    return vectors

def save_index(index, output_path):
    faiss.write_index(index, str(output_path))

def save_metadata(chunks, metadata_path):
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def main(chunks_path, index_path, metadata_path, embedding_model):
    print("Loading chunks...")
    chunks = load_chunks(chunks_path)

    print("Generating embeddings...")
    vectors = build_index(chunks, embedding_model)

    print("Building FAISS index...")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    print(f"Saving index to {index_path}")
    save_index(index, index_path)

    print(f"Saving metadata to {metadata_path}")
    save_metadata(chunks, metadata_path)

    print(f"Done. Indexed {len(chunks)} chunks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_path", default=CHUNKED_JSONL_FILE, help="Path to chunks.jsonl")
    parser.add_argument("--index_path", default=str(MODEL_DIR/"td_index_mini.faiss"), help="Path to save FAISS index")
    parser.add_argument("--metadata_path", default=str(MODEL_DIR/"td_metadata_mini.json"), help="Path to save chunk metadata")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name")
    args = parser.parse_args()
    main(args.chunks_path, args.index_path, args.metadata_path, args.embedding_model)
