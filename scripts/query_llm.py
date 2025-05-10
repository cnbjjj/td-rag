from config import (
    DEFAULT_INDEX_PATH, 
    DEFAULT_METADATA_PATH, 
    MODEL_DIR, 
    DEFAULT_EMBEDDING_MODEL, 
    PROMPT
)
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import subprocess
import os

# disable TensorFlow to avoid loading it
os.environ["TRANSFORMERS_NO_TF"] = "1"

def search(question, model, index, metadata, top_k=3):
    query_vector = model.encode([question])
    D, I = index.search(np.array(query_vector), top_k)
    return [metadata[i] for i in I[0]]

def ask_ollama(context_chunks, question, model_name="deepseek-r1:7b"):
    system_prompt = PROMPT

    # merge context chunks into a single string
    reference_section = "\n\nReference snippets:\n\n"
    for i, chunk in enumerate(context_chunks):
        title = chunk.get("title", "unknown")
        category = chunk.get("category", "")
        reference_section += f"[Doc {i+1} from {title} - {category}]:\n{chunk['text']}\n\n"

    # create the full prompt
    full_prompt = f"{system_prompt}{reference_section}User question:\n{question}\n\nAnswer:"

    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=full_prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            print("Ollama Error:", result.stderr.decode())
            return ""
        return result.stdout.decode()
    except Exception as e:
        print("Failed to run Ollama:", e)
        return ""

def load_index_and_metadata(index_path=DEFAULT_INDEX_PATH, metadata_path=DEFAULT_METADATA_PATH, model_name="all-mpnet-base-v2"):
    index = faiss.read_index(str(index_path))
    model = SentenceTransformer(model_name)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata, model

def main():
    print("Loading index and metadata...")
    index, metadata, model = load_index_and_metadata()
    
    while True:
        question = input("\nAsk your TouchDesigner question (or 'exit'): ")
        if question.strip().lower() in {"exit", "quit"}:
            break

        print("Searching...")
        context = search(question, model, index, metadata)
        if context:
            print("\nContext snippets:")
            for i, chunk in enumerate(context):
                title = chunk.get("title", "unknown")
                category = chunk.get("category", "")
                print(f"[Doc {i+1} from {title} - {category}]:\n{chunk['text']}\n")

        print("Answering...")
        answer = ask_ollama(context, question)
        print("\nAnswer:\n", answer.strip())

if __name__ == "__main__":
    main()
