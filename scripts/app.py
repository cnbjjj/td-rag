from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFUALT_INDEX_NAME,
    DEFAULT_METADATA_PATH,
    MODEL_DIR,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_METADATA_NAME,
    PORT,
    HOST,
)
from fastapi import FastAPI
import gradio as gr
from query_llm import search, ask_ollama
from sentence_transformers import SentenceTransformer
import faiss, json, numpy as np
import os

llm_model, model, index, metadata = None, None, None, None

def get_available_models_and_indexes():
    models = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]
    indexes = sorted([
        f.name for f in MODEL_DIR.glob("*.faiss")
    ])
    metadata = sorted([
        f.name for f in MODEL_DIR.glob("*.json")
    ])
    llm_models = ["deepseek-r1:7b", "llama3.1:latest", "mistral:latest"]

    return {
        "embedding_models": models,
        "index_files": indexes,
        "metadata_files": metadata,
        "llm_models": llm_models
    }

def set_model_and_index(selected_model, selected_index, ll_model=llm_model, metadata_file=DEFAULT_METADATA_NAME):
    global model, index, llm_model, metadata
    model = SentenceTransformer(selected_model)
    index = faiss.read_index(str(MODEL_DIR / selected_index))
    llm_model = ll_model
    with open(MODEL_DIR/metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

def answer_question(query, selected_model, selected_index, selected_llm_model=llm_model, metadata_file=DEFAULT_METADATA_NAME):
    set_model_and_index(selected_model, selected_index, selected_llm_model, metadata_file)
    print(f"Selected model: {selected_model}, index: {selected_index}, llm_model: {selected_llm_model}")
    chunks = search(query, model, index, metadata, top_k=3)
    response = ask_ollama(chunks, query, llm_model)
    refs = "\n\n".join([f"[{c['title']}]: {c['text']}" for c in chunks])
    return response.strip(), refs

def save_feedback(query, answer):
    os.makedirs("flagged_logs", exist_ok=True)
    with open("flagged_logs/custom_flags.txt", "a", encoding="utf-8") as f:
        f.write(f"\n---\nQUESTION:\n{query}\nANSWER:\n{answer}\n")
    return "Feedback submitted!"

app = FastAPI()

set_model_and_index(DEFAULT_EMBEDDING_MODEL, DEFUALT_INDEX_NAME, DEFAULT_OLLAMA_MODEL, DEFAULT_METADATA_PATH)
available = get_available_models_and_indexes()
model_options = available["embedding_models"]
index_options = available["index_files"]
metadata_options = available["metadata_files"]
llm_model_options = available["llm_models"]

with gr.Blocks() as iface:
    gr.Markdown("<h1 style='text-align: center;'>TouchDesigner RAG</h1>")
    gr.Markdown("<p style='text-align: center;'>Ask questions about TouchDesigner. This tool uses local RAG with FAISS and Ollama.</p>")

    with gr.Row():
        with gr.Column():
            query = gr.Textbox(label="Query", lines=10, placeholder="Ask about TouchDesigner...")
            model_dropdown = gr.Dropdown(choices=model_options, label="Select Embedding Model", value=model_options[0])
            index_dropdown = gr.Dropdown(choices=index_options, label="Select FAISS Index", value=index_options[0])
            metadata_dropdown = gr.Dropdown(choices=metadata_options, label="Select Metadata File", value=metadata_options[0])
            llm_model_dropdown = gr.Dropdown(choices=llm_model_options, label="Select LLM Model", value=llm_model_options[0])
            submit_button = gr.Button("Submit", variant="primary")

        with gr.Column():
            answer_output = gr.Textbox(label="Answer", lines=10)
            reference_output = gr.Textbox(label="Reference Snippets", lines=10)

    submit_button.click(
        fn=answer_question,
        inputs=[query, model_dropdown, index_dropdown, llm_model_dropdown, metadata_dropdown],
        outputs=[answer_output, reference_output]
    )


@app.get("/models")
def get_models():
    return get_available_models_and_indexes()

app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)