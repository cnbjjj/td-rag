from config import (
    RAW_HTML_DIR, 
    CHUNKS_DIR, 
    CHUNKED_JSONL_FILE
)
import os
import argparse
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
import html

def categorize(filename):
    if filename.startswith("Category-"):
        return "Category"
    if filename.startswith("Palette-"):
        return "Palette"
    if filename.startswith("Experimental-"):
        return "Experimental"
    if "_Class" in filename:
        return "PythonClass"
    for suffix in ["SOP", "TOP", "CHOP", "COMP", "DAT", "MAT"]:
        if f"_{suffix}" in filename:
            return suffix
    return "Misc"

def split_param_line(line):
    match = re.match(r"(.+?)([a-zA-Z0-9_]+)-$", line.strip())
    if match:
        label, param = match.groups()
        return f"- **{label.strip()}** (`{param}`)"
    return f"- {line.strip()}"

def clean_html_to_markdown(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    title_tag = soup.select_one("h1.mw-first-heading span.mw-page-title-main")
    filename = os.path.basename(filepath)
    title = title_tag.text.strip() if title_tag else filename.replace(".htm", "").replace("_", " ")
    content_div = soup.select_one("#mw-content-text .mw-parser-output")
    if not content_div:
        return title, None, []

    lines = [f"# {title}\n"]
    chunks = []

    for elem in content_div.find_all(['h2', 'h3', 'p', 'li']):
        text = elem.get_text(strip=True)
        if not text:
            continue
        tag = elem.name

        if tag == 'h2':
            lines.append(f"\n---\n\n## {text}\n")
        elif tag == 'h3':
            lines.append(f"\n### {text}\n")
        elif tag == 'li':
            lines.append(split_param_line(text))
        else:
            lines.append(text)

        if len(text) > 20:
            chunks.append(text)

    return title, '\n\n'.join(lines), chunks

def clean_all_html(input_dir, output_dir, jsonl_path):
    os.makedirs(output_dir, exist_ok=True)
    chunk_records = []

    for filename in os.listdir(input_dir):
        if not filename.endswith(".htm") and not filename.endswith(".html"):
            continue

        path = os.path.join(input_dir, filename)
        category = categorize(filename)
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        title, markdown, chunks = clean_html_to_markdown(path)
        if markdown:
            out_name = f"{title.replace(' ', '_')}.md"
            out_path = os.path.join(category_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(markdown)

            for i, chunk in enumerate(chunks):
                chunk_records.append({
                    "filename": filename,
                    "category": category,
                    "title": title,
                    "chunk_index": i,
                    "text": chunk.strip()
                })

            print(f"{filename} → {out_name}")
        else:
            print(f"Skipped (no content): {filename}")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in chunk_records:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{len(chunk_records)} chunks saved to {jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=RAW_HTML_DIR)
    parser.add_argument("--output_dir", default=CHUNKS_DIR)
    parser.add_argument("--jsonl", default=CHUNKED_JSONL_FILE)
    args = parser.parse_args()
    clean_all_html(args.input_dir, args.output_dir, args.jsonl)
