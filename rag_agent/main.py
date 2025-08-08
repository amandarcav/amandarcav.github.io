from __future__ import annotations

import argparse
import os
from typing import List

from langchain_core.documents import Document

from .config import CONFIG
from .vectorstore import create_faiss_retriever, load_text_documents_from_dir, split_documents
from .graph import build_graph


def run(question: str, data_dir: str) -> None:
    print("Loading documents from:", os.path.abspath(data_dir))
    raw_docs: List[Document] = load_text_documents_from_dir(data_dir)
    if not raw_docs:
        print("No .txt documents found. Please add files to:", data_dir)
        return

    print(f"Loaded {len(raw_docs)} raw docs. Splitting...")
    chunks = split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks. Building vector store...")
    retriever = create_faiss_retriever(chunks, k=CONFIG.top_k)

    print("Compiling LangGraph...")
    app = build_graph(retriever)

    print("Running agent...\n")
    result = app.invoke({"question": question, "rewrites": 0})

    answer = result.get("generation") or ""
    print("Answer:\n")
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Agentic RAG")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory with .txt documents",
    )
    args = parser.parse_args()

    run(args.question, args.data_dir)