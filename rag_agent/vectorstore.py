from __future__ import annotations

import glob
import os
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .config import CONFIG


def load_text_documents_from_dir(directory: str) -> List[Document]:
    text_file_paths = sorted(glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True))
    documents: List[Document] = []
    for file_path in text_file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
        metadata = {"source": os.path.relpath(file_path, directory)}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def split_documents(documents: Iterable[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(list(documents))


def create_faiss_retriever(
    documents: List[Document],
    embedding_model_name: str | None = None,
    k: int = 6,
):
    embedding_model_name = embedding_model_name or CONFIG.openai.embedding_model
    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever