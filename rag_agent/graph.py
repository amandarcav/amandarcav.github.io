from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .config import CONFIG
from .tools import web_search


class RAGState(TypedDict, total=False):
    question: str
    query: str
    documents: List[Document]
    filtered_documents: List[Document]
    generation: str
    rewrites: int


def _get_chat_llm() -> ChatOpenAI:
    return ChatOpenAI(model=CONFIG.openai.chat_model, temperature=0)


def node_transform_query(state: RAGState) -> Dict:
    llm = _get_chat_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a query rewriting assistant for retrieval. Make the user's question concise and retrieval-friendly."),
            (
                "human",
                "Original question: {question}\n\nIf helpful, here are partial context snippets (may be empty):\n{context}\n\nRewrite the question as a single concise search query only.",
            ),
        ]
    )

    context = "\n\n".join(doc.page_content[:400] for doc in state.get("documents", [])[:3])
    chain: Runnable = prompt | llm
    response = chain.invoke({"question": state.get("question", ""), "context": context})
    query = response.content.strip().replace("\n", " ")
    return {"query": query, "rewrites": int(state.get("rewrites", 0))}


def node_retrieve(state: RAGState, retriever) -> Dict:
    query = state.get("query") or state.get("question")
    if not query:
        return {"documents": []}
    documents = retriever.get_relevant_documents(query)
    # Augment with web search if available
    web_docs = web_search(query, k=max(2, CONFIG.top_k // 2))
    if web_docs:
        documents.extend(web_docs)
    return {"documents": documents}


def node_grade_documents(state: RAGState) -> Dict:
    documents = state.get("documents", [])
    if not documents:
        return {"filtered_documents": []}

    llm = _get_chat_llm()
    grading_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a strict relevance grader. Answer with only 'YES' or 'NO'."),
            (
                "human",
                "Question: {question}\n\nContext snippet:\n{context}\n\nIs this context relevant to answering the question? Reply 'YES' or 'NO' only.",
            ),
        ]
    )

    filtered: List[Document] = []
    for doc in documents:
        content = doc.page_content[:1200]
        chain: Runnable = grading_prompt | llm
        res = chain.invoke({"question": state.get("question", ""), "context": content})
        decision = (res.content or "").strip().upper()
        if decision.startswith("Y"):
            filtered.append(doc)

    return {"filtered_documents": filtered}


def router_after_grading(state: RAGState) -> Literal["generate", "rewrite"]:
    filtered = state.get("filtered_documents", [])
    rewrites = int(state.get("rewrites", 0))
    if len(filtered) >= 2 or rewrites >= CONFIG.max_rewrites:
        return "generate"
    return "rewrite"


def node_increment_rewrites(state: RAGState) -> Dict:
    current = int(state.get("rewrites", 0))
    return {"rewrites": current + 1}


def node_generate(state: RAGState) -> Dict:
    llm = _get_chat_llm()
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use ONLY the provided context to answer. If the context is insufficient, say you don't know. Cite sources as [n] where n is the index of the cited chunk.",
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}\n\nWrite a concise answer with citations.",
            ),
        ]
    )

    docs = state.get("filtered_documents") or state.get("documents") or []
    # Prepare context and a mapping for citations
    context_lines: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source") if hasattr(doc, "metadata") else None
        title = doc.metadata.get("title") if hasattr(doc, "metadata") else None
        header = f"[{idx}] " + (title or source or "document")
        context_lines.append(header + "\n" + doc.page_content[:1600])
    context_str = "\n\n".join(context_lines)

    chain: Runnable = answer_prompt | llm
    res = chain.invoke({"question": state.get("question", ""), "context": context_str})
    generation = (res.content or "").strip()
    return {"generation": generation}


def build_graph(retriever) -> Runnable:
    graph = StateGraph(RAGState)

    # Bind nodes
    graph.add_node("transform_query", node_transform_query)
    graph.add_node("retrieve", lambda state: node_retrieve(state, retriever))
    graph.add_node("grade_documents", node_grade_documents)
    graph.add_node("increment_rewrites", node_increment_rewrites)
    graph.add_node("generate", node_generate)

    # Edges
    graph.add_edge(START, "transform_query")
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        router_after_grading,
        {
            "rewrite": "increment_rewrites",
            "generate": "generate",
        },
    )

    graph.add_edge("increment_rewrites", "transform_query")
    graph.add_edge("generate", END)

    return graph.compile()