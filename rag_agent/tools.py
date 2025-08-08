from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from .config import CONFIG

try:
    from tavily import TavilyClient  # type: ignore
except Exception:  # pragma: no cover
    TavilyClient = None  # type: ignore


def web_search(query: str, k: int = 5) -> List[Document]:
    """Perform web search via Tavily, return as Documents.

    If Tavily is not configured or available, returns an empty list.
    """
    api_key = CONFIG.search.tavily_api_key
    if not api_key or TavilyClient is None:
        return []

    client = TavilyClient(api_key=api_key)
    results = client.search(query=query, max_results=k)

    documents: List[Document] = []
    for item in results.get("results", []):
        content = item.get("content") or item.get("snippet") or ""
        url = item.get("url") or item.get("source") or ""
        title = item.get("title") or ""
        metadata = {"source": url, "title": title, "provider": "tavily"}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents