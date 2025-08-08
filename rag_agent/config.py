from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")
    chat_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


@dataclass(frozen=True)
class SearchConfig:
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY")


@dataclass(frozen=True)
class AppConfig:
    openai: OpenAIConfig = OpenAIConfig()
    search: SearchConfig = SearchConfig()

    # Agentic loop constraints
    max_rewrites: int = int(os.getenv("MAX_REWRITES", "2"))
    top_k: int = int(os.getenv("TOP_K", "6"))


CONFIG = AppConfig()