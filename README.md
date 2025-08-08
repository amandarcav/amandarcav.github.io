# LangGraph Agentic RAG

A minimal, runnable Agentic RAG built with LangGraph, inspired by the tutorial at `https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/`.

### Features
- Document ingestion from local `rag_agent/data` (text files)
- FAISS vector store with OpenAI embeddings
- Agentic loop: retrieve → LLM-based document grading → query rewrite if needed → generate with citations
- Optional web search via Tavily (if `TAVILY_API_KEY` provided)

### Requirements
- Python 3.10+
- OpenAI API key set as `OPENAI_API_KEY`
- Optional: `TAVILY_API_KEY` for web search

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Create a `.env` file (or export env vars):

```bash
OPENAI_API_KEY=YOUR_OPENAI_KEY
# Optional overrides
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
TAVILY_API_KEY=YOUR_TAVILY_KEY
```

### Usage

Run the CLI with a question:

```bash
python -m rag_agent.main --question "What is LangGraph and how does it help build RAG systems?"
```

You can also point to another data folder:

```bash
python -m rag_agent.main --question "Explain retrieval augmented generation" --data-dir ./rag_agent/data
```

### Structure
- `rag_agent/config.py`: configuration and environment
- `rag_agent/vectorstore.py`: load/split docs and build FAISS retriever
- `rag_agent/tools.py`: optional web search tool (Tavily)
- `rag_agent/graph.py`: LangGraph state, nodes, and graph wiring
- `rag_agent/main.py`: CLI entry point
- `rag_agent/data/`: sample text corpus

### Notes
- If no web search key is provided, the agent will only use local docs.
- The agent includes a retry loop for query rewriting when retrieval quality is low.
- All LLM calls use OpenAI via `langchain-openai`.