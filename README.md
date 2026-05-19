# Secure Banking RAG Engine (Local Inference)

A privacy-first, on-premise Retrieval-Augmented Generation (RAG) pipeline designed for querying highly sensitive regulatory documents (e.g., RBI KYC Master Directions). 

This project was built to demonstrate how enterprise banking environments can leverage LLMs without exposing customer data or proprietary policy information to third-party cloud APIs like OpenAI or Anthropic.

## Architecture Highlights

* **Zero Data Leakage:** 100% local inference using Llama 3 (8B) via Ollama. No data ever leaves the host machine.
* **Custom Hybrid Search:** Bypassed standard, fragile library wrappers to engineer a custom, deterministic merging of Semantic and Lexical search algorithms.
  * **Semantic (Dense):** ChromaDB powered by `all-MiniLM-L6-v2` for contextual understanding.
  * **Lexical (Sparse):** BM25 implementation for strict, keyword-exact matching (crucial for regulatory compliance acronyms and strict timelines).
* **Token Optimization & Deduplication:** Implemented strict Python dictionary-based deduplication *before* context window injection. This prevents token waste, reduces latency, and mitigates hallucination risks caused by repetitive context.
* **Component Isolation:** The architecture strictly separates ingestion, retrieval, and generation into independently testable microservices.

## Project Structure

```text
bank_rag_api/
├── data/
│   └── rbi_kyc_master.pdf       # The regulatory document (Not tracked in Git)
├── src/
│   ├── ingestion.py             # Vectorizes the PDF into ChromaDB chunks
│   ├── retrieval.py             # Executes and merges BM25 & Semantic searches
│   └── generator.py             # The main execution engine tying RAG to Llama 3
├── vector_db/                   # Local persistent Chroma database (Not tracked)
├── requirements.txt
├── .gitignore
└── README.md