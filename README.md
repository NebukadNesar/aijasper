# AIJasper — Local Java RAG Pipeline

Local, privacy-first Retrieval-Augmented Generation (RAG) pipeline built in Java using **ONNX Runtime** for dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and **Ollama** (`qwen3:8b`) running on Docker with GPU acceleration.

---

## Features

- **Document Ingestion & Chunking:** Extracts text from PDFs (and extensible to DOCX) with configurable word-level chunking and overlap.
- **Local ONNX Embeddings:** Generates 384-dimensional normalized vectors locally via `all-MiniLM-L6-v2` and WordPiece tokenization (`vocab.txt`).
- **Binary Index Persistence (`IndexStore`):** Saves and loads precomputed chunks and embeddings to disk (`.ragidx`) for near-instant startup without re-embedding.
- **Vector Similarity Retrieval:** Ranks top-$k$ relevant passages using cosine similarity (`dotProduct`).
- **Local LLM Generation via Ollama:** Connects to Ollama's `/api/chat` endpoint with strict grounding rules and page/source citations.

---

## Architecture

```text
Documents (PDF / DOCX)
        │
        ▼
MultiDocumentLoader & TextChunker
        │
        ▼
WordPieceTokenizer ──► MiniLmEmbeddingService (ONNX 384-d)
        │
        ▼
    IndexStore (.ragidx binary cache)
        │
        ▼
DocumentRetriever (Top-K Cosine Similarity)
        │
        ▼
RagAnswerService ──► Ollama (qwen3:8b on GPU) ──► Grounded Answer + Citations
