# AVY System Architecture & Technical Specifications

## 1. System Overview

**AVY** is an end-to-end streaming live conversational RAG pipeline built for continuous speech transcripts. The architecture is engineered around low-latency state transitions, parallel subquery execution, rank fusion, and grounded streaming.

```text
┌──────────────────────────────────────────────────────────────┐
│                    Streaming Transcript                      │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│            Three-State Retrieval Controller                  │
│       • WAIT: Incomplete semantic units                      │
│       • NO_RETRIEVE: Conversational filler & greetings       │
│       • RETRIEVE: Sufficient information                     │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│          Session Context & Anaphora Refinement               │
│       • Contextualizes pronouns ('its', 'their')             │
│       • Fuses past session evidence                          │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│           Multi-Intent Query Decomposition                   │
│       • Disaggregates compound spoken questions              │
│       • Cleans and deduplicates subqueries                   │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              Multi-Query Vector Retrieval                    │
│       • Parallel search in FAISS IndexFlatIP                 │
│       • Preserves query_id, document_id, chunk_id            │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│         Evidence Fusion & Deduplication (RRF)                │
│       • Formula: sum(1 / (k + rank))                         │
│       • Normalizes scores & merges duplicates                │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                 Two-Stage Reranking                          │
│       • Cross-Encoder / Lexical cross-attention              │
│       • Records latency in milliseconds                      │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│           Grounded Synthesis & Traceable Citations           │
│       • Constrained prompt with numbered evidence            │
│       • Maps [1], [2] to exact document chunks               │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                 Live SSE Streaming Engine                    │
│       • Real-time token streaming with TTFT gauges           │
│       • JSONL structured telemetry emission                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 Retrieval Controller (`src/avy/retrieval/controller.py`)
- **Inputs**: Incoming text chunk, accumulated session transcript, active session context.
- **Outputs**: `ControllerDecision(state, reason, confidence, is_early_retrieval)`.
- **States**:
  - `WAIT`: Incomplete query, trailing prepositions, dangling conjunctions.
  - `NO_RETRIEVE`: Conversational greetings, gratitude, small talk.
  - `RETRIEVE`: Informational predicate and actionable entity present.

### 2.2 Multi-Intent Decomposition (`src/avy/retrieval/decomposition.py`)
- **Inputs**: Spoken natural language utterance.
- **Mechanism**: LLM provider reasoning with structured JSON output schema + heuristic conjunction parser fallback.
- **Outputs**: `QueryDecomposition` with unique `Subquery` instances.

### 2.3 Vector Store & Embeddings (`src/avy/retrieval/vectorstore.py`, `embeddings.py`)
- **Index**: FAISS `IndexFlatIP` (Exact Inner Product on L2-normalized float32 vectors).
- **Embeddings**: `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors) with pure NumPy subword fallback.
- **Search Complexity**: $O(N \cdot D)$ candidate scoring with zero external database dependencies.

### 2.4 Evidence Fusion (`src/avy/retrieval/fusion.py`)
- **Algorithm**: Reciprocal Rank Fusion (RRF) with constant $k = 60$.
- **Formula**:
  $$RRF(d) = \sum_{q \in Q} \frac{1}{60 + r_q(d)}$$
- **Deduplication**: Chunks appearing across multiple subqueries accumulate rank weights and are deduplicated by chunk ID.

### 2.5 Reranker (`src/avy/retrieval/reranker.py`)
- **Models**: `BAAI/bge-reranker-base` cross-attention or lexical term-overlap scorer.
- **Latency**: Measured via high-resolution monotonic timer (`Stopwatch`).

### 2.6 Grounding & Citations (`src/avy/retrieval/grounding.py`)
- **Prompt Structure**: System prompt enforces strict compliance with numbered evidence blocks `[1]`, `[2]`.
- **Extraction**: Scans generated text for bracketed reference tokens and constructs `Citation` metadata.

### 2.7 Telemetry Engine (`src/avy/telemetry/logger.py`)
- **Format**: JSON Lines (JSONL).
- **Safety**: Deep recursive credential scrubber (`_scrub_secrets`) removes API keys, tokens, and authorization headers before logging.
