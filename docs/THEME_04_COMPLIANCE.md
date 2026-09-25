# Samsung PRISM GenAI Hackathon (Theme 04) — Compliance Matrix

| Hackathon Requirement | Implementation Component | Source File | Verification Method |
|---|---|---|---|
| **Streaming Transcript Input** | Incremental chunk ingestion via SSE generator | `src/avy/streaming/engine.py` | `tests/test_streaming.py` |
| **Retrieval Controller** | Three-state decision engine (`WAIT`, `RETRIEVE`, `NO_RETRIEVE`) | `src/avy/retrieval/controller.py` | `tests/test_controller.py` |
| **Early Retrieval** | Mid-utterance semantic triggering flag | `src/avy/retrieval/controller.py` | `tests/test_controller.py::test_controller_early_retrieval` |
| **Multi-Intent Decomposition** | LLM structured query decomposition & deduplication | `src/avy/retrieval/decomposition.py` | `tests/test_decomposition.py` |
| **Multi-Query Retrieval** | Independent subquery search with provenance tracking | `src/avy/retrieval/multi_query.py` | `tests/test_retrieval.py` |
| **FAISS Vector Search** | L2-normalized cosine index and metadata store | `src/avy/retrieval/vectorstore.py` | `tests/test_retrieval.py` |
| **Evidence Fusion & Dedup** | Reciprocal Rank Fusion ($k=60$) | `src/avy/retrieval/fusion.py` | `tests/test_fusion.py` |
| **Reranking** | Cross-attention / lexical reranker with latency tracking | `src/avy/retrieval/reranker.py` | `tests/test_reranker.py` |
| **Session Context** | Session manager with multi-turn conversation memory | `src/avy/context/session.py` | `tests/test_session.py` |
| **Session Refinement** | Anaphora resolution and late detail query synthesis | `src/avy/context/session.py` | `tests/test_session.py::test_query_contextualization` |
| **Grounded Synthesis** | Strict evidence-constrained prompt assembly | `src/avy/retrieval/grounding.py` | `tests/test_grounding.py` |
| **Traceable Citations** | Deterministic `[1]`, `[2]` citation mapper | `src/avy/retrieval/grounding.py` | `tests/test_grounding.py::test_extract_citations` |
| **Streaming Synthesis** | Token streaming with TTFT and phase timestamps | `src/avy/streaming/engine.py` | `tests/test_streaming.py` |
| **Structured Telemetry** | JSON/JSONL logger for 14 required event types | `src/avy/telemetry/logger.py` | `tests/test_telemetry.py` |
| **Evaluation Harness** | Benchmark runner evaluating recall, groundedness, TTFT | `src/avy/evaluation/harness.py` | `scripts/evaluate.py`, `tests/test_end_to_end.py` |
| **Reproducibility** | Complete setup scripts, Makefile, `.env.example` | `Makefile`, `scripts/build_index.py` | End-to-end execution |
