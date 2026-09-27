# AVY — Streaming Live RAG Assistant

Video link : https://drive.google.com/file/d/1uVeuRLFEjg_w_902f3jcKOu31ays_KnL/view?usp=sharing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-41%20passing-brightgreen.svg)](#testing)
[![Theme](https://img.shields.io/badge/Samsung%20PRISM%203rd%20GenAI-Theme%2004%3A%20Streaming%20Live%20RAG-blue)](https://github.com/Banisher2005/AVY)

> **AVY is a voice-first Streaming Live RAG assistant.**
> Speak naturally. AVY listens, understands complete utterances, retrieves grounded evidence, and speaks its answers back.

---

## The Core Problem

Traditional RAG assumes:

```
complete query → retrieve → generate
```

Human speech doesn't work that way. When speaking to an AI, people:

- Pause mid-sentence: *"Tell me about Samsung's..."*
- Ask compound questions: *"Compare Exynos 2400 with Exynos 2200 and explain battery efficiency"*
- Use pronouns in follow-ups: *"Now compare its display with the S23"*
- Say conversational filler: *"Hello AVY", "Thanks, that helps"*

**AVY solves this** by treating speech as a stream of chunks that must be buffered into a coherent utterance before retrieval.

---

## Architecture

```
                         USER
                          │
                          ▼
                    🎙  MICROPHONE
                          │
                    Web Speech API
                          │
                          ▼
                 STREAMING TRANSCRIPT
              (incremental chunks arrive)
                          │
                          ▼
                  UTTERANCE BUFFER
              (accumulate → one utterance)
                          │
                     complete utterance
                          │
                          ▼
               RETRIEVAL CONTROLLER
                   /         \         \
                WAIT        NO_RETRIEVE  RETRIEVE
                │               │           │
           buffer more    direct LLM    query refinement
                                          (pronoun resolve)
                                              │
                                     INTENT DECOMPOSITION
                                              │
                                      VECTOR RETRIEVAL
                                       (FAISS parallel)
                                              │
                                        RRF FUSION
                                              │
                                        RERANKING
                                              │
                                        GROUNDING
                                              │
                                    GROUNDED SYNTHESIS
                                              │
                                    STREAMING RESPONSE
                                       /            \
                                      ▼              ▼
                                 TEXT UI           TTS
                                                    │
                                                    ▼
                                            🔊 SPOKEN AVY
```

---

## Key Design Principles

### 1. Chunks Are Not Queries

The most important architectural principle.

**WRONG:**
```
chunk 1 → RAG pipeline → answer 1
chunk 2 → RAG pipeline → answer 2
chunk 3 → RAG pipeline → answer 3
```

**CORRECT:**
```
chunk 1 → utterance buffer
chunk 2 → utterance buffer
chunk 3 → utterance buffer → complete utterance → ONE RAG pipeline → ONE answer
```

The `UtteranceManager` (`src/avy/streaming/utterance_manager.py`) handles this. One utterance always produces exactly one RAG execution.

### 2. Three-State Retrieval Controller

Every utterance is classified before any retrieval happens:

| State | When | What happens |
|---|---|---|
| `WAIT` | Utterance seems incomplete (*"Tell me about..."*) | Buffer more input |
| `NO_RETRIEVE` | Conversational (*"Hello AVY", "Thanks"*) | Direct LLM response, no retrieval |
| `RETRIEVE` | Informational intent detected | Full RAG pipeline |

### 3. Voice Is Primary

```
IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
```

The Web Speech API captures voice input. SpeechSynthesis speaks the response. Users can interrupt AVY while it's speaking by pressing the mic button.

---

## System Components

### Backend Modules (`src/avy/`)

| Module | Purpose |
|---|---|
| `streaming/utterance_manager.py` | **Utterance buffer** — accumulates chunks into complete utterances |
| `streaming/engine.py` | Central RAG orchestrator — 10-stage pipeline |
| `retrieval/controller.py` | 3-state classifier: WAIT / NO_RETRIEVE / RETRIEVE |
| `retrieval/decomposition.py` | Multi-intent query decomposer |
| `retrieval/embeddings.py` | FastEmbed (BAAI/bge-small-en-v1.5) + NumPy offline fallback |
| `retrieval/vectorstore.py` | FAISS with cosine similarity, save/load |
| `retrieval/multi_query.py` | Parallel multi-query retrieval |
| `retrieval/fusion.py` | Reciprocal Rank Fusion (RRF) |
| `retrieval/reranker.py` | Two-stage: cross-encoder + lexical fallback |
| `retrieval/grounding.py` | Evidence-constrained synthesis prompt builder |
| `context/session.py` | Multi-turn session memory, pronoun resolution |
| `providers/ollama.py` | Ollama local LLM (Qwen3, Llama, etc.) |
| `providers/mock.py` | Deterministic offline provider for testing |
| `providers/openai_compatible.py` | OpenAI-compatible endpoint (GPT-4o, vLLM) |
| `telemetry/logger.py` | Structured JSONL telemetry with secret scrubbing |
| `ui/server.py` | FastAPI: WebSocket (primary) + REST/SSE (fallback) |

### Frontend (`src/avy/ui/static/`)

| File | Purpose |
|---|---|
| `index.html` | Voice-first 2-column layout |
| `style.css` | Dark research-lab design system |
| `app.js` | Full voice state machine, WS client, STT, TTS, pipeline trace |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `WS` | `/ws/{session_id}` | **Primary voice channel** — bidirectional real-time |
| `POST` | `/api/query` | REST/SSE — complete utterance → streaming RAG events |
| `POST` | `/api/scenarios/run` | Stream a demo scenario via SSE |
| `GET` | `/api/scenarios` | List all demo scenarios |
| `GET` | `/api/health` | System health (provider, chunks, embeddings) |
| `GET` | `/api/session/{id}` | Inspect session history + evidence pool |
| `DELETE` | `/api/session/{id}` | Clear session memory |
| `GET` | `/api/sessions` | List all active sessions |
| `GET` | `/api/telemetry/{id}` | Structured telemetry events |
| `GET` | `/api/citations/{id}` | Session citations |

### WebSocket Protocol

**Client → Server:**
```json
{ "type": "utterance",  "text": "Complete spoken question" }
{ "type": "chunk",      "chunk": "partial...", "is_final": false }
{ "type": "chunk",      "chunk": "text",       "is_final": true }
{ "type": "clear" }
{ "type": "ping" }
```

**Server → Client:**
```json
{ "type": "utterance.finalized",  "text": "..." }
{ "type": "pipeline.started",     "utterance_id": "..." }
{ "type": "pipeline_stage",       "stage": "vector_retrieval", "status": "completed", "duration_ms": 43 }
{ "type": "controller_decision",  "state": "RETRIEVE", "is_early_retrieval": true }
{ "type": "evidence",             "evidence": [...] }
{ "type": "token",                "token": "Samsung" }
{ "type": "citations",            "citations": [...] }
{ "type": "timings",              "timings": { "ttft_ms": 142, ... } }
{ "type": "pipeline.completed",   "utterance_id": "..." }
```

---

## Installation & Setup

### Prerequisites
- Linux / macOS / WSL
- Python 3.10+
- (Optional) [Ollama](https://ollama.ai) with `qwen3:4b`
- Chrome or Edge for voice input (Web Speech API)

### Setup

```bash
git clone https://github.com/Banisher2005/AVY.git
cd AVY

# Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install
uv pip install -e ".[dev]"
# or: make install

# Configure
cp .env.example .env
# Edit .env to set AVY_PROVIDER=ollama (or mock for testing)
```

### Build the Vector Index

```bash
python scripts/build_index.py
```

This ingests all corpus documents from `corpus/` and builds a FAISS index in `data/index/`.

### Launch

```bash
python scripts/demo_server.py
```

Open **http://127.0.0.1:8000** in Chrome or Edge.

---

## Configuration

```ini
# .env

# LLM Provider: "ollama" | "openai" | "mock"
AVY_PROVIDER=ollama
AVY_MODEL=qwen3:4b
AVY_OLLAMA_HOST=http://127.0.0.1:11434

# Embeddings: "fastembed" | "local_dense"
AVY_EMBEDDING_PROVIDER=fastembed
AVY_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Reranker
AVY_RERANKER_ENABLED=true
AVY_RERANKER_MODEL=BAAI/bge-reranker-base

# Vector Store
AVY_CORPUS_DIR=corpus
AVY_INDEX_PATH=data/index/avy_faiss.index
AVY_METADATA_PATH=data/index/avy_metadata.json
AVY_TOP_K=5
AVY_RRF_K=60

# Early Retrieval
AVY_EARLY_RETRIEVAL_ENABLED=true

# Server
AVY_SERVER_HOST=127.0.0.1
AVY_SERVER_PORT=8000
```

---

## Voice Interaction Guide

### Speaking to AVY

1. **Open** `http://127.0.0.1:8000` in **Chrome or Edge** (required for Web Speech API)
2. **Tap the microphone** button — it turns blue and starts pulsing
3. **Speak naturally** — AVY displays your transcript live as you talk
4. **Stop speaking** — AVY automatically finalizes your utterance and runs the pipeline
5. **AVY responds** — text streams into the conversation, then AVY speaks the answer
6. **Interrupt** — tap the mic or press Stop while AVY is speaking

### Text Input Fallback

If microphone is unavailable, use the text input field at the bottom.  
Text input uses the **identical pipeline** — WebSocket → utterance → RAG → response.

### Chunk Simulator

The **📡 Chunk Simulator** panel demonstrates utterance buffering:
- Click Chunk 1, 2, 3 in sequence
- Each click sends a partial chunk to the utterance buffer
- Only Chunk 3 (marked "final") triggers the RAG pipeline
- One utterance → one answer

### Demo Scenarios

The **🎯 Demo Scenarios** panel provides one-click demos:

| Scenario | Query | Expected State |
|---|---|---|
| **A** | *"Hello AVY"* | NO_RETRIEVE |
| **B** | *"Tell me about Samsung's..."* | WAIT |
| **C** | Multi-chunk: "Tell me about Samsung's" + "latest processor" | RETRIEVE (Early) |
| **D** | Compound: "Compare Exynos 2400 with Exynos 2200 and explain battery efficiency" | RETRIEVE (Multi-Intent) |
| **E** | Follow-up: Turn 1 → Turn 2 with "its" pronoun | RETRIEVE (Session) |
| **F** | Late detail: chunk 1 + qualifier chunk 2 | RETRIEVE |

All scenarios use the **production RAG pipeline** — no fake responses.

---

## Pipeline Trace

The **🔄 Pipeline Trace** panel shows all 10 stages with real measured durations:

```
01  Transcript Chunk         ✓  0.1 ms
02  Retrieval Controller     ✓  0.4 ms   → RETRIEVE (Early)
03  Query Refinement         ✓  1.2 ms
04  Intent Decomposition     ✓  82 ms    (only for compound queries)
05  Vector Retrieval         ✓  43 ms
06  RRF Evidence Fusion      ✓  1.8 ms
07  Two-Stage Reranking      ✓  61 ms
08  Grounding Engine         ✓  0.8 ms
09  Grounded Synthesis       ✓  —        (streaming)
10  Streaming Response       ✓  —
```

> All durations are **real wall-clock measurements** from the pipeline. Nothing is fabricated.

---

## Benchmark Results

Measured with `--provider mock` (deterministic, offline):

> [!IMPORTANT]
> **Mock Provider Benchmark (deterministic, reproducible, zero real-LLM-latency):**
> - Controller Decision Accuracy: **100%** (8/8 correct)
> - Mean Retrieval Recall@K: **100%**
> - Mean TTFT: **~15 ms** *(in-process mock — not real LLM latency)*
> - Groundedness (keyword overlap): C1: 41%, D1: 68%, E1: 59%, F1: 50%
>
> **With Ollama + qwen3:4b:** TTFT typically 300–2000 ms (hardware-dependent).
> Run `python scripts/evaluate.py --provider ollama` for real measurements.

```bash
python scripts/evaluate.py --provider mock
```

---

## Testing

```bash
pytest tests/ -v
# 41 tests — all passing
```

| Test File | Coverage |
|---|---|
| `test_controller.py` | WAIT / RETRIEVE / NO_RETRIEVE / early retrieval |
| `test_decomposition.py` | Multi-intent extraction, fallback parsing |
| `test_retrieval.py` | FAISS search, save/load |
| `test_fusion.py` | RRF scoring, deduplication |
| `test_reranker.py` | Cross-encoder + lexical reranking |
| `test_session.py` | Pronoun resolution, evidence pool persistence |
| `test_grounding.py` | Prompt synthesis, citation extraction, groundedness |
| `test_streaming.py` | SSE events, token streaming, timings |
| `test_telemetry.py` | Event logging, thread safety, secret scrubbing |
| `test_api.py` | REST endpoints, session CRUD, scenario streaming |
| `test_end_to_end.py` | Full pipeline across all 6 scenarios |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | FastAPI + Uvicorn |
| Real-time Transport | WebSocket (primary) + Server-Sent Events (fallback) |
| Voice Input (STT) | Web Speech API (browser-native: Chrome/Edge) |
| Voice Output (TTS) | Web SpeechSynthesis API (browser-native) |
| Vector Search | FAISS (faiss-cpu) |
| Embeddings | FastEmbed — BAAI/bge-small-en-v1.5 (ONNX local) |
| Reranker | FastEmbed — BAAI/bge-reranker-base + lexical fallback |
| Local LLM | Ollama (qwen3:4b, Llama, Mistral, etc.) |
| Cloud LLM | OpenAI-compatible endpoint (GPT-4o, vLLM, DeepSeek) |
| Testing | Pytest + Pytest-Asyncio |
| Linting | Ruff |

---

## Limitations & Future Work

- **Browser STT**: Web Speech API requires Chrome or Edge. Firefox does not support it. Text input always works as fallback.
- **Offline TTS**: SpeechSynthesis is browser-native with varying voice quality. A future version could integrate edge-tts or piper-tts for higher quality offline speech.
- **Corpus scope**: The demo corpus covers Samsung semiconductors, Galaxy devices, and Exynos processors. Extend by adding markdown files to `corpus/` and re-running `build_index.py`.
- **Groundedness metric**: Uses keyword overlap (honest and reproducible). Semantic entailment (NLI-based) would be more precise.
- **Early retrieval lead time**: Measured as actual pipeline wall-clock elapsed time at the trigger point. On local hardware this is typically 1–20 ms.
- **Audio VAD**: Future versions will couple with WebRTC or Silero VAD for sub-50ms silence detection.

---

## License

MIT — see [LICENSE](LICENSE).
