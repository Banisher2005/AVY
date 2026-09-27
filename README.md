# AVY — Streaming Live RAG Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Theme](https://img.shields.io/badge/Samsung%20PRISM%203rd%20GenAI-Theme%2004%3A%20Streaming%20Live%20RAG-brightgreen)](https://github.com/Banisher2005/AVY)
[![Architecture](https://img.shields.io/badge/Base-AVI%20Architecture-purple)](https://github.com/Banisher2005/avi)

> **A real-time, low-latency conversational RAG system designed for speech-driven interactions where utterances arrive incrementally, contain multi-intent compound questions, and require immediate, grounded streaming synthesis.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Core Problem](#2-the-core-problem)
3. [Why Conventional RAG Fails for Spoken Conversations](#3-why-conventional-rag-fails-for-spoken-conversations)
4. [Samsung PRISM Theme 04 Alignment](#4-samsung-prism-theme-04-alignment)
5. [System Architecture](#5-system-architecture)
6. [Architectural Heritage: AVI → AVY](#6-architectural-heritage-avi--avy)
7. [Three-State Retrieval Controller](#7-three-state-retrieval-controller)
8. [Early Retrieval Execution](#8-early-retrieval-execution)
9. [Multi-Intent Query Decomposition](#9-multi-intent-query-decomposition)
10. [Multi-Query Vector Retrieval](#10-multi-query-vector-retrieval)
11. [Evidence Fusion & Deduplication (RRF)](#11-evidence-fusion--deduplication-rrf)
12. [Two-Stage Reranking](#12-two-stage-reranking)
13. [Session Context & Conversational Refinement](#13-session-context--conversational-refinement)
14. [Strict Grounding & Factual Verification](#14-strict-grounding--factual-verification)
15. [Deterministic Citations & Provenance](#15-deterministic-citations--provenance)
16. [Streaming Synthesis & Latency Breakdown](#16-streaming-synthesis--latency-breakdown)
17. [Structured Telemetry](#17-structured-telemetry)
18. [Evaluation Harness & Metrics](#18-evaluation-harness--metrics)
19. [Technology Stack](#19-technology-stack)
20. [Installation & Setup](#20-installation--setup)
21. [Configuration](#21-configuration)
22. [Running AVY](#22-running-avy)
23. [Testing Suite](#23-testing-suite)
24. [Demo Scenarios](#24-demo-scenarios)
25. [Limitations & Future Roadmap](#25-limitations--future-roadmap)

---

## 1. Overview

**AVY** (pronounced */ˈeɪ.vi/*) is an intelligent Streaming Live Retrieval-Augmented Generation (RAG) assistant created for the **Samsung PRISM GenAI Hackathon (3rd Edition) — Theme 04: Streaming Live RAG**.

Unlike traditional RAG systems that passively await clean, complete search queries, AVY processes live streaming speech-to-text transcripts incrementally. It answers three questions in real time:
- *Should I retrieve right now?*
- *Should I wait for more spoken context?*
- *Or should I not retrieve at all?*

When retrieval is warranted, AVY decomposes compound utterances, executes parallel vector searches, fuses and deduplicates evidence using **Reciprocal Rank Fusion (RRF)**, re-ranks candidates with cross-attention, incorporates conversational session memory, and streams answers with traceable citations `[1]`, `[2]`.

---

## 2. The Core Problem

Human speech is inherently messy. When speaking to a voice assistant, users do not formulate pristine database search queries. Spoken transcripts feature:
- **Incremental delivery**: Words arrive in partial chunks every 200–500 ms.
- **Incomplete grammar**: Users pause mid-sentence (*"Tell me about Samsung's..."*).
- **Compound multi-intent requests**: Users combine multiple queries into one sentence (*"Compare Exynos 2400 with Exynos 2200 and explain its battery power efficiency"*).
- **Conversational filler**: Greetings (*"Hello AVY"*), gratitude (*"Thanks, that helps"*), and filler phrases (*"Can you tell me about..."*).
- **Late qualifications & follow-ups**: Pronouns and late additions (*"Now compare its display with S23"*).

---

## 3. Why Conventional RAG Fails for Spoken Conversations

| Conventional RAG | AVY Streaming Live RAG |
|---|---|
| Waits for complete utterance before starting pipeline (High TTFT) | **Early Retrieval**: Triggers vector search before final transcript chunk finishes |
| Executes vector search on raw conversational query with filler words | **Decomposes** compound speech into atomic search-ready queries |
| Retrieves on greetings and conversational acknowledgments | **3-State Controller** gates retrieval: `WAIT` / `RETRIEVE` / `NO_RETRIEVE` |
| Discards past conversation evidence on follow-up questions | **Session Evidence Pool**: Fuses prior evidence with newly retrieved facts |
| Opaque sources with unverified citations | **Deterministic Traceability**: Every statement cited with `[1]` linked to doc ID and chunk |
| Monolithic slow generation block | **Low-latency SSE streaming** with exact TTFT phase measurements |

---

## 4. Samsung PRISM Theme 04 Alignment

AVY directly implements all requirements mandated by Theme 04:
1. **Streaming Audio/Transcript Input**: Seamlessly ingests incremental speech chunks.
2. **Retrieval Gating**: 3-state controller (`WAIT`, `RETRIEVE`, `NO_RETRIEVE`).
3. **Early Retrieval**: Begins vector search mid-utterance to shave hundreds of milliseconds off TTFT.
4. **Multi-Intent Decomposition**: Disaggregates compound spoken questions into independent subqueries.
5. **Multi-Query Retrieval**: Executes parallel searches preserving full query and chunk provenance.
6. **Evidence Fusion & Deduplication**: Implements **Reciprocal Rank Fusion (RRF)**.
7. **Two-Stage Reranking**: Re-scores top candidates with cross-attention or lexical ranking.
8. **Session Refinement**: Handles follow-up anaphora (*"its"*, *"their"*) and late details across turns.
9. **Grounded Synthesis**: Restricts LLM generation to cited evidence without hallucinations.
10. **Traceable Citations**: Produces indexed markers `[1]`, `[2]` mapped to exact document provenance.
11. **Streaming Synthesis**: Measures and exposes TTFT, retrieval latency, reranking latency, and total duration.
12. **Structured Telemetry**: Emits standard JSON/JSONL events with secret scrubbing.

---

## 5. System Architecture

```text
┌────────────────────────────────────────────────────────┐
│               Streaming Transcript Input               │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│            Three-State Retrieval Controller            │
│         [ WAIT ]   /   [ RETRIEVE ]   /   [ NO_RETRIEVE ]
└─────────────┬─────────────────┬───────────────────┬────┘
              │                 │                   │
       (needs context)   (sufficient info)    (conversational)
              │                 │                   │
              ▼                 │                   ▼
           [ WAIT ]             │          [ Direct Synthesis ]
      (buffer chunk)            │                   │
                                ▼                   ▼
             ┌────────────────────────────────────┐ │
             │    Session Context & Refinement    │ │
             └──────────────────┬─────────────────┘ │
                                │                   │
                                ▼                   │
             ┌────────────────────────────────────┐ │
             │  Multi-Intent Query Decomposition  │ │
             └──────────────────┬─────────────────┘ │
                                │                   │
                                ▼                   │
             ┌────────────────────────────────────┐ │
             │   Multi-Query Vector Retrieval     │ │
             │     (FAISS FlatIP / Dense Cosine)  │ │
             └──────────────────┬─────────────────┘ │
                                │                   │
                                ▼                   │
             ┌────────────────────────────────────┐ │
             │    Evidence Fusion (RRF) & Dedup   │ │
             └──────────────────┬─────────────────┘ │
                                │                   │
                                ▼                   │
             ┌────────────────────────────────────┐ │
             │       Two-Stage Reranking          │ │
             └──────────────────┬─────────────────┘ │
                                │                   │
                                ▼                   │
             ┌────────────────────────────────────┐ │
             │   Grounded Synthesis + Citations   │ │
             └──────────────────┬─────────────────┘ │
                                │                   │
                                ├───────────────────┘
                                ▼
             ┌────────────────────────────────────┐
             │    Live SSE Streaming Output       │
             │   [Token Chunks] + [Citations]     │
             └──────────────────┬─────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────┐
             │   Telemetry & Evaluation Metrics   │
             └────────────────────────────────────┘
```

---

## 6. Architectural Heritage: AVI → AVY

AVY is built upon the architectural principles established by [AVI](https://github.com/Banisher2005/avi):

| AVI Architectural Subsystem | AVY Adaptation & Evolution | Rationale for Reuse / Evolution |
|---|---|---|
| **Core Router** | **Retrieval Controller** | AVI's fast-path matching was adapted into a 3-state semantic gating controller (`WAIT`, `RETRIEVE`, `NO_RETRIEVE`). |
| **Context Subsystem** | **Session Context & Refinement** | Adapted AVI's snapshot pattern into multi-turn conversational session memory and anaphoric query contextualization. |
| **Provider Abstraction** | **Live Provider Subsystem** | Reused AVI's decoupled `BaseProvider`, `AgentRequest`, and streaming generator interfaces, adding Ollama, OpenAI-compatible, and offline Mock providers. |
| **Gateway & Streaming** | **SSE Live RAG Gateway** | Adapted AVI's protocol gateway into a real-time SSE streaming pipeline exposing token-by-token synthesis and latency gauges. |
| **Configuration** | **AVYConfig** | Maintained AVI's precedence hierarchy (CLI overrides > environment variables > config files > defaults). |

---

## 7. Three-State Retrieval Controller

The Retrieval Controller (`src/avy/retrieval/controller.py`) inspects incoming speech chunks:

1. **`WAIT`**: The transcript is syntactically or semantically incomplete.
   - *Examples*: `"Tell me about Samsung's..."`, `"Can you explain the"`, `"What is the"`.
   - *Action*: Buffers chunk, does not waste vector compute or hallucinate partial answers.
2. **`RETRIEVE`**: The transcript possesses sufficient semantic subject and predicate to formulate a clean query.
   - *Example*: `"Tell me about Samsung's latest semiconductor processor specifications."`
   - *Action*: Triggers decomposition and vector search.
3. **`NO_RETRIEVE`**: Conversational pleasantries, small talk, gratitude, or identity inquiries.
   - *Examples*: `"Hello AVY"`, `"Thanks, that's helpful"`, `"Who are you?"`.
   - *Action*: Routes directly to conversational LLM synthesis without corpus retrieval.

---

## 8. Early Retrieval Execution

Conventional voice systems wait until a user stops speaking before starting retrieval. AVY initiates vector search **as soon as an informational subject is recognized**:
```text
Chunk 1: "Tell me about Samsung's"                                 ──> WAIT
Chunk 2: "latest semiconductor processor"                         ──> RETRIEVE (EARLY!)
Chunk 3: "and compare battery efficiency with previous generation" ──> REFINE & FUSE
```
By starting retrieval at Chunk 2, vector candidates are already loaded by the time Chunk 3 arrives, significantly reducing **Time-to-First-Token (TTFT)**.

---

## 9. Multi-Intent Query Decomposition

Spoken queries frequently combine multiple questions into a single sentence. The `MultiIntentDecomposer` (`src/avy/retrieval/decomposition.py`) uses model reasoning to disaggregate requests:

```text
Input Utterance:
"Compare Samsung's latest processor with the previous generation and explain the battery efficiency difference."

Decomposed Queries:
[1] "Samsung latest processor specifications"
[2] "Samsung previous generation processor specifications"
[3] "battery efficiency comparison between the processors"
```

The decomposer includes robust JSON extraction, code-fence sanitization, bullet-point parsing, and fallback conjunction splitting to handle malformed LLM outputs.

---

## 10. Multi-Query Vector Retrieval

Each subquery independently retrieves candidates from the FAISS vector index (`src/avy/retrieval/multi_query.py`). Provenance metadata is preserved throughout the pipeline:
- `query_id`: Unique identifier of the decomposed subquery.
- `subquery`: Subquery text string.
- `document_id`: Corpus document origin.
- `source`: Source filename.
- `chunk_id`: Unique chunk identifier.
- `rank`: Subquery candidate rank ($1 \dots K$).
- `score`: Inner-product cosine similarity.

---

## 11. Evidence Fusion & Deduplication (RRF)

When multiple queries return overlapping chunks, `EvidenceFusion` (`src/avy/retrieval/fusion.py`) combines candidate lists using **Reciprocal Rank Fusion (RRF)**:

$$RRF(d) = \sum_{q \in Q} \frac{1}{k + r_q(d)}$$

where $k = 60$ and $r_q(d)$ is the 1-based rank of chunk $d$ in subquery $q$.
- **Automatic Boosting**: Chunks answering multiple subqueries accumulate reciprocal ranks, naturally elevating cross-cutting evidence to the top.
- **Deduplication**: Identical chunks across queries are merged into a single candidate with normalized scores.

---

## 12. Two-Stage Reranking

The top fused evidence passes through `EvidenceReranker` (`src/avy/retrieval/reranker.py`):
- **Cross-Attention Mode**: Uses `BAAI/bge-reranker-base` via ONNX runtime to score query-document pairs.
- **Lexical BM25 Fallback**: If offline or running in lightweight mode, computes term-overlap cross-scores.
- **Configurable**: Can be toggled on/off (`AVY_RERANKER_ENABLED=false`) to test baseline RRF latency.
- **Latency Recorded**: Execution duration in milliseconds is measured and reported.

---

## 13. Session Context & Conversational Refinement

Human conversation relies on anaphora (*"it"*, *"its"*, *"their"*). `SessionManager` (`src/avy/context/session.py`) maintains conversational continuity:
- **Turn 1**:
  - *User*: *"Tell me about the Samsung Galaxy S24 Ultra display."*
  - *AVY*: Returns grounded specs on the 6.8-inch QHD+ 1-120Hz display [1].
- **Turn 2**:
  - *User*: *"Now compare its battery with the S23."*
  - *AVY*: Detects pronoun *"its"*, contextualizes the query to *"Samsung Galaxy S24 Ultra battery comparison with S23"*, fuses Turn 1 evidence with Turn 2 retrieval, and streams a comparative answer.

---

## 14. Strict Grounding & Factual Verification

To prevent hallucinations, `GroundingEngine` (`src/avy/retrieval/grounding.py`) provides:
1. **Strict Prompting**: Directs the LLM to synthesize responses exclusively from the provided numbered evidence blocks.
2. **Groundedness Scoring**: Computes keyword and token overlap between the synthesized text and cited evidence chunks.

---

## 15. Deterministic Citations & Provenance

Every factual claim in an answer is tied to indexed source citations:
```text
The Exynos 2400 features a 10-core CPU architecture and AMD Xclipse 940 GPU [1].
The Galaxy S24 Ultra incorporates a 5,000 mAh battery with 45W charging [2].
```
The citation engine maps each `[i]` marker back to:
- Document ID
- Source filename
- Exact chunk ID
- Text snippet

---

## 16. Streaming Synthesis & Latency Breakdown

AVY streams tokens in real time and reports exact latency timestamps:
- **TTFT (Time-to-First-Token)**: Latency from utterance receipt to the first generated token.
- **Decomposition Latency**: Time to decompose compound questions.
- **Retrieval & Fusion Latency**: Time to execute vector searches and compute RRF.
- **Reranking Latency**: Time to re-score evidence.
- **Synthesis Latency**: Total token generation duration.
- **Total Pipeline Latency**: End-to-end processing time.

---

## 17. Structured Telemetry

All pipeline transitions emit structured JSONL events to `data/telemetry/events.jsonl` (scrubbing sensitive keys):
- `input.chunk`
- `retrieval.wait`
- `retrieval.triggered`
- `retrieval.skipped`
- `query.decomposed`
- `retrieval.started`
- `retrieval.completed`
- `fusion.completed`
- `rerank.completed`
- `synthesis.started`
- `synthesis.first_token`
- `synthesis.completed`
- `citation.generated`
- `pipeline.error`

---

## 18. Evaluation Harness & Metrics

AVY includes an automated benchmark runner (`scripts/evaluate.py` / `avy eval`) evaluating:
- **Controller Decision Accuracy (%)**: Correct categorization into `WAIT`, `RETRIEVE`, `NO_RETRIEVE`.
- **Retrieval Recall@K**: Proportion of ground-truth document chunks retrieved.
- **Answer Groundedness Score**: Factual support ratio in synthesized text (keyword overlap between answer and retrieved evidence).
- **Time-to-First-Token (TTFT)**: Measured from pipeline entry to first streamed token in milliseconds.
- **Cost per Turn ($)**: Token expenditure tracking (0.00 USD for local Ollama).

> [!IMPORTANT]
> **Benchmark Methodology & Provider Transparency**
>
> The default benchmark (`scripts/evaluate.py --provider mock`) uses the **deterministic MockProvider** (zero-latency, offline, CPU-only). This ensures reproducible results in CI and on machines without Ollama.
>
> **Mock provider benchmark results (deterministic, reproducible):**
> - Controller Decision Accuracy: **100%** (8/8 test cases correct)
> - Mean Retrieval Recall@K: **100%** (all expected doc IDs retrieved)
> - Mean TTFT: **~14 ms** *(mock provider — not representative of real LLM latency)*
> - Groundedness scores: C1 Early Retrieval ~41%, D1 Multi-Intent ~68%, E1 Session ~59%, F1 Late Detail ~50%
>   *(keyword overlap metric between mock LLM answer and retrieved evidence chunks)*
>
> **With Ollama + qwen3:4b (real provider):**
> - TTFT is typically **300–2000 ms** depending on hardware and model load.
> - Run with `scripts/evaluate.py --provider ollama` to measure real provider latency.
>
> Groundedness is calculated as the ratio of 4+ character tokens in the generated answer that also appear in the retrieved evidence text. It measures grounding alignment, not factual correctness.
>
> **Early retrieval lead time** is measured as the actual pipeline wall-clock time elapsed between pipeline start and retrieval trigger. On early retrieval queries this typically ranges from 1–20ms (local hardware), representing the real overlap window saved vs. waiting for a complete utterance.


## 19. Technology Stack

- **Language**: Python 3.10+ (tested on Python 3.12)
- **Vector Search**: FAISS (`faiss-cpu`) with cosine similarity normalization
- **Embeddings**: `BAAI/bge-small-en-v1.5` via ONNX FastEmbed + pure NumPy offline fallback
- **Reranker**: `BAAI/bge-reranker-base` via ONNX FastEmbed + Lexical cross-scorer
- **Local LLM**: Ollama (`qwen3:4b` or any local model)
- **Cloud/API LLM**: OpenAI-compatible endpoint support (GPT-4o, vLLM, DeepSeek)
- **Web Server & SSE**: FastAPI, Uvicorn, Server-Sent Events
- **Testing & Quality**: Pytest, Pytest-Asyncio, Ruff

---

## 20. Installation & Setup

### Prerequisites
- Linux / macOS / WSL
- Python 3.10 or higher
- (Optional) [Ollama](https://ollama.ai) installed with `qwen3:4b`

### Setup Commands
```bash
# Clone the repository
git clone https://github.com/Banisher2005/AVY.git
cd AVY

# Create virtual environment and install dependencies
make install
# Or manually with uv:
# uv venv .venv --python 3.12
# uv pip install -e . --python .venv/bin/python
# uv pip install pytest pytest-asyncio ruff --python .venv/bin/python

# Activate virtual environment
source .venv/bin/activate
```

---

## 21. Configuration

Copy the example environment configuration:
```bash
cp .env.example .env
```
Key configuration parameters:
```ini
# LLM Provider ("ollama", "openai", "mock")
AVY_PROVIDER=ollama
AVY_MODEL=qwen3:4b
AVY_OLLAMA_HOST=http://127.0.0.1:11434

# Embeddings ("fastembed", "local_dense")
AVY_EMBEDDING_PROVIDER=fastembed
AVY_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Reranker
AVY_RERANKER_ENABLED=true
AVY_RERANKER_MODEL=BAAI/bge-reranker-base

# Vector Store Parameters
AVY_CORPUS_DIR=corpus
AVY_INDEX_PATH=data/index/avy_faiss.index
AVY_METADATA_PATH=data/index/avy_metadata.json
AVY_TOP_K=5
AVY_RRF_K=60
```

---

## 22. Running AVY

### 1. Ingest Corpus & Build FAISS Index
```bash
python scripts/build_index.py
# Or: avy index
```

### 2. Launch the Web UI & SSE Server
```bash
python scripts/demo_server.py
# Or: avy serve --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to access the live dashboard.

### 3. Run Interactive CLI REPL
```bash
avy repl
```

### 4. Single-Query Streaming CLI
```bash
avy stream "Compare Exynos 2400 with previous generation and explain battery efficiency"
```

---

## 23. Production Web Dashboard & UI Experience

AVY's centerpiece is a **dark, research-lab command center dashboard** designed specifically for live hackathon demonstration:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  [AVY]  AVY Streaming Live RAG Assistant   [• LIVE CONNECTED]  [MODEL: qwen3:4b]  [CORPUS: 32]   │
├──────────────────────────────┬────────────────────────────────────┬──────────────────────────────┤
│ 🎙️ SPEECH & SCENARIOS        │ 💬 LIVE RAG CONVERSATION           │ 🔄 TECHNICAL DEEP-DIVE       │
│                              │                                    │                              │
│ 1-Click Benchmark Matrix:    │ [User]                             │ Tabs:                        │
│ [A: Early Retrieval]         │ "Compare Samsung Exynos 2400..."   │ ┌────────┬──────┬─────┬────┐ │
│ [B: Fast-Path NO_RETRIEVE]   │                                    │ │Trace 10│Decomp│ RRF │KPIs│ │
│ [C: Incomplete WAIT]         │ 🔗 Refined: "Galaxy S24 Ultra..."  │ └────────┴──────┴─────┴────┘ │
│ [D: Compound Multi-Intent]   │                                    │                              │
│ [E: Anaphora Refinement]     │ [AVY Assistant]                    │ 10-Stage Pipeline Trace:     │
│ [F: Comparative Deep-Dive]   │ "The Exynos 2400 features a        │ 01 Transcript Chunk      [✓] │
│                              │  10-core CPU with AMD Xclipse      │ 02 Retrieval Controller  [✓] │
│ Audio Chunk Simulator:       │  940 GPU [1]. In terms of battery  │ 03 Query Refinement      [✓] │
│ [1] "Tell me about..."       │  efficiency, it achieves 22%       │ 04 Intent Decomposition  [✓] │
│ [2] "latest processor..."    │  longer battery life [2]."         │ 05 Vector Retrieval      [✓] │
│ [3] "and compare battery..." │                                    │ 06 RRF Evidence Fusion   [✓] │
│                              │ Grounded Sources Tray:             │ 07 Two-Stage Reranking   [✓] │
│ Live Transcript Feed:        │ [[1] exynos_2400.md]               │ 08 Grounding Engine      [✓] │
│ "Chunk 1: Tell me..."        │ [[2] battery_power.md]             │ 09 Grounded Synthesis    [✓] │
│                              │                                    │ 10 Streaming Response    [✓] │
│ 3-State Controller Indicator:│ (Click [1] to open Evidence        │                              │
│ [ WAIT ] [• RETRIEVE] [ NO ] │  Inspector Drawer/Modal)           │ Latency KPIs:                │
│                              │                                    │ TTFT: 142ms  Total: 840ms    │
│ ⚡ Early Retrieval Banner:   │                                    │ Lead Time: actual measured   │
│ [Speech: ======>           ] │                                    │ pipeline elapsed time (ms)   │
│ [Search: ========> (Δ ms)]  │                                    │                              │
└──────────────────────────────┴────────────────────────────────────┴──────────────────────────────┘
```

### Key UI Features:
1. **Interactive Evidence Inspector (Modal / Slide-Over)**:
   Clicking any `[1]`, `[2]` citation marker opens an inspection modal displaying:
   - Document Title & Source Markdown file
   - Chunk ID & Chunk Index
   - Initial Vector Score, RRF Score, and Cross-Encoder Rerank Score
   - Matched Subquery & Keyword Highlights
   - Exact supporting evidence passage
2. **10-Stage Real-Time Pipeline Trace**:
   Visually monitors every phase of the pipeline in real-time (`idle`, `running`, `completed`, `skipped`) with millisecond timestamps.
3. **Score Comparator (Vector → RRF → Rerank)**:
   Explores how candidates are re-ordered across the 3 stages.
4. **Early Retrieval Lead Time Banner**:
   Shows the actual measured pipeline elapsed time when retrieval is triggered early — representing the real overlap window gained vs. waiting for a complete utterance.

---

## 24. REST & SSE API Reference

AVY exposes a clean, high-performance API suitable for production integration:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health, loaded model, vector dimension, chunk count, and configuration |
| `GET` | `/api/scenarios` | List benchmark scenarios (A through F) with expected controller states |
| `POST` | `/api/scenarios/run` | Stream simulated audio chunks for a scenario via Server-Sent Events (SSE) |
| `POST` | `/api/stream/rag` | Stream live RAG pipeline events and tokens token-by-token via SSE |
| `POST` | `/api/session` | Create or reset an active conversational session |
| `GET` | `/api/session/{id}` | Inspect multi-turn session history, active intent, and evidence pool |
| `DELETE` | `/api/session/{id}` | Clear session memory |
| `GET` | `/api/sessions` | List all active sessions |
| `GET` | `/api/telemetry/{id}` | Retrieve structured telemetry events for a session |
| `GET` | `/api/citations/{id}` | Retrieve deduplicated citations generated in a session |

---

## 25. Testing Suite

Run the complete 37-test automated test suite:
```bash
pytest tests/ -v
# Or: make test
```

Coverage includes:
- `tests/test_controller.py`: WAIT, RETRIEVE, NO_RETRIEVE state classification & early retrieval
- `tests/test_decomposition.py`: Multi-intent JSON extraction, markdown fences, fallback splitting
- `tests/test_retrieval.py`: FAISS vector search, dimension matching, save/load persistence
- `tests/test_fusion.py`: Reciprocal Rank Fusion (RRF), score normalization, multi-query deduplication
- `tests/test_reranker.py`: Cross-encoder & lexical reranking, score re-ordering
- `tests/test_session.py`: Pronoun/anaphora resolution, multi-turn evidence pool caching
- `tests/test_grounding.py`: Prompt synthesis, citation extraction, groundedness scoring
- `tests/test_streaming.py`: SSE event generator, token streaming, timing measurements
- `tests/test_telemetry.py`: Event logging, thread-safety, credential scrubbing
- `tests/test_api.py`: REST endpoints, session CRUD, telemetry querying, citations lookup
- `tests/test_end_to_end.py`: End-to-end execution of all 6 hackathon scenarios

---

## 26. Demo Scenarios

AVY provides built-in demo scenarios matching the hackathon specifications:

| Scenario | Input Query / Transcript Chunks | Expected Controller State | Key Pipeline Behavior |
|---|---|:---:|---|
| **Scenario A** | *"Hello AVY, how are you today?"* | `NO_RETRIEVE` | Bypasses vector retrieval; streams direct conversational greeting. |
| **Scenario B** | *"Tell me about Samsung's..."* | `WAIT` | Detects incomplete grammar/trailing prepositions; buffers input. |
| **Scenario C** | Chunk 1: *"Tell me about Samsung's"*<br>Chunk 2: *"latest semiconductor processor"* | `RETRIEVE` | **Early Retrieval**: triggers search on Chunk 2 before sentence completes. |
| **Scenario D** | *"Compare Samsung's latest processor with the previous generation and explain the battery efficiency difference."* | `RETRIEVE` | Decomposes into 3 subqueries; executes multi-query search and RRF. |
| **Scenario E** | Turn 1: *"Tell me about S24 Ultra display"*<br>Turn 2: *"Now compare its battery with S23"* | `RETRIEVE` | **Session Refinement**: resolves *"its"* to S24 Ultra and fuses past evidence. |
| **Scenario F** | Chunk 1: *"Tell me about Exynos 2400"*<br>Chunk 2: *"and explain its AMD Xclipse GPU"* | `RETRIEVE` | Late detail qualification updates query scope and refines search. |

To run all scenarios in the terminal:
```bash
python scripts/run_demo.py
# Or run a specific scenario:
# python scripts/run_demo.py --scenario C
```

---

## 27. Limitations & Future Roadmap

- **Audio VAD Integration**: The current demonstration ingests text transcript chunks. Future iterations will couple directly with client-side WebRTC / Silero VAD for sub-50ms acoustic voice activity detection.
- **Dynamic Speculative Retrieval**: Exploring speculative multi-path decoding where top-2 hypotheses from an ASR acoustic model trigger dual parallel vector searches.
- **Multimodal Visual Evidence**: Extending the FAISS index to embed Samsung circuit schematics and hardware architectural block diagrams using CLIP/SigLIP.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
