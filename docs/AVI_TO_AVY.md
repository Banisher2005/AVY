# Architectural Evolution: From AVI to AVY

## Context & Project Background
**AVY** is designed as a direct architectural evolution and adaptation of **AVI** ([github.com/Banisher2005/avi](https://github.com/Banisher2005/avi)), engineered specifically for the **Samsung PRISM GenAI Hackathon (3rd Edition) — Theme 04: Streaming Live RAG**.

Rather than copying AVI blindly, AVY carefully studies AVI's core strengths, isolates its highest-value architectural patterns, and re-architects them to power a streaming live conversational RAG system.

---

## High-Level Architectural Mapping

```text
AVI Subsystem                  AVY Adaptation & Evolution
─────────────                  ──────────────────────────
Core Router            ───►    Three-State Retrieval Controller (WAIT / RETRIEVE / NO_RETRIEVE)
                               + Early Retrieval Detector

Context Subsystem      ───►    Session Context, Turn History & Anaphoric Refinement
                               + Multi-Turn Evidence Pool Persistence

Provider Abstraction   ───►    Unified LLM Interface with Streaming Generators
                               + Multi-Intent Query Decomposer & Grounded Synthesizer

Gateway & Protocol     ───►    Real-time Server-Sent Events (SSE) Streaming Gateway
                               + Latency Telemetry (TTFT, Rerank, Retrieval, Synthesis)

Config Engine          ───►    AVYConfig (Hierarchy: CLI overrides > Env > File > Defaults)

NEW Subsystems in AVY:
─────────────────────
• Dense Embeddings (FastEmbed BGE-small + NumPy Subword Fallback)
• FAISS Vector Store (IndexFlatIP with Cosine Normalization)
• Multi-Intent Query Decomposition Engine
• Multi-Query Parallel Vector Retriever
• Evidence Fusion with Reciprocal Rank Fusion (RRF) & Deduplication
• Two-Stage Cross-Encoder Reranker (BGE-reranker-base)
• Grounding Verification & Deterministic Citation Engine
• Structured JSONL Telemetry Logger (with Secret Scrubbing)
• Automated Evaluation Benchmark Harness
• Interactive Real-Time Web Dashboard (SSE)
```

---

## Detailed Subsystem Comparison

### 1. Router ──► Retrieval Controller

#### In AVI
AVI's router (`src/avi/core/router.py`) used fast-path regex and AST expressions to intercept local terminal queries (e.g. `pwd`, `git branch`, disk space, safe calculator) in < 10 ms without calling an LLM. Requests not matching fast-paths were forwarded to the model provider.

#### In AVY
In streaming voice conversations, the critical gating problem is deciding *when* to search a knowledge base. AVY adapts the fast-path routing philosophy into a **Three-State Semantic Retrieval Controller** (`src/avy/retrieval/controller.py`):
- **`NO_RETRIEVE` Fast Path**: Intercepts greetings (*"Hello AVY"*), gratitude (*"Thanks"*), and system questions, streaming conversational replies without touching the vector store.
- **`WAIT` Gating**: Identifies incomplete sentences, trailing prepositions, and grammatical starters, buffering chunks and preventing premature vector searches.
- **`RETRIEVE` & Early Retrieval**: Detects when enough semantic information exists to initiate vector search mid-utterance, shaving significant latency off the final response.

---

### 2. Context Subsystem ──► Session Refinement

#### In AVI
AVI collected operating system snapshots (`TerminalContext`, `GitContext`, `PreviousCommandContext`) to provide environmental context to terminal generation prompts.

#### In AVY
AVY transforms the context abstraction into a **Conversational Session Engine** (`src/avy/context/session.py`):
- Tracks `session_id`, turns, past intents, subqueries, retrieved evidence, and synthesized answers.
- **Anaphoric Resolution**: Resolves pronouns (*"it"*, *"its"*, *"their"*) against preceding turns.
- **Late Detail Fusing**: Updates search queries when a user adds qualifying details (*"Now compare its battery with S23"*).
- **Session Evidence Pool**: Retains past retrieved chunks so multi-turn comparisons do not lose previously retrieved context.

---

### 3. Provider Abstraction ──► Decomposition & Grounded Synthesis

#### In AVI
AVI established an abstract `BaseProvider` and `ProviderRegistry`, supporting local Ollama models (`qwen3:4b`), Antigravity runner, and standard request/response dataclasses (`AgentRequest`, `AgentResponse`, `ResponseMetrics`).

#### In AVY
AVY preserves this decoupled design:
- Core AVY code depends only on `BaseProvider`.
- Integrates `OllamaProvider`, `OpenAICompatibleProvider`, and a deterministic `MockProvider`.
- Extends the provider's role to drive two specialized pipelines:
  1. **Multi-Intent Query Decomposition**: Reasoning over compound speech to produce clean search subqueries.
  2. **Grounded Synthesis**: Generating structured text constrained to numbered evidence blocks with traceable citations `[1]`, `[2]`.

---

### 4. Gateway & Streaming ──► SSE Live RAG Gateway

#### In AVI
AVI's `GatewayCore` handled protocol bridging (MCP, JSON-RPC, stdio/tcp) and confirmation tokens for dangerous terminal commands.

#### In AVY
Theme 04 requires real-time streaming to the user interface. AVY adapts the gateway concept into `LiveRAGGateway` and a **FastAPI SSE Streaming Gateway** (`src/avy/ui/server.py`):
- Accepts incremental transcript chunks via HTTP POST.
- Emits real-time Server-Sent Events (`controller_decision`, `decomposition`, `evidence`, `token`, `citations`, `timings`).
- Exposes detailed timing metrics: TTFT, decomposition latency, retrieval latency, reranking latency, synthesis latency, and total latency.

---

## Components Purposely Removed from AVI

To keep AVY lean, focused, and strictly compliant with Theme 04, unrelated AVI components were deliberately excluded:
- ❌ System command execution (`CommandExecutor`, `subprocess` invocation).
- ❌ Destructive command safety policies (`SafetyEngine`, risk confirmation tokens).
- ❌ Local desktop tools (`git.status`, `filesystem.list`, `system.processes`).
- ❌ GTK4 GUI popup desktop daemon and X11/Wayland global hotkey listeners.
- ❌ Browser automation CDP controllers.

By eliminating terminal execution code, AVY remains a pure, reproducible, and secure conversational RAG engine.
