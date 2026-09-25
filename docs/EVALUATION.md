# AVY — Evaluation Harness & Benchmark Methodology

## 1. Overview
The **AVY Evaluation Harness** (`scripts/evaluate.py` or `avy eval`) provides reproducible, automated assessment of the Streaming Live RAG pipeline against the **Samsung PRISM GenAI Hackathon (Theme 04)** requirements.

## 2. Core Metrics Evaluated

1. **Controller Decision Accuracy (%)**:
   Evaluates whether the 3-state Retrieval Controller correctly decides `WAIT`, `RETRIEVE`, or `NO_RETRIEVE` across incremental speech chunks, conversational greetings, and incomplete phrases.
2. **Retrieval Recall@K**:
   Proportion of ground-truth reference documents retrieved in top-K candidate sets after Reciprocal Rank Fusion (RRF) and reranking.
3. **Answer Groundedness Score**:
   Measures factual token overlap between synthesized answer claims and retrieved source evidence.
4. **Time-to-First-Token (TTFT)**:
   Actual measured latency (in milliseconds) from receipt of utterance/chunk to emission of the very first token.
5. **Phase-by-Phase Latency Breakdown**:
   Tracks decomposition, retrieval, reranking, and synthesis durations using monotonic microsecond timers (`Stopwatch`).
6. **Cost per Turn ($ USD)**:
   Computes token expenditure per conversational turn (0.00 USD for local Ollama / quantified for cloud endpoints).

---

## 3. Evaluation Commands

### Automated Benchmark Execution
```bash
# Run benchmark through CLI
avy eval

# Or run standalone evaluation script
python scripts/evaluate.py
```

### Run Benchmark with Pytest
```bash
pytest tests/test_end_to_end.py -v
```

---

## 4. Benchmark Scenario Descriptions

- **Scenario A (NO_RETRIEVE)**:
  - *Input*: `"Hello AVY, how are you today?"` / `"Thanks, that's really helpful."`
  - *Expected State*: `NO_RETRIEVE`
  - *Behavior*: Direct conversational response without vector store overhead.
- **Scenario B (WAIT)**:
  - *Input*: `"Tell me about Samsung's..."` / `"Can you explain the"`
  - *Expected State*: `WAIT`
  - *Behavior*: Buffers incomplete speech chunks; awaits semantic predicate.
- **Scenario C (EARLY RETRIEVAL)**:
  - *Input*: Chunk 1: `"Tell me about Samsung's"` ──► Chunk 2: `"latest semiconductor processor"`
  - *Expected State*: `RETRIEVE`
  - *Behavior*: Initiates search on Chunk 2 mid-speech before final utterance concludes.
- **Scenario D (MULTI-INTENT)**:
  - *Input*: `"Compare Samsung's latest processor with the previous generation and explain the battery efficiency difference."`
  - *Expected State*: `RETRIEVE`
  - *Behavior*: Decomposes compound inquiry into independent subqueries, executes parallel searches, and fuses results via RRF.
- **Scenario E (SESSION REFINEMENT)**:
  - *Input*: Turn 1: `"Tell me about the Samsung Galaxy S24 Ultra display."` ──► Turn 2: `"Now compare its battery with the S23."`
  - *Expected State*: `RETRIEVE`
  - *Behavior*: Contextualizes pronoun *"its"* to S24 Ultra, updates query, and fuses Turn 1 evidence with Turn 2 results.
- **Scenario F (LATE DETAIL)**:
  - *Input*: Chunk 1: `"Tell me about the Exynos 2400"` ──► Chunk 2: `"and explain its AMD Xclipse GPU architecture."`
  - *Expected State*: `RETRIEVE`
  - *Behavior*: Dynamically refines search scope with newly arriving audio detail.
