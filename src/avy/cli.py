"""Command line interface for AVY Streaming Live RAG Assistant."""

import argparse
import sys

from avy import __version__
from avy.config import AVYConfig
from avy.evaluation.dataset import get_standard_eval_dataset
from avy.evaluation.harness import EvaluationHarness
from avy.retrieval.embeddings import get_embedding_engine
from avy.retrieval.ingest import CorpusIngestor
from avy.retrieval.vectorstore import FAISSVectorStore
from avy.streaming.engine import StreamingLiveRAGEngine
from avy.streaming.gateway import LiveRAGGateway


def cmd_ingest(args: argparse.Namespace, config: AVYConfig) -> int:
    """Ingest documents from corpus directory and print chunk statistics."""
    corpus_dir = args.corpus or config.corpus_dir
    print(f"Ingesting corpus documents from: {corpus_dir}")
    ingestor = CorpusIngestor()
    docs = ingestor.load_documents(corpus_dir)
    print(f"Loaded {len(docs)} documents:")
    for d in docs:
        print(f"  • {d.source}: {d.title} ({d.metadata.get('filesize', 0)} bytes)")

    chunks = ingestor.ingest_corpus(corpus_dir)
    print(f"\nGenerated {len(chunks)} text chunks.")
    return 0


def cmd_build_index(args: argparse.Namespace, config: AVYConfig) -> int:
    """Ingest corpus and build FAISS vector index."""
    corpus_dir = args.corpus or config.corpus_dir
    index_path = args.index or config.index_path
    meta_path = args.meta or config.metadata_path

    print(f"1. Ingesting documents from {corpus_dir}...")
    ingestor = CorpusIngestor()
    chunks = ingestor.ingest_corpus(corpus_dir)
    if not chunks:
        print("Error: No documents found in corpus directory.")
        return 1

    print(f"2. Generating dense embeddings for {len(chunks)} chunks via {config.embedding_provider}...")
    embeddings = get_embedding_engine(config)
    texts = [c.text for c in chunks]
    emb_matrix = embeddings.embed_documents(texts)

    print(f"3. Building FAISS index (dimension={embeddings.dimension})...")
    vstore = FAISSVectorStore(dimension=embeddings.dimension)
    vstore.add_chunks(chunks, emb_matrix)

    print(f"4. Saving index to {index_path} and metadata to {meta_path}...")
    vstore.save(index_path, meta_path)
    print("✅ Index built successfully!")
    return 0


def cmd_stream(args: argparse.Namespace, config: AVYConfig) -> int:
    """Stream a single query directly to stdout."""
    engine = StreamingLiveRAGEngine(config=config)
    query = " ".join(args.query)
    print(f"\nAVY Query: \"{query}\"\n")

    for event in engine.process_transcript_stream(transcript_chunk=query, is_final_chunk=True):
        etype = event.get("type")
        if etype == "controller_decision":
            state = event.get("state")
            reason = event.get("reason")
            is_early = " [EARLY]" if event.get("is_early_retrieval") else ""
            print(f"[Controller] State: {state}{is_early} ({reason})")
        elif etype == "decomposition":
            subs = [sq["subquery_text"] for sq in event.get("decomposition", {}).get("subqueries", [])]
            print(f"[Decomposition] Subqueries ({len(subs)}): {', '.join(subs)}")
        elif etype == "evidence":
            print(f"[Retrieval] Fused Evidence: {event.get('count', 0)} chunks retrieved and reranked")
        elif etype == "token":
            sys.stdout.write(event.get("token", ""))
            sys.stdout.flush()
        elif etype == "citations":
            cits = event.get("citations", [])
            if cits:
                print("\n\n[Citations]")
                for c in cits:
                    print(f"  [{c['index']}] {c['source']} ({c['chunk_id']}) — {c['title']}")
        elif etype == "timings":
            t = event.get("timings", {})
            print(f"\n[Timings] TTFT: {t.get('ttft_ms', 0):.0f} ms | Total: {t.get('total_ms', 0):.0f} ms\n")

    return 0


def cmd_repl(args: argparse.Namespace, config: AVYConfig) -> int:
    """Run interactive REPL with multi-turn session refinement."""
    import uuid

    engine = StreamingLiveRAGEngine(config=config)
    session_id = f"repl_{uuid.uuid4().hex[:8]}"

    print(f"AVY Interactive Live RAG Session (v{__version__})")
    print(f"Session: {session_id} • Model: {config.model} • Provider: {config.provider}")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    try:
        while True:
            try:
                line = input("AVY > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break

            for event in engine.process_transcript_stream(
                transcript_chunk=line,
                session_id=session_id,
                is_final_chunk=True,
            ):
                etype = event.get("type")
                if etype == "controller_decision":
                    print(f"[{event.get('state')}] {event.get('reason')}")
                elif etype == "token":
                    sys.stdout.write(event.get("token", ""))
                    sys.stdout.flush()
                elif etype == "citations":
                    cits = event.get("citations", [])
                    if cits:
                        print("\n[Citations: " + ", ".join(f"[{c['index']}] {c['source']}" for c in cits) + "]")
                elif etype == "timings":
                    t = event.get("timings", {})
                    print(f" [TTFT: {t.get('ttft_ms', 0):.0f}ms | Total: {t.get('total_ms', 0):.0f}ms]\n")
    except Exception as err:
        print(f"Error: {err}")
        return 1

    return 0


def cmd_eval(args: argparse.Namespace, config: AVYConfig) -> int:
    """Run automated benchmark evaluation across all Theme 04 scenarios."""
    print("Running AVY Streaming Live RAG Benchmark Evaluation...")
    harness = EvaluationHarness()
    report = harness.run_benchmark()
    summary_table = harness.format_summary_table(report)
    print("\n" + summary_table + "\n")
    return 0


def cmd_demo(args: argparse.Namespace, config: AVYConfig) -> int:
    """Run interactive CLI demo for all benchmark scenarios."""
    gateway = LiveRAGGateway(config=config)
    dataset = get_standard_eval_dataset()

    print("=" * 70)
    print("AVY — STREAMING LIVE RAG DEMONSTRATION")
    print("Samsung PRISM GenAI Hackathon (Theme 04)")
    print("=" * 70)

    for tc in dataset:
        if args.scenario and args.scenario.upper() not in tc.test_id:
            continue

        print(f"\n▶ Running: {tc.scenario_name}")
        print(f"  Description: {tc.description}")
        print(f"  Expected State: {tc.expected_controller_state.value}")

        if tc.is_multi_turn:
            for turn_idx, t_text in enumerate(tc.inputs, start=1):
                print(f"\n  [Turn {turn_idx}] User: \"{t_text}\"")
                res = gateway.execute_query(t_text, session_id=f"demo_{tc.test_id}")
                state = res.get("controller_decision", {}).get("state", "UNKNOWN")
                print(f"  [Controller] -> {state}")
                if res.get("decomposition"):
                    subs = [s["subquery_text"] for s in res["decomposition"].get("subqueries", [])]
                    print(f"  [Decomposed Queries] -> {subs}")
                print(f"  [Assistant Answer]\n  {res.get('answer')}")
        else:
            accumulated = ""
            for i, chunk in enumerate(tc.inputs, start=1):
                accumulated = f"{accumulated} {chunk}".strip()
                is_final = (i == len(tc.inputs))
                print(f"\n  [Audio Chunk {i}] \"{chunk}\" (Accumulated: \"{accumulated}\")")
                for event in gateway.engine.process_transcript_stream(
                    transcript_chunk=chunk,
                    session_id=f"demo_{tc.test_id}",
                    accumulated_transcript=accumulated,
                    is_final_chunk=is_final,
                ):
                    etype = event.get("type")
                    if etype == "controller_decision":
                        early = " (EARLY TRIGGER)" if event.get("is_early_retrieval") else ""
                        print(f"  [Controller Decision] -> {event.get('state')}{early}: {event.get('reason')}")
                    elif etype == "decomposition":
                        subs = [s["subquery_text"] for s in event.get("decomposition", {}).get("subqueries", [])]
                        print(f"  [Decomposed Queries] -> {subs}")
                    elif etype == "token":
                        sys.stdout.write(event.get("token", ""))
                        sys.stdout.flush()
            print()

        print("-" * 70)

    return 0


def cmd_serve(args: argparse.Namespace, config: AVYConfig) -> int:
    """Start FastAPI streaming server."""
    import uvicorn

    from avy.ui.server import create_app

    host = args.host or config.server_host
    port = int(args.port or config.server_port)

    print(f"Starting AVY Live RAG Server at http://{host}:{port}")
    app = create_app(config=config)
    uvicorn.run(app, host=host, port=port)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="avy",
        description="AVY — Streaming Live RAG Assistant for Samsung PRISM Hackathon (Theme 04)",
    )
    parser.add_argument("--version", action="version", version=f"avy {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Inspect and chunk corpus documents")
    p_ingest.add_argument("--corpus", default=None, help="Corpus directory path")

    # Index
    p_index = subparsers.add_parser("index", help="Build FAISS vector index from corpus")
    p_index.add_argument("--corpus", default=None, help="Corpus directory path")
    p_index.add_argument("--index", default=None, help="Index output path")
    p_index.add_argument("--meta", default=None, help="Metadata output path")

    # Stream
    p_stream = subparsers.add_parser("stream", help="Execute single streaming query")
    p_stream.add_argument("query", nargs="+", help="Natural language query")

    # REPL
    subparsers.add_parser("repl", help="Start interactive conversational session")

    # Eval
    subparsers.add_parser("eval", help="Run automated evaluation harness")

    # Demo
    p_demo = subparsers.add_parser("demo", help="Run scenario demonstration")
    p_demo.add_argument("--scenario", default=None, help="Specific scenario ID (e.g. A, B, C, D, E, F)")

    # Serve
    p_serve = subparsers.add_parser("serve", help="Start Web UI & SSE streaming server")
    p_serve.add_argument("--host", default=None, help="Bind host (default 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=None, help="Bind port (default 8000)")

    args = parser.parse_args()
    config = AVYConfig.load()

    if args.command == "ingest":
        sys.exit(cmd_ingest(args, config))
    elif args.command == "index":
        sys.exit(cmd_build_index(args, config))
    elif args.command == "stream":
        sys.exit(cmd_stream(args, config))
    elif args.command == "repl":
        sys.exit(cmd_repl(args, config))
    elif args.command == "eval":
        sys.exit(cmd_eval(args, config))
    elif args.command == "demo":
        sys.exit(cmd_demo(args, config))
    elif args.command == "serve":
        sys.exit(cmd_serve(args, config))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
