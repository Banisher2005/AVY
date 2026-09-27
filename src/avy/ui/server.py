"""AVY FastAPI server — voice-first Streaming Live RAG.

Architecture:
    WebSocket /ws/{session_id}     — primary channel (voice + pipeline events)
    POST /api/query                — REST fallback (SSE streaming)
    POST /api/scenarios/run        — demo scenario streaming
    GET  /api/health               — system health
    GET  /api/scenarios            — list available demo scenarios
    GET  /api/session/{id}         — session inspection
    DELETE /api/session/{id}       — clear session
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from avy.config import AVYConfig
from avy.evaluation.dataset import get_standard_eval_dataset
from avy.streaming.engine import StreamingLiveRAGEngine
from avy.streaming.utterance_manager import UtteranceManager

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Complete utterance (not a partial chunk) for REST fallback clients."""

    text: str
    session_id: str | None = None


class ChunkRequest(BaseModel):
    """Partial transcript chunk — accumulated server-side into a full utterance."""

    chunk: str
    session_id: str | None = None
    is_final: bool = False  # True = finalize utterance and run RAG


class ScenarioRequest(BaseModel):
    scenario_id: str
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    engine: StreamingLiveRAGEngine | None = None,
    config: AVYConfig | None = None,
) -> FastAPI:
    cfg = config or AVYConfig.load()
    live_engine = engine or StreamingLiveRAGEngine(config=cfg)
    utterance_mgr = UtteranceManager()

    app = FastAPI(
        title="AVY — Streaming Live RAG Assistant",
        description="Samsung PRISM GenAI Hackathon — Theme 04: Streaming Live RAG",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # -----------------------------------------------------------------------
    # Frontend
    # -----------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html_file = STATIC_DIR / "index.html"
        if html_file.is_file():
            return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>AVY — Streaming Live RAG</h1><p>Frontend not found.</p>")

    # -----------------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------------

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "provider": live_engine.provider.get_model_name(),
            "provider_type": live_engine.provider.__class__.__name__,
            "provider_available": live_engine.provider.is_available(),
            "indexed_chunks": len(live_engine.vectorstore._chunks),
            "vector_dimension": live_engine.vectorstore.dimension,
            "embeddings_model": live_engine.embeddings.model_name,
            "reranker_enabled": live_engine.config.reranker_enabled,
            "early_retrieval_enabled": live_engine.config.early_retrieval_enabled,
            "active_sessions": len(live_engine.session_manager._sessions),
        }

    # -----------------------------------------------------------------------
    # WebSocket — primary voice channel
    # -----------------------------------------------------------------------

    @app.websocket("/ws/{session_id}")
    async def voice_session(websocket: WebSocket, session_id: str) -> None:
        """Bidirectional voice session WebSocket.

        Client → Server message types:
            { "type": "utterance",  "text": "..." }      — complete utterance
            { "type": "chunk",      "chunk": "...",
              "is_final": false }                          — partial transcript chunk
            { "type": "cancel" }                          — cancel active pipeline
            { "type": "clear" }                           — reset session
            { "type": "ping" }                            — keepalive

        Server → Client events mirror process_transcript_stream() output:
            pipeline_stage, controller_decision, query_refinement,
            decomposition, evidence, token, citations, timings, error
        Plus meta events:
            { "type": "utterance.buffered",   "text": ... }
            { "type": "utterance.finalized",  "text": ... }
            { "type": "pipeline.started",     "utterance_id": ... }
            { "type": "pipeline.completed" }
            { "type": "session.cleared" }
            { "type": "pong" }
        """
        await websocket.accept()
        active_pipeline = False

        try:
            async for raw in websocket.iter_text():
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = message.get("type", "")

                # ── keepalive ──────────────────────────────────────────────
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                # ── clear session ──────────────────────────────────────────
                elif msg_type == "clear":
                    live_engine.session_manager.clear(session_id)
                    utterance_mgr.clear(session_id)
                    active_pipeline = False
                    await websocket.send_json({"type": "session.cleared", "session_id": session_id})

                # ── cancel current pipeline ────────────────────────────────
                elif msg_type == "cancel":
                    active_pipeline = False
                    await websocket.send_json({"type": "pipeline.cancelled"})

                # ── incremental chunk (from STT interim results) ───────────
                elif msg_type == "chunk":
                    chunk = message.get("chunk", "").strip()
                    is_final = message.get("is_final", False)

                    if chunk:
                        buf = utterance_mgr.add_chunk(session_id, chunk)
                        # Echo buffered state to UI
                        await websocket.send_json({
                            "type": "utterance.buffered",
                            "chunk": chunk,
                            "accumulated": buf.accumulated_text,
                            "chunk_count": buf.chunk_count,
                        })

                    if is_final:
                        text = utterance_mgr.finalize_utterance(session_id)
                        if text and not active_pipeline:
                            active_pipeline = True
                            utt_id = f"utt_{uuid.uuid4().hex[:8]}"
                            await websocket.send_json({
                                "type": "utterance.finalized",
                                "text": text,
                                "utterance_id": utt_id,
                            })
                            await _run_pipeline(websocket, live_engine, text, session_id, utt_id)
                            active_pipeline = False

                # ── complete utterance (text input or finalized STT) ───────
                elif msg_type == "utterance":
                    text = message.get("text", "").strip()
                    if not text:
                        continue
                    if active_pipeline:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Pipeline already running — please wait",
                        })
                        continue

                    active_pipeline = True
                    utt_id = f"utt_{uuid.uuid4().hex[:8]}"
                    await websocket.send_json({
                        "type": "utterance.finalized",
                        "text": text,
                        "utterance_id": utt_id,
                    })
                    await _run_pipeline(websocket, live_engine, text, session_id, utt_id)
                    active_pipeline = False

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # REST query — SSE fallback for non-WebSocket clients
    # -----------------------------------------------------------------------

    @app.post("/api/query")
    async def query_rag(req: QueryRequest) -> StreamingResponse:
        """Process a complete utterance and stream RAG events via SSE."""
        sid = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"

        def event_generator():
            for event in live_engine.process_transcript_stream(
                transcript_chunk=req.text,
                session_id=sid,
                is_final_chunk=True,
            ):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.post("/api/stream/rag")
    async def stream_rag_legacy(req: "LegacyQueryRequest") -> StreamingResponse:
        """Legacy SSE endpoint — kept for compatibility, maps to /api/query."""
        sid = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"
        text = req.accumulated or req.chunk

        def event_generator():
            for event in live_engine.process_transcript_stream(
                transcript_chunk=text,
                session_id=sid,
                is_final_chunk=req.is_final,
            ):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # -----------------------------------------------------------------------
    # Demo scenarios
    # -----------------------------------------------------------------------

    @app.get("/api/scenarios")
    async def list_scenarios() -> list[dict[str, Any]]:
        dataset = get_standard_eval_dataset()
        return [
            {
                "id": tc.test_id,
                "name": tc.scenario_name,
                "description": tc.description,
                "inputs": tc.inputs,
                "expected_state": tc.expected_controller_state.value,
                "is_multi_turn": tc.is_multi_turn,
            }
            for tc in dataset
        ]

    @app.post("/api/scenarios/run")
    async def run_scenario_stream(req: ScenarioRequest) -> StreamingResponse:
        """Stream a complete demo scenario through the production RAG pipeline."""
        dataset = get_standard_eval_dataset()
        selected = next((tc for tc in dataset if tc.test_id == req.scenario_id), None)
        if not selected:
            return JSONResponse(status_code=404, content={"error": "Scenario not found"})

        async def scenario_generator():
            sid = req.session_id or f"demo_{selected.test_id}_{uuid.uuid4().hex[:4]}"

            if selected.is_multi_turn:
                # Multi-turn: each input is a separate conversational turn
                for turn_idx, turn_text in enumerate(selected.inputs):
                    yield f"data: {json.dumps({'type': 'turn_start', 'turn': turn_idx + 1, 'text': turn_text})}\n\n"
                    await asyncio.sleep(0.2)
                    for event in live_engine.process_transcript_stream(
                        transcript_chunk=turn_text,
                        session_id=sid,
                        is_final_chunk=True,
                    ):
                        yield f"data: {json.dumps(event)}\n\n"
                        await asyncio.sleep(0.01)
                    await asyncio.sleep(0.4)
            else:
                # Single-turn scenario: ALL inputs are chunks of ONE utterance.
                # They are joined into ONE query → ONE RAG pipeline → ONE answer.
                full_text = " ".join(selected.inputs)
                yield f"data: {json.dumps({'type': 'utterance.finalized', 'text': full_text, 'chunk_count': len(selected.inputs)})}\n\n"
                await asyncio.sleep(0.1)
                for event in live_engine.process_transcript_stream(
                    transcript_chunk=full_text,
                    session_id=sid,
                    is_final_chunk=True,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
                    await asyncio.sleep(0.015)

        return StreamingResponse(
            scenario_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # -----------------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------------

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict[str, Any]]:
        return live_engine.session_manager.list_sessions()

    @app.get("/api/session/{session_id}")
    async def get_session(session_id: str) -> Any:
        if session_id not in live_engine.session_manager._sessions:
            return JSONResponse(status_code=404, content={"error": f"Session '{session_id}' not found"})
        sess = live_engine.session_manager.get_or_create(session_id)
        return {
            "session_id": sess.session_id,
            "created_at": sess.created_at,
            "active_intent": sess.active_intent,
            "turns_count": len(sess.turns),
            "turns": [t.to_dict() for t in sess.turns],
            "evidence_pool_size": len(sess.fused_evidence_pool),
        }

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str) -> dict[str, Any]:
        live_engine.session_manager.clear(session_id)
        utterance_mgr.clear(session_id)
        return {"session_id": session_id, "status": "cleared"}

    @app.get("/api/telemetry/{session_id}")
    async def get_session_telemetry(session_id: str, limit: int = 100) -> dict[str, Any]:
        events = live_engine.telemetry.get_events(session_id=session_id, limit=limit)
        return {"session_id": session_id, "count": len(events), "events": events}

    @app.get("/api/citations/{session_id}")
    async def get_session_citations(session_id: str) -> dict[str, Any]:
        citations = live_engine.session_manager.get_citations(session_id)
        return {"session_id": session_id, "count": len(citations), "citations": citations}

    return app


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _run_pipeline(
    websocket: WebSocket,
    engine: StreamingLiveRAGEngine,
    text: str,
    session_id: str,
    utterance_id: str,
) -> None:
    """Run the RAG pipeline and stream all events over the WebSocket."""
    await websocket.send_json({"type": "pipeline.started", "utterance_id": utterance_id})
    try:
        for event in engine.process_transcript_stream(
            transcript_chunk=text,
            session_id=session_id,
            is_final_chunk=True,
        ):
            await websocket.send_json(event)
            await asyncio.sleep(0)  # yield to event loop between events
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc), "stage": "pipeline"})
    finally:
        await websocket.send_json({"type": "pipeline.completed", "utterance_id": utterance_id})


# ---------------------------------------------------------------------------
# Legacy request model (kept for /api/stream/rag backward compatibility)
# ---------------------------------------------------------------------------


class LegacyQueryRequest(BaseModel):
    chunk: str = ""
    accumulated: str | None = None
    session_id: str | None = None
    is_final: bool = True
