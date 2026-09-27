"""FastAPI web server providing Server-Sent Events (SSE) streaming and REST endpoints."""

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from avy.config import AVYConfig
from avy.evaluation.dataset import get_standard_eval_dataset
from avy.streaming.engine import StreamingLiveRAGEngine

STATIC_DIR = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    chunk: str
    accumulated: str | None = None
    session_id: str | None = None
    is_final: bool = True


class CreateSessionRequest(BaseModel):
    session_id: str | None = None


class ScenarioRequest(BaseModel):
    scenario_id: str
    session_id: str | None = None


def create_app(engine: StreamingLiveRAGEngine | None = None, config: AVYConfig | None = None) -> FastAPI:
    cfg = config or AVYConfig.load()
    live_engine = engine or StreamingLiveRAGEngine(config=cfg)

    app = FastAPI(
        title="AVY — Streaming Live RAG Assistant",
        description="Samsung PRISM GenAI Hackathon (Theme 04: Streaming Live RAG)",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static assets
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html_file = STATIC_DIR / "index.html"
        if html_file.is_file():
            return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>AVY Streaming Live RAG Dashboard</h1>")

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
            "reranker_model": live_engine.config.reranker_model,
            "active_sessions": len(live_engine.session_manager._sessions),
            "config": {
                "top_k": live_engine.config.top_k,
                "rrf_k": live_engine.config.rrf_k,
                "similarity_threshold": live_engine.config.similarity_threshold,
                "early_retrieval_enabled": live_engine.config.early_retrieval_enabled,
                "telemetry_enabled": live_engine.config.telemetry_enabled,
            },
        }

    @app.post("/api/session")
    async def create_session(req: CreateSessionRequest | None = None) -> dict[str, Any]:
        sid = req.session_id if req and req.session_id else None
        sess = live_engine.session_manager.get_or_create(sid)
        return {
            "session_id": sess.session_id,
            "created_at": sess.created_at,
            "status": "active",
            "turns_count": len(sess.turns),
        }

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
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "timestamp": t.timestamp,
                    "user_utterance": t.user_utterance,
                    "controller_state": t.controller_state,
                    "decomposed_queries": t.decomposed_queries,
                    "assistant_response": t.assistant_response,
                    "citations": t.citations,
                }
                for t in sess.turns
            ],
            "evidence_pool_size": len(sess.fused_evidence_pool),
        }

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str) -> dict[str, Any]:
        live_engine.session_manager.clear(session_id)
        return {"session_id": session_id, "status": "cleared"}

    @app.get("/api/telemetry/{session_id}")
    async def get_session_telemetry(session_id: str, limit: int = 100) -> dict[str, Any]:
        events = live_engine.telemetry.get_events(session_id=session_id, limit=limit)
        return {
            "session_id": session_id,
            "count": len(events),
            "events": events,
        }

    @app.get("/api/citations/{session_id}")
    async def get_session_citations(session_id: str) -> dict[str, Any]:
        citations = live_engine.session_manager.get_citations(session_id)
        return {
            "session_id": session_id,
            "count": len(citations),
            "citations": citations,
        }

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

    @app.post("/api/stream/rag")
    async def stream_rag(req: QueryRequest) -> StreamingResponse:
        """Stream RAG pipeline execution via Server-Sent Events (SSE)."""

        def event_generator():
            for event in live_engine.process_transcript_stream(
                transcript_chunk=req.chunk,
                session_id=req.session_id,
                accumulated_transcript=req.accumulated,
                is_final_chunk=req.is_final,
            ):
                payload = json.dumps(event)
                yield f"data: {payload}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.post("/api/scenarios/run")
    async def run_scenario_stream(req: ScenarioRequest) -> StreamingResponse:
        """Stream simulated audio chunks for a benchmark scenario."""
        dataset = get_standard_eval_dataset()
        selected = next((tc for tc in dataset if tc.test_id == req.scenario_id), None)
        if not selected:
            return JSONResponse(status_code=404, content={"error": "Scenario not found"})

        async def scenario_generator():
            sid = req.session_id or f"demo_{selected.test_id}"
            if selected.is_multi_turn:
                # Run turn 1, then turn 2
                for turn_idx, turn_text in enumerate(selected.inputs):
                    yield f"data: {json.dumps({'type': 'turn_start', 'turn': turn_idx + 1, 'text': turn_text})}\n\n"
                    await asyncio.sleep(0.3)
                    for event in live_engine.process_transcript_stream(
                        transcript_chunk=turn_text,
                        session_id=sid,
                        is_final_chunk=True,
                    ):
                        yield f"data: {json.dumps(event)}\n\n"
                        await asyncio.sleep(0.01)
                    await asyncio.sleep(0.5)
            else:
                # Simulate speech chunks arriving incrementally
                accumulated = ""
                for i, chunk in enumerate(selected.inputs):
                    accumulated = f"{accumulated} {chunk}".strip()
                    is_final = (i == len(selected.inputs) - 1)
                    for event in live_engine.process_transcript_stream(
                        transcript_chunk=chunk,
                        session_id=sid,
                        accumulated_transcript=accumulated,
                        is_final_chunk=is_final,
                    ):
                        yield f"data: {json.dumps(event)}\n\n"
                        await asyncio.sleep(0.02)
                    if not is_final:
                        await asyncio.sleep(0.4)

        return StreamingResponse(
            scenario_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return app
