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
            "provider_available": live_engine.provider.is_available(),
            "indexed_chunks": len(live_engine.vectorstore._chunks),
            "reranker_enabled": live_engine.config.reranker_enabled,
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
