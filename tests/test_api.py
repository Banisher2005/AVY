"""API test suite validating REST endpoints for the AVY voice-first server."""

from fastapi.testclient import TestClient

from avy.streaming.engine import StreamingLiveRAGEngine
from avy.ui.server import create_app


def test_api_health_endpoint(test_engine: StreamingLiveRAGEngine) -> None:
    app = create_app(engine=test_engine)
    client = TestClient(app)

    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "provider" in data
    assert "indexed_chunks" in data
    assert "vector_dimension" in data
    assert data["indexed_chunks"] > 0


def test_api_scenarios_list(test_engine: StreamingLiveRAGEngine) -> None:
    app = create_app(engine=test_engine)
    client = TestClient(app)

    response = client.get("/api/scenarios")
    assert response.status_code == 200
    scenarios = response.json()
    assert len(scenarios) >= 6
    ids = [s["id"] for s in scenarios]
    assert any(s.startswith("A") for s in ids)
    assert any(s.startswith("B") for s in ids)
    assert any(s.startswith("C") for s in ids)
    assert any(s.startswith("D") for s in ids)
    assert any(s.startswith("E") for s in ids)
    assert any(s.startswith("F") for s in ids)
    # Each scenario has required fields
    for sc in scenarios:
        assert "id" in sc
        assert "name" in sc
        assert "expected_state" in sc


def test_api_query_sse_streaming(test_engine: StreamingLiveRAGEngine) -> None:
    """POST /api/query should stream SSE events for a complete utterance."""
    app = create_app(engine=test_engine)
    client = TestClient(app)

    response = client.post(
        "/api/query",
        json={
            "text": "Tell me about Samsung's Exynos 2400 processor specifications",
            "session_id": "test_api_query_sess",
        },
    )
    assert response.status_code == 200
    assert "data: " in response.text


def test_api_session_get_and_delete(test_engine: StreamingLiveRAGEngine) -> None:
    """Session GET and DELETE endpoints should work after a query runs."""
    app = create_app(engine=test_engine)
    client = TestClient(app)

    sid = "test_api_sess_lifecycle"

    # 1. Run a query to create the session
    client.post(
        "/api/query",
        json={
            "text": "What is Samsung Gauss?",
            "session_id": sid,
        },
    )

    # 2. Get Session
    get_res = client.get(f"/api/session/{sid}")
    assert get_res.status_code == 200
    session_data = get_res.json()
    assert session_data["session_id"] == sid
    assert "turns_count" in session_data

    # 3. List Sessions
    list_res = client.get("/api/sessions")
    assert list_res.status_code == 200
    sids = [s["session_id"] for s in list_res.json()]
    assert sid in sids

    # 4. Telemetry
    telem_res = client.get(f"/api/telemetry/{sid}")
    assert telem_res.status_code == 200
    telem_data = telem_res.json()
    assert telem_data["count"] >= 0  # May be 0 for WAIT/NO_RETRIEVE states

    # 5. Citations
    cits_res = client.get(f"/api/citations/{sid}")
    assert cits_res.status_code == 200

    # 6. Delete Session
    del_res = client.delete(f"/api/session/{sid}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "cleared"

    # 7. Session should be gone
    get_after = client.get(f"/api/session/{sid}")
    assert get_after.status_code == 404


def test_api_legacy_sse_endpoint(test_engine: StreamingLiveRAGEngine) -> None:
    """Legacy /api/stream/rag endpoint should still work."""
    app = create_app(engine=test_engine)
    client = TestClient(app)

    response = client.post(
        "/api/stream/rag",
        json={
            "chunk": "Tell me about Samsung Galaxy S24",
            "session_id": "test_legacy_sess",
            "is_final": True,
        },
    )
    assert response.status_code == 200
    assert "data: " in response.text


def test_api_scenario_run_sse(test_engine: StreamingLiveRAGEngine) -> None:
    """POST /api/scenarios/run should stream a complete scenario via SSE."""
    app = create_app(engine=test_engine)
    client = TestClient(app)

    response = client.post(
        "/api/scenarios/run",
        json={"scenario_id": "A1", "session_id": "test_scenario_sess"},
    )
    assert response.status_code == 200
    assert "data: " in response.text


def test_api_scenario_not_found(test_engine: StreamingLiveRAGEngine) -> None:
    app = create_app(engine=test_engine)
    client = TestClient(app)
    response = client.post("/api/scenarios/run", json={"scenario_id": "NONEXISTENT"})
    assert response.status_code == 404
