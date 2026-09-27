"""API test suite validating REST and streaming endpoints."""

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


def test_api_session_lifecycle(test_engine: StreamingLiveRAGEngine) -> None:
    app = create_app(engine=test_engine)
    client = TestClient(app)

    # 1. Create Session
    create_res = client.post("/api/session", json={"session_id": "test_api_sess_1"})
    assert create_res.status_code == 200
    assert create_res.json()["session_id"] == "test_api_sess_1"

    # 2. Get Session
    get_res = client.get("/api/session/test_api_sess_1")
    assert get_res.status_code == 200
    session_data = get_res.json()
    assert session_data["session_id"] == "test_api_sess_1"
    assert session_data["turns_count"] == 0

    # 3. List Sessions
    list_res = client.get("/api/sessions")
    assert list_res.status_code == 200
    sids = [s["session_id"] for s in list_res.json()]
    assert "test_api_sess_1" in sids

    # 4. Stream a turn to record telemetry and citations
    stream_res = client.post(
        "/api/stream/rag",
        json={
            "chunk": "Tell me about Samsung's Exynos 2400 processor specifications",
            "session_id": "test_api_sess_1",
            "is_final": True,
        },
    )
    assert stream_res.status_code == 200
    assert "data: " in stream_res.text

    # 5. Check Telemetry Endpoint
    telem_res = client.get("/api/telemetry/test_api_sess_1")
    assert telem_res.status_code == 200
    telem_data = telem_res.json()
    assert telem_data["count"] > 0
    assert len(telem_data["events"]) > 0

    # 6. Check Citations Endpoint
    cits_res = client.get("/api/citations/test_api_sess_1")
    assert cits_res.status_code == 200

    # 7. Delete Session
    del_res = client.delete("/api/session/test_api_sess_1")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "cleared"

    # Verify session deleted
    get_after = client.get("/api/session/test_api_sess_1")
    assert get_after.status_code == 404
