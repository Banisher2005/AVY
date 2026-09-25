"""Tests for Session Context, Follow-ups, and Query Refinement."""

from avy.context.session import SessionManager
from avy.retrieval.models import FusedEvidence


def test_session_lifecycle() -> None:
    sm = SessionManager()
    session = sm.get_or_create("sess_test_1")

    assert session.session_id == "sess_test_1"
    assert len(session.turns) == 0

    turn = sm.add_turn(
        session_id="sess_test_1",
        user_utterance="Tell me about Galaxy S24 display",
        controller_state="RETRIEVE",
        assistant_response="Galaxy S24 features a 1-120Hz Dynamic AMOLED 2X display.",
    )

    assert turn.turn_id == 1
    assert session.last_turn == turn
    assert len(session.turns) == 1


def test_followup_refinement_detection() -> None:
    sm = SessionManager()
    session = sm.get_or_create("sess_test_2")

    sm.add_turn(
        session_id="sess_test_2",
        user_utterance="Tell me about the Exynos 2400 processor",
        controller_state="RETRIEVE",
        assistant_response="Exynos 2400 is a 10-core AP.",
    )

    # Utterances that refer back to previous entity
    assert sm.is_followup_refinement(session, "Now compare its battery with S23") is True
    assert sm.is_followup_refinement(session, "what about its GPU?") is True
    assert sm.is_followup_refinement(session, "battery") is True

    # Standalone utterance with no pronouns
    assert sm.is_followup_refinement(session, "Tell me about Galaxy Z Fold 5 hinge") is False


def test_query_contextualization() -> None:
    sm = SessionManager()
    session = sm.get_or_create("sess_test_3")

    sm.add_turn(
        session_id="sess_test_3",
        user_utterance="Tell me about the Exynos 2400 processor",
        controller_state="RETRIEVE",
        assistant_response="Exynos 2400 is Samsung's AP.",
    )

    followup = "Now compare its battery life with S23."
    contextualized = sm.contextualize_query(session, followup)

    # Must contain both original intent and new comparative detail
    assert "Exynos 2400" in contextualized
    assert "battery life" in contextualized


def test_session_evidence_pool_persistence() -> None:
    sm = SessionManager()
    session_id = "sess_pool_1"

    ev1 = FusedEvidence(
        chunk_id="chunk_1",
        document_id="doc1",
        source="src1.md",
        title="Title 1",
        text="Evidence text 1",
        rrf_score=0.9,
    )
    ev2 = FusedEvidence(
        chunk_id="chunk_2",
        document_id="doc2",
        source="src2.md",
        title="Title 2",
        text="Evidence text 2",
        rrf_score=0.8,
    )

    sm.update_evidence_pool(session_id, [ev1, ev2])
    pool = sm.get_evidence_pool(session_id)

    assert len(pool) == 2
    assert {e.chunk_id for e in pool} == {"chunk_1", "chunk_2"}
