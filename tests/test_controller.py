"""Tests for the three-state Retrieval Controller."""

import pytest

from avy.config import AVYConfig
from avy.context.models import ConversationTurn, SessionContext
from avy.core.models import ControllerState
from avy.retrieval.controller import RetrievalController


@pytest.fixture
def controller() -> RetrievalController:
    cfg = AVYConfig(wait_token_min=3, early_retrieval_enabled=True)
    return RetrievalController(config=cfg)


def test_controller_wait_on_empty(controller: RetrievalController) -> None:
    decision = controller.evaluate("")
    assert decision.state == ControllerState.WAIT
    assert "empty" in decision.reason.lower()


def test_controller_wait_on_dangling_preposition(controller: RetrievalController) -> None:
    decision = controller.evaluate("Tell me about Samsung's")
    assert decision.state == ControllerState.WAIT
    assert "dangling" in decision.reason.lower() or "starter" in decision.reason.lower()


def test_controller_wait_on_ellipsis(controller: RetrievalController) -> None:
    decision = controller.evaluate("Explain the latest processor...")
    assert decision.state == ControllerState.WAIT
    assert "ellipsis" in decision.reason.lower()


def test_controller_wait_on_incomplete_starter(controller: RetrievalController) -> None:
    decision = controller.evaluate("Can you explain the")
    assert decision.state == ControllerState.WAIT


def test_controller_no_retrieve_on_greeting(controller: RetrievalController) -> None:
    for greeting in ["Hello AVY", "Hi there", "Good morning", "Hey"]:
        decision = controller.evaluate(greeting)
        assert decision.state == ControllerState.NO_RETRIEVE
        assert decision.confidence >= 0.9


def test_controller_no_retrieve_on_gratitude(controller: RetrievalController) -> None:
    for thanks in ["Thanks, that's helpful", "Thank you very much", "Got it, thanks"]:
        decision = controller.evaluate(thanks)
        assert decision.state == ControllerState.NO_RETRIEVE


def test_controller_no_retrieve_on_identity(controller: RetrievalController) -> None:
    decision = controller.evaluate("Who are you?")
    assert decision.state == ControllerState.NO_RETRIEVE


def test_controller_retrieve_on_factual_query(controller: RetrievalController) -> None:
    query = "Tell me about Samsung's latest semiconductor processor specifications."
    decision = controller.evaluate(query)
    assert decision.state == ControllerState.RETRIEVE
    assert decision.confidence > 0.9


def test_controller_early_retrieval(controller: RetrievalController) -> None:
    # A mid-utterance phrase that already contains a complete subject & predicate
    partial = "Samsung latest semiconductor processor"
    decision = controller.evaluate(partial)
    assert decision.state == ControllerState.RETRIEVE
    assert decision.is_early_retrieval is True


def test_controller_session_followup(controller: RetrievalController) -> None:
    session = SessionContext(session_id="s1")
    session.turns.append(
        ConversationTurn(
            turn_id=1,
            user_utterance="Tell me about Exynos 2400",
            controller_state="RETRIEVE",
            assistant_response="Exynos 2400 is Samsung's 10-core AP.",
        )
    )

    followup = "Now compare its battery with S23."
    decision = controller.evaluate(followup, session_context=session)
    assert decision.state == ControllerState.RETRIEVE
    assert decision.metadata.get("refinement") is True
