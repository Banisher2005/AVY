"""Curated evaluation dataset for Samsung PRISM Theme 04 benchmark scenarios."""

from dataclasses import dataclass, field
from typing import Any

from avy.core.models import ControllerState


@dataclass
class EvalTestCase:
    """A test case for evaluating streaming live RAG capabilities."""

    test_id: str
    scenario_name: str
    description: str
    inputs: list[str]  # sequential chunks or turns
    expected_controller_state: ControllerState
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_topics: list[str] = field(default_factory=list)
    is_multi_turn: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def get_standard_eval_dataset() -> list[EvalTestCase]:
    """Return standard test suite covering all 6 Theme 04 scenarios."""
    return [
        # Scenario A: NO_RETRIEVE (Conversational / Greeting)
        EvalTestCase(
            test_id="A1",
            scenario_name="Scenario A: NO_RETRIEVE (Greeting)",
            description="Conversational greeting should not trigger corpus retrieval",
            inputs=["Hello AVY, how are you today?"],
            expected_controller_state=ControllerState.NO_RETRIEVE,
            expected_doc_ids=[],
            expected_topics=["greeting"],
        ),
        EvalTestCase(
            test_id="A2",
            scenario_name="Scenario A: NO_RETRIEVE (Gratitude)",
            description="Polite acknowledgment does not require corpus retrieval",
            inputs=["Thanks, that's really helpful."],
            expected_controller_state=ControllerState.NO_RETRIEVE,
            expected_doc_ids=[],
            expected_topics=["gratitude"],
        ),

        # Scenario B: WAIT (Incomplete Utterance)
        EvalTestCase(
            test_id="B1",
            scenario_name="Scenario B: WAIT (Trailing Preposition)",
            description="Incomplete sentence ending with dangling preposition must WAIT",
            inputs=["Tell me about Samsung's..."],
            expected_controller_state=ControllerState.WAIT,
            expected_doc_ids=[],
            expected_topics=[],
        ),
        EvalTestCase(
            test_id="B2",
            scenario_name="Scenario B: WAIT (Phrase Starter)",
            description="Incomplete question starter without semantic object must WAIT",
            inputs=["Can you explain the"],
            expected_controller_state=ControllerState.WAIT,
            expected_doc_ids=[],
            expected_topics=[],
        ),

        # Scenario C: EARLY RETRIEVAL (Mid-utterance trigger)
        EvalTestCase(
            test_id="C1",
            scenario_name="Scenario C: EARLY RETRIEVAL",
            description="Transcript has enough semantic information to begin early retrieval",
            inputs=[
                "Tell me about Samsung's",
                "latest semiconductor processor",
            ],
            expected_controller_state=ControllerState.RETRIEVE,
            expected_doc_ids=["samsung_semiconductor", "samsung_exynos_2400"],
            expected_topics=["semiconductor", "processor"],
        ),

        # Scenario D: MULTI-INTENT (Compound question)
        EvalTestCase(
            test_id="D1",
            scenario_name="Scenario D: MULTI-INTENT DECOMPOSITION",
            description="Compound query requiring multi-intent decomposition and multi-query retrieval",
            inputs=[
                "Compare Samsung's latest processor with the previous generation and explain the battery efficiency difference."
            ],
            expected_controller_state=ControllerState.RETRIEVE,
            expected_doc_ids=["samsung_exynos_2400", "samsung_battery_power"],
            expected_topics=["processor", "battery", "efficiency"],
        ),

        # Scenario E: SESSION REFINEMENT (Multi-turn follow-up)
        EvalTestCase(
            test_id="E1",
            scenario_name="Scenario E: SESSION REFINEMENT",
            description="Follow-up question with pronoun must use prior session context",
            inputs=[
                "Tell me about the Samsung Galaxy S24 Ultra display.",
                "Now compare its battery with the S23.",
            ],
            expected_controller_state=ControllerState.RETRIEVE,
            expected_doc_ids=["samsung_galaxy_s24", "samsung_battery_power"],
            expected_topics=["display", "battery", "s24", "s23"],
            is_multi_turn=True,
        ),

        # Scenario F: LATE DETAIL (Utterance refinement)
        EvalTestCase(
            test_id="F1",
            scenario_name="Scenario F: LATE DETAIL",
            description="Subsequent audio chunk refines original retrieval scope",
            inputs=[
                "Tell me about the Exynos 2400",
                "and explain its AMD Xclipse GPU architecture.",
            ],
            expected_controller_state=ControllerState.RETRIEVE,
            expected_doc_ids=["samsung_exynos_2400"],
            expected_topics=["exynos", "gpu", "xclipse"],
        ),
    ]
