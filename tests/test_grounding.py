"""Tests for Grounding Engine and Citations Traceability."""

from avy.retrieval.grounding import GroundingEngine
from avy.retrieval.models import FusedEvidence


def test_build_synthesis_prompt() -> None:
    engine = GroundingEngine()
    evidence = [
        FusedEvidence(
            chunk_id="chunk_proc_1",
            document_id="doc_proc",
            source="semiconductor.md",
            title="3nm GAA Process",
            text="Samsung mass produces 3nm GAA MBCFET transistors reducing power by 45%.",
            rrf_score=0.95,
            rank=1,
        )
    ]

    sys_prompt, user_prompt = engine.build_synthesis_prompt(
        utterance="Explain Samsung's 3nm technology",
        evidence=evidence,
    )

    assert "strictly" in sys_prompt.lower()
    assert "[1]" in user_prompt
    assert "semiconductor.md" in user_prompt
    assert "45%" in user_prompt


def test_extract_citations() -> None:
    engine = GroundingEngine()
    evidence = [
        FusedEvidence(
            chunk_id="exynos#c0",
            document_id="exynos_2400",
            source="exynos_2400.md",
            title="Exynos 2400 Specs",
            text="The Exynos 2400 features a 10-core CPU layout and AMD Xclipse 940 GPU.",
            rrf_score=0.9,
            rank=1,
        ),
        FusedEvidence(
            chunk_id="battery#c0",
            document_id="battery_power",
            source="battery_power.md",
            title="Battery Specs",
            text="The S24 Ultra has a 5,000 mAh battery with 45W wired charging.",
            rrf_score=0.8,
            rank=2,
        ),
    ]

    answer = (
        "Samsung's Exynos 2400 utilizes a 10-core processor and AMD Xclipse 940 GPU [1]. "
        "Additionally, the device supports 45W fast charging with a 5,000 mAh capacity [2]."
    )

    citations = engine.extract_citations(answer, evidence)

    assert len(citations) == 2
    assert citations[0].index == 1
    assert citations[0].document_id == "exynos_2400"
    assert citations[0].source == "exynos_2400.md"
    assert citations[0].chunk_id == "exynos#c0"

    assert citations[1].index == 2
    assert citations[1].document_id == "battery_power"
    assert citations[1].source == "battery_power.md"


def test_groundedness_score_calculation() -> None:
    engine = GroundingEngine()
    evidence = [
        FusedEvidence(
            chunk_id="c1",
            document_id="d1",
            source="s1.md",
            title="T1",
            text="Samsung Exynos 2400 deca core CPU and ray tracing graphics.",
            rrf_score=0.9,
        )
    ]

    grounded_answer = "The Exynos 2400 includes deca core CPU and ray tracing graphics."
    score = engine.calculate_groundedness_score(grounded_answer, evidence)
    assert score >= 0.7

    hallucinated_answer = "Apple A18 Bionic chip with quantum computing teleportation."
    hallucinated_score = engine.calculate_groundedness_score(hallucinated_answer, evidence)
    assert hallucinated_score < 0.2
