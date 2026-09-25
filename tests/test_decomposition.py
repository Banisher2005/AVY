"""Tests for Multi-Intent Query Decomposition."""

from avy.providers.mock import MockProvider
from avy.retrieval.decomposition import MultiIntentDecomposer


def test_decomposition_compound_query(mock_provider: MockProvider) -> None:
    decomposer = MultiIntentDecomposer(provider=mock_provider)
    query = (
        "Compare Samsung's latest processor with the previous generation and explain the battery efficiency difference."
    )
    result = decomposer.decompose(query)

    assert result.is_compound is True
    assert len(result.subqueries) >= 2
    assert result.duration_ms >= 0.0

    sub_texts = [sq.subquery_text for sq in result.subqueries]
    assert any("processor" in s.lower() for s in sub_texts)
    assert any("battery" in s.lower() for s in sub_texts)


def test_decomposition_markdown_fence_parsing(mock_provider: MockProvider) -> None:
    decomposer = MultiIntentDecomposer(provider=mock_provider)
    raw_markdown = '```json\n{"queries": ["Galaxy S24 camera specs", "Galaxy S23 camera specs"]}\n```'
    parsed = decomposer._parse_provider_output(raw_markdown)

    assert len(parsed) == 2
    assert parsed[0] == "Galaxy S24 camera specs"
    assert parsed[1] == "Galaxy S23 camera specs"


def test_decomposition_bullet_fallback(mock_provider: MockProvider) -> None:
    decomposer = MultiIntentDecomposer(provider=mock_provider)
    bullet_text = "- Samsung semiconductor 3nm GAA\n- TSMC 4nm FinFET comparison"
    parsed = decomposer._parse_provider_output(bullet_text)

    assert len(parsed) == 2
    assert "3nm GAA" in parsed[0]


def test_decomposition_deduplication(mock_provider: MockProvider) -> None:
    decomposer = MultiIntentDecomposer(provider=mock_provider)
    queries = [
        "Samsung Exynos 2400 specifications",
        "tell me about Samsung Exynos 2400 specifications",
        "Samsung Exynos 2400 specifications?",
    ]
    cleaned = decomposer._clean_and_deduplicate(queries, fallback="fallback")
    assert len(cleaned) == 1
    assert "Samsung Exynos 2400 specifications" in cleaned[0]
