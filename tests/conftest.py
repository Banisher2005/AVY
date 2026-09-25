"""Pytest fixtures for AVY test suite."""

import sys
from pathlib import Path

import pytest

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from avy.config import AVYConfig
from avy.providers.mock import MockProvider
from avy.retrieval.embeddings import LocalDenseEmbedding
from avy.retrieval.models import Chunk
from avy.retrieval.vectorstore import FAISSVectorStore
from avy.streaming.engine import StreamingLiveRAGEngine
from avy.telemetry.logger import TelemetryLogger


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider(model_name="mock-test-model")


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="samsung_exynos_2400#c0",
            doc_id="samsung_exynos_2400",
            source="samsung_exynos_2400.md",
            title="Exynos 2400 Specifications",
            chunk_index=0,
            text="The Samsung Exynos 2400 has a 10-core CPU and AMD Xclipse 940 GPU with ray tracing.",
            metadata={"category": "processor"},
        ),
        Chunk(
            chunk_id="samsung_semiconductor#c0",
            doc_id="samsung_semiconductor",
            source="samsung_semiconductor.md",
            title="Samsung Semiconductor Innovations",
            chunk_index=0,
            text="Samsung Foundry mass produces 3nm GAA MBCFET semiconductor chips with 45% power reduction.",
            metadata={"category": "semiconductor"},
        ),
        Chunk(
            chunk_id="samsung_battery_power#c0",
            doc_id="samsung_battery_power",
            source="samsung_battery_power.md",
            title="Battery and Efficiency",
            chunk_index=0,
            text="The Galaxy S24 Ultra has a 5,000 mAh battery with 45W Super Fast Charging and 25% better power efficiency.",
            metadata={"category": "battery"},
        ),
        Chunk(
            chunk_id="samsung_galaxy_s24#c0",
            doc_id="samsung_galaxy_s24",
            source="samsung_galaxy_s24.md",
            title="Samsung Galaxy S24 Series",
            chunk_index=0,
            text="The Galaxy S24 Ultra display features a 6.8-inch Dynamic AMOLED 2X panel with 2,600 nits peak brightness.",
            metadata={"category": "phone"},
        ),
        Chunk(
            chunk_id="samsung_foldable_displays#c0",
            doc_id="samsung_foldable_displays",
            source="samsung_foldable_displays.md",
            title="Flex Hinge and Display",
            chunk_index=0,
            text="Samsung's Flex Hinge allows zero gap closure and Ultra Thin Glass provides 200,000 fold cycles.",
            metadata={"category": "display"},
        ),
    ]


@pytest.fixture
def local_embedding() -> LocalDenseEmbedding:
    return LocalDenseEmbedding(dimension=128)


@pytest.fixture
def sample_vectorstore(sample_chunks: list[Chunk], local_embedding: LocalDenseEmbedding) -> FAISSVectorStore:
    vstore = FAISSVectorStore(dimension=local_embedding.dimension)
    texts = [c.text for c in sample_chunks]
    embs = local_embedding.embed_documents(texts)
    vstore.add_chunks(sample_chunks, embs)
    return vstore


@pytest.fixture
def test_engine(
    mock_provider: MockProvider,
    sample_vectorstore: FAISSVectorStore,
    local_embedding: LocalDenseEmbedding,
    tmp_path: Path,
) -> StreamingLiveRAGEngine:
    cfg = AVYConfig(
        provider="mock",
        embedding_provider="local_dense",
        reranker_enabled=True,
        telemetry_enabled=True,
        telemetry_log=str(tmp_path / "test_events.jsonl"),
    )
    telemetry = TelemetryLogger(log_path=tmp_path / "test_events.jsonl", enabled=True)
    engine = StreamingLiveRAGEngine(
        config=cfg,
        provider=mock_provider,
        vectorstore=sample_vectorstore,
        telemetry=telemetry,
    )
    engine.embeddings = local_embedding
    engine.retriever.embeddings = local_embedding
    return engine
