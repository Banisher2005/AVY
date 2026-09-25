"""Configuration management for AVY."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_TIMEOUT = 30.0
DEFAULT_TEMPERATURE = 0.1

DEFAULT_EMBEDDING_PROVIDER = "fastembed"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_RERANKER_ENABLED = True
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

DEFAULT_CORPUS_DIR = "corpus"
DEFAULT_INDEX_PATH = "data/index/avy_faiss.index"
DEFAULT_METADATA_PATH = "data/index/avy_metadata.json"
DEFAULT_TELEMETRY_LOG = "data/telemetry/events.jsonl"


@dataclass
class AVYConfig:
    """Runtime configuration for AVY Streaming Live RAG Assistant."""

    # LLM Settings
    provider: str = "ollama"
    model: str = DEFAULT_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    temperature: float = DEFAULT_TEMPERATURE

    # Embedding Settings
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    # Reranker Settings
    reranker_enabled: bool = DEFAULT_RERANKER_ENABLED
    reranker_model: str = DEFAULT_RERANKER_MODEL

    # Retrieval & Vector Store Parameters
    corpus_dir: str = DEFAULT_CORPUS_DIR
    index_path: str = DEFAULT_INDEX_PATH
    metadata_path: str = DEFAULT_METADATA_PATH
    top_k: int = 5
    rrf_k: int = 60
    similarity_threshold: float = 0.25

    # Retrieval Controller Settings
    controller_mode: str = "hybrid"
    wait_token_min: int = 3
    early_retrieval_enabled: bool = True

    # Telemetry Settings
    telemetry_enabled: bool = True
    telemetry_log: str = DEFAULT_TELEMETRY_LOG

    # Web & API Server
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    @classmethod
    def load(cls, **overrides: Any) -> "AVYConfig":
        """Load configuration prioritizing: overrides > env vars > file > defaults."""
        data: dict[str, Any] = {}

        # 1. Config file ~/.config/avy/config.json if present
        config_file = Path.home() / ".config" / "avy" / "config.json"
        if config_file.is_file():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    if isinstance(file_data, dict):
                        data.update(file_data)
            except Exception:
                pass

        # 2. Environment variables
        env_mappings = {
            "AVY_PROVIDER": ("provider", str),
            "AVY_MODEL": ("model", str),
            "AVY_OLLAMA_HOST": ("ollama_host", str),
            "AVY_API_KEY": ("api_key", str),
            "AVY_BASE_URL": ("base_url", str),
            "AVY_TIMEOUT": ("timeout", float),
            "AVY_TEMPERATURE": ("temperature", float),
            "AVY_EMBEDDING_PROVIDER": ("embedding_provider", str),
            "AVY_EMBEDDING_MODEL": ("embedding_model", str),
            "AVY_RERANKER_ENABLED": ("reranker_enabled", lambda v: v.lower() in ("1", "true", "yes")),
            "AVY_RERANKER_MODEL": ("reranker_model", str),
            "AVY_CORPUS_DIR": ("corpus_dir", str),
            "AVY_INDEX_PATH": ("index_path", str),
            "AVY_METADATA_PATH": ("metadata_path", str),
            "AVY_TOP_K": ("top_k", int),
            "AVY_RRF_K": ("rrf_k", int),
            "AVY_SIMILARITY_THRESHOLD": ("similarity_threshold", float),
            "AVY_CONTROLLER_MODE": ("controller_mode", str),
            "AVY_WAIT_TOKEN_MIN": ("wait_token_min", int),
            "AVY_TELEMETRY_ENABLED": ("telemetry_enabled", lambda v: v.lower() in ("1", "true", "yes")),
            "AVY_TELEMETRY_LOG": ("telemetry_log", str),
            "AVY_SERVER_HOST": ("server_host", str),
            "AVY_SERVER_PORT": ("server_port", int),
        }

        for env_key, (attr_key, parser) in env_mappings.items():
            val = os.getenv(env_key)
            if val is not None and val != "":
                try:
                    data[attr_key] = parser(val)
                except Exception:
                    pass

        # 3. Explicit overrides
        for k, v in overrides.items():
            if v is not None:
                data[k] = v

        return cls(
            provider=data.get("provider", "ollama"),
            model=data.get("model", DEFAULT_MODEL),
            ollama_host=data.get("ollama_host", DEFAULT_OLLAMA_HOST),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            timeout=float(data.get("timeout", DEFAULT_TIMEOUT)),
            temperature=float(data.get("temperature", DEFAULT_TEMPERATURE)),
            embedding_provider=data.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER),
            embedding_model=data.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            reranker_enabled=bool(data.get("reranker_enabled", DEFAULT_RERANKER_ENABLED)),
            reranker_model=data.get("reranker_model", DEFAULT_RERANKER_MODEL),
            corpus_dir=data.get("corpus_dir", DEFAULT_CORPUS_DIR),
            index_path=data.get("index_path", DEFAULT_INDEX_PATH),
            metadata_path=data.get("metadata_path", DEFAULT_METADATA_PATH),
            top_k=int(data.get("top_k", 5)),
            rrf_k=int(data.get("rrf_k", 60)),
            similarity_threshold=float(data.get("similarity_threshold", 0.25)),
            controller_mode=data.get("controller_mode", "hybrid"),
            wait_token_min=int(data.get("wait_token_min", 3)),
            early_retrieval_enabled=bool(data.get("early_retrieval_enabled", True)),
            telemetry_enabled=bool(data.get("telemetry_enabled", True)),
            telemetry_log=data.get("telemetry_log", DEFAULT_TELEMETRY_LOG),
            server_host=data.get("server_host", "127.0.0.1"),
            server_port=int(data.get("server_port", 8000)),
        )
