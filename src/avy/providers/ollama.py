"""Ollama provider implementation for AVY."""

import json
import time
import urllib.error
import urllib.request
from typing import Iterator

from avy.core.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
)
from avy.providers.base import BaseProvider
from avy.providers.models import (
    AgentRequest,
    AgentResponse,
    ProviderCapabilities,
    ResponseMetrics,
)


class OllamaProvider(BaseProvider):
    """Direct HTTP client for Ollama LLM service."""

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen3:4b",
        timeout: float = 30.0,
        temperature: float = 0.1,
    ) -> None:
        self.host = host.rstrip("/")
        if not (self.host.startswith("http://") or self.host.startswith("https://")):
            self.host = f"http://{self.host}"
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self._last_metrics: ResponseMetrics | None = None

    @property
    def last_metrics(self) -> ResponseMetrics | None:
        return self._last_metrics

    def get_model_name(self) -> str:
        return self.model

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            structured_output=True,
            local=True,
        )

    def is_available(self) -> bool:
        """Ping Ollama API to check if server is running."""
        url = f"{self.host}/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def send(self, request: AgentRequest) -> AgentResponse:
        """Send synchronous generation request."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "system": request.system_prompt or "",
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }
        if request.extra_options:
            payload["options"].update(request.extra_options)

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as err:
            if "timed out" in str(err).lower():
                raise ProviderTimeoutError(f"Ollama request timed out after {self.timeout}s: {err}")
            raise ProviderConnectionError(f"Cannot connect to Ollama at {self.host}: {err}")
        except Exception as err:
            raise ProviderError(f"Ollama error: {err}")

        total_ms = (time.perf_counter() - t0) * 1000.0
        text = raw_json.get("response", "") or raw_json.get("thinking", "")
        metrics = ResponseMetrics(
            total_duration_ms=total_ms,
            prompt_tokens=raw_json.get("prompt_eval_count"),
            completion_tokens=raw_json.get("eval_count"),
        )
        self._last_metrics = metrics
        return AgentResponse(text=text, metrics=metrics, raw=raw_json)

    def stream(self, request: AgentRequest) -> Iterator[str]:
        """Stream response chunks line-by-line from Ollama."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "system": request.system_prompt or "",
            "stream": True,
            "options": {
                "temperature": request.temperature,
            },
        }
        if request.extra_options:
            payload["options"].update(request.extra_options)

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        first_token_time: float | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    if not line:
                        continue
                    try:
                        chunk_obj = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue

                    chunk_text = chunk_obj.get("response", "") or chunk_obj.get("thinking", "")
                    if chunk_text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        yield chunk_text

                    if chunk_obj.get("done", False):
                        prompt_tokens = chunk_obj.get("prompt_eval_count")
                        completion_tokens = chunk_obj.get("eval_count")
                        break
        except urllib.error.URLError as err:
            if "timed out" in str(err).lower():
                raise ProviderTimeoutError(f"Ollama streaming timed out: {err}")
            raise ProviderConnectionError(f"Ollama connection broken: {err}")
        except Exception as err:
            raise ProviderError(f"Ollama stream error: {err}")
        finally:
            total_ms = (time.perf_counter() - t0) * 1000.0
            ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else total_ms
            self._last_metrics = ResponseMetrics(
                total_duration_ms=total_ms,
                ttft_ms=ttft_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
