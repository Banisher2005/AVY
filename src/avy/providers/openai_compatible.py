"""OpenAI-compatible provider implementation for AVY."""

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


class OpenAICompatibleProvider(BaseProvider):
    """Client for any OpenAI-compatible completions API."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        temperature: float = 0.1,
    ) -> None:
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or ""
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
            local=False,
        )

    def is_available(self) -> bool:
        """Check API reachability."""
        url = f"{self.base_url}/models"
        req = urllib.request.Request(url, method="GET")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def send(self, request: AgentRequest) -> AgentResponse:
        """Synchronous chat completion."""
        url = f"{self.base_url}/chat/completions"
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": False,
        }
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as err:
            if "timed out" in str(err).lower():
                raise ProviderTimeoutError(f"API request timed out: {err}")
            raise ProviderConnectionError(f"Cannot connect to API at {self.base_url}: {err}")
        except Exception as err:
            raise ProviderError(f"API error: {err}")

        total_ms = (time.perf_counter() - t0) * 1000.0
        text = raw_json["choices"][0]["message"]["content"]
        usage = raw_json.get("usage", {})
        metrics = ResponseMetrics(
            total_duration_ms=total_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
        self._last_metrics = metrics
        return AgentResponse(text=text, metrics=metrics, raw=raw_json)

    def stream(self, request: AgentRequest) -> Iterator[str]:
        """Stream SSE response chunks from OpenAI-compatible API."""
        url = f"{self.base_url}/chat/completions"
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        t0 = time.perf_counter()
        first_token_time: float | None = None

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    decoded = line.decode("utf-8").strip()
                    if not decoded or not decoded.startswith("data:"):
                        continue
                    payload_part = decoded[5:].strip()
                    if payload_part == "[DONE]":
                        break
                    try:
                        chunk_obj = json.loads(payload_part)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk_obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        chunk_text = delta.get("content", "")
                        if chunk_text:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            yield chunk_text
        except urllib.error.URLError as err:
            if "timed out" in str(err).lower():
                raise ProviderTimeoutError(f"API streaming timed out: {err}")
            raise ProviderConnectionError(f"API streaming broken: {err}")
        except Exception as err:
            raise ProviderError(f"API stream error: {err}")
        finally:
            total_ms = (time.perf_counter() - t0) * 1000.0
            ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else total_ms
            self._last_metrics = ResponseMetrics(
                total_duration_ms=total_ms,
                ttft_ms=ttft_ms,
            )
