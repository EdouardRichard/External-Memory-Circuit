"""OpenAI-compatible LLM client (Model Gateway, blueprint sec 18).

Real HTTP calls via httpx — vendor-agnostic: works with any
OpenAI-compatible endpoint (DeepSeek, OpenAI, Anthropic-compatible
proxies, local models). No vendor is hardcoded (Constitution
architecture constraint); base_url / api_key / model all come from
run-config / environment.

Failure contract (SC-011): every error — connection, timeout, HTTP
status, malformed JSON — returns None so the calling agent falls back
to its deterministic equivalent. An LLM failure NEVER blocks or raises
into the state machine.

Response cache (T070, SC-008): an opt-in disk cache keyed by
(model, system prompt, user payload) makes the Agent orchestration
path byte-reproducible across same-session evaluation passes. The
first pass records real responses (successes AND failures — the eval
session replays both so pass 2 equals pass 1 exactly); later passes
replay them without any HTTP call. Cache hits do NOT count toward the
usage chars so recorded cost always reflects real LLM usage (SC-007).
The cache is enabled only when AGENTIC_LLM_CACHE_PATH (or an explicit
cache_dir) is configured — production callers never set it.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import re
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx

from rag_mcp.agents.consolidation_replay import (
    REPLAY_DENIED_REASON,
    cache_write_frozen,
    observe_cache_write,
    transport_denied,
)
from rag_mcp.agents.consolidation_replay import (
    cache_key as _replay_cache_key,
)
from rag_mcp.providers.concurrency import SyncConcurrencyLimiter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMCallReceipt:
    output: dict | None = None
    reason: str | None = None
    transport_calls: int = 0
    cache_hits: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    # Identifies one real transport attempt across the two receipts the strict
    # path emits for it (attempt started, attempt settled).  A consumer that
    # accumulates receipts must credit each call id once, or the attempt is
    # counted twice; 0 means "no real attempt" (configuration/cache-hit paths).
    call_id: int = 0


# Installed inside the provider worker, never shared between concurrent calls.
# The observer receives only accounting, never a DB handle or eligibility token.
receipt_observer = ContextVar('llm_receipt_observer', default=None)


def _emit_receipt(receipt):
    observer = receipt_observer.get()
    if observer is not None:
        observer(replace(receipt, output=None))
    return receipt


def _strict_object(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def invalid_constant(_):
        raise ValueError('nonfinite JSON')

    obj = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(obj, dict):
        raise TypeError('JSON object required')
    json.dumps(obj, allow_nan=False)
    return obj


def safe_failure_code(value):
    return isinstance(value, str) and (value in {
        'MODEL_CONFIGURATION_REQUIRED', 'MODEL_SCHEMA_INVALID', 'MODEL_JSON_INVALID',
        'MODEL_RESPONSE_TOO_LARGE', 'INPUT_CONTENT_UNSAFE', 'INPUT_BUDGET_EXCEEDED',
        'SCOPE_MISMATCH', 'AGENT_EXECUTION_FAILED', 'PROVIDER_TIMEOUT',
        'PROVIDER_NETWORK_ERROR', 'PROVIDER_RESPONSE_INVALID', 'GENERATED_CONTENT_UNSAFE',
    } or re.fullmatch(r'PROVIDER_HTTP_[1-5][0-9]{2}', value) is not None)


def _cache_validation_reason(output, validate_output):
    """A trusted validator may reject or fail, never leak its diagnostic body."""
    try:
        reason = validate_output(output)
    except Exception:  # noqa: BLE001 - validator failures must not persist the unvalidated package
        return 'AGENT_EXECUTION_FAILED'
    return reason if reason is None or safe_failure_code(reason) else 'AGENT_EXECUTION_FAILED'

# Markdown code-fence marker (three backticks), built without a literal
# backtick character in this source line.
_FENCE = chr(96) * 3
_FENCE_RE = re.compile(_FENCE + r"(?:json)?s*(.*?)s*" + _FENCE, re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM response body.

    Handles responses wrapped in markdown code fences or surrounded by
    prose. Returns None when no JSON object can be recovered.
    """
    if not text:
        return None
    stripped = text.strip()
    # Strip markdown code fences
    fence = _FENCE_RE.search(stripped)
    if fence:
        stripped = fence.group(1).strip()
    # Direct parse first
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back: locate the first { ... } block with brace balancing
    start = stripped.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(stripped)):
            if stripped[i] == "{":
                depth += 1
            elif stripped[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
        start = stripped.find("{", start + 1)
    return None


class LLMClient:
    """OpenAI-compatible chat client with JSON structured-output support.

    Makes REAL HTTP POST calls to {base_url}/chat/completions.
    All failures return None (the agent then degrades deterministically).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 5.0,
        cache_dir: str | None = None,
        limiter: SyncConcurrencyLimiter | None = None,
    ) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._api_key = api_key or ""
        self._model = model or ""
        self._timeout_s = timeout_s if timeout_s > 0 else 5.0
        # Response cache (T070, SC-008): empty/None disables caching.
        self._cache_dir = cache_dir or None
        # FR-009/§3.5: bound the number of in-flight HTTP calls. None keeps the
        # pre-existing unbounded behaviour for callers that construct a client
        # directly (tests, the capability router).
        self._limiter = limiter
        # Usage accounting for run-cost recording (T063, SC-007).
        # "calls" counts REAL HTTP calls only; cache hits are tracked
        # separately so recorded cost always reflects real usage.
        self.calls = 0
        self.prompt_chars = 0
        self.completion_chars = 0
        self.cache_hits = 0
        self.cache_misses = 0
        # Strict-path transport attempts are emitted twice (started, settled);
        # both emissions carry the same call_id so accounting can dedupe them.
        self._call_ids = itertools.count(1)

    @property
    def model(self) -> str:
        return self._model

    @property
    def configured(self) -> bool:
        return bool(self._base_url and self._model)

    def chat_json_receipt(self, system_prompt, user_payload, *, timeout_s=None, validate_output=None):
        """Strict whole-package parsing with call-local, sanitized diagnostics.

        Existing retrieval chat_json/cache semantics stay unchanged. Strict cache
        entries have a separate namespace so salvaged legacy objects cannot pass.
        The caller supplies a trusted complete-package validator for persistence;
        its rejection is stored and replayed as a failure with no original output.
        """
        if not self.configured:
            return _emit_receipt(LLMCallReceipt(reason='MODEL_CONFIGURATION_REQUIRED'))
        user_text = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, ensure_ascii=False)
        path = (Path(self._cache_dir) / 'strict-v1' / (self._cache_key(system_prompt, user_text) + '.json')) if self._cache_dir else None

        def persist(document):
            # A replay consumes recorded bytes (success or failure) and never
            # repairs, rewrites or gap-fills them (T091).
            if cache_write_frozen():
                return
            path.write_text(json.dumps(document, allow_nan=False), encoding='utf-8')
            # The record round's runner observes the exact key it persisted so
            # the sealed manifest is the real attempt set, never a prediction.
            observe_cache_write(path.name[: -len('.json')], document)

        if path:
            try:
                entry = _strict_object(path.read_text(encoding='utf-8'))
                if entry.get('parser') == 'strict-v1' and 'reason' in entry:
                    cached_reason = entry['reason']
                    if cached_reason is not None:
                        if not safe_failure_code(cached_reason):
                            trusted = {'parser': 'strict-v1', 'output': None, 'reason': 'AGENT_EXECUTION_FAILED'}
                            persist(trusted)
                            raise ValueError('invalid cached reason')
                        else:
                            trusted = {'parser': 'strict-v1', 'output': None, 'reason': cached_reason}
                            if entry != trusted:
                                persist(trusted)
                            self.cache_hits += 1
                            return _emit_receipt(LLMCallReceipt(reason=cached_reason, cache_hits=1))
                    if not isinstance(entry.get('output'), dict):
                        trusted = {'parser': 'strict-v1', 'output': None, 'reason': 'MODEL_SCHEMA_INVALID'}
                        persist(trusted)
                        self.cache_hits += 1
                        return _emit_receipt(LLMCallReceipt(
                            reason='MODEL_SCHEMA_INVALID', cache_hits=1))
                    if validate_output is not None:
                        validation_reason = _cache_validation_reason(entry['output'], validate_output)
                        if validation_reason:
                            trusted = {'parser': 'strict-v1', 'output': None, 'reason': validation_reason}
                            persist(trusted)
                            self.cache_hits += 1
                            return _emit_receipt(LLMCallReceipt(
                                reason=validation_reason, cache_hits=1))
                    trusted = {'parser': 'strict-v1', 'output': entry['output'], 'reason': None}
                    if entry != trusted:
                        persist(trusted)
                    self.cache_hits += 1
                    return _emit_receipt(LLMCallReceipt(output=entry['output'], cache_hits=1))
            except (OSError, ValueError, TypeError, RecursionError):
                # Read-through cache misses carry no response body diagnostics.
                pass
        self.cache_misses += 1
        if transport_denied():
            # Replay: a missing/corrupt/unusable entry is a deterministic
            # degradation, never a provider call and never a new cache entry.
            return _emit_receipt(LLMCallReceipt(reason=REPLAY_DENIED_REASON))
        receipt = LLMCallReceipt()
        body = {'model': self._model, 'messages': [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_text}], 'temperature': 0.0}
        headers = {'Content-Type': 'application/json'}
        if self._api_key:
            headers['Authorization'] = 'Bearer ' + self._api_key
        try:
            with httpx.Client(timeout=timeout_s or self._timeout_s) as http:
                receipt = LLMCallReceipt(transport_calls=1, prompt_chars=len(system_prompt) + len(user_text),
                                         call_id=next(self._call_ids))
                self.calls += 1
                self.prompt_chars += receipt.prompt_chars
                _emit_receipt(receipt)
                with self._limiter if self._limiter is not None else nullcontext():
                    response = http.post(self._base_url + '/chat/completions', json=body, headers=headers)
            if response.status_code != 200:
                receipt = replace(receipt, reason=f'PROVIDER_HTTP_{response.status_code}')
            else:
                try:
                    envelope = response.json()
                    content = envelope['choices'][0]['message']['content']
                    if not isinstance(content, str):
                        raise TypeError('invalid content')
                except (ValueError, TypeError, KeyError, IndexError):
                    receipt = replace(receipt, reason='PROVIDER_RESPONSE_INVALID')
                else:
                    receipt = replace(receipt, completion_chars=len(content))
                    usage = envelope.get('usage') or {}
                    for source, target in (('prompt_tokens', 'input_tokens'), ('completion_tokens', 'output_tokens')):
                        value = usage.get(source) if isinstance(usage, dict) else None
                        if type(value) is int and value >= 0:
                            receipt = replace(receipt, **{target: value})
                    cost = usage.get('cost_usd') if isinstance(usage, dict) else None
                    if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
                        receipt = replace(receipt, cost_usd=cost)
                    if len(content) > 1000000:
                        receipt = replace(receipt, reason='MODEL_RESPONSE_TOO_LARGE')
                    else:
                        try:
                            receipt = replace(receipt, output=_strict_object(content))
                        except (ValueError, TypeError, RecursionError):
                            receipt = replace(receipt, reason='MODEL_JSON_INVALID')
                        else:
                            if validate_output is not None and path is not None:
                                validation_reason = _cache_validation_reason(receipt.output, validate_output)
                                if validation_reason:
                                    receipt = replace(receipt, output=None, reason=validation_reason)
        except httpx.TimeoutException:
            receipt = replace(receipt, reason='PROVIDER_TIMEOUT')
        except httpx.RequestError:
            receipt = replace(receipt, reason='PROVIDER_NETWORK_ERROR')
        except Exception:  # noqa: BLE001 - external failures have sanitized stable reason codes
            receipt = replace(receipt, output=None, reason='PROVIDER_RESPONSE_INVALID')
        self.completion_chars += receipt.completion_chars
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                persist({'parser': 'strict-v1', 'output': receipt.output, 'reason': receipt.reason})
            except (OSError, ValueError, TypeError):
                logger.warning('LLMClient strict cache write failed')
        return _emit_receipt(receipt)

    # ------------------------------------------------------------------
    # Response cache (T070, SC-008)
    # ------------------------------------------------------------------

    def _cache_key(self, system_prompt: str, user_text: str) -> str:
        """Stable cache key: sha256 over (model, system prompt, user text)."""
        return _replay_cache_key(self._model, system_prompt, user_text)

    def _cache_lookup(
        self, system_prompt: str, user_text: str,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Return (found, payload) for a cached call.

        found=True with payload=None means a recorded FAILURE being
        replayed (record-replay semantics: pass 2 replays pass 1
        failures too). found=False means a cache miss. Corrupt or
        unreadable entries are treated as misses (best-effort cache).
        """
        if not self._cache_dir:
            return False, None
        path = Path(self._cache_dir) / (self._cache_key(system_prompt, user_text) + ".json")
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False, None
        if not isinstance(entry, dict) or "ok" not in entry:
            return False, None
        return True, entry.get("response") if entry.get("ok") else None

    def _cache_store(
        self,
        system_prompt: str,
        user_text: str,
        payload: dict[str, Any] | None,
    ) -> None:
        """Record a call outcome (success or failure) in the cache."""
        if not self._cache_dir:
            return
        if cache_write_frozen():
            # Replay keeps the recorded bytes and adds no new provider outcome.
            return
        entry = {"ok": payload is not None, "response": payload}
        path = Path(self._cache_dir) / (self._cache_key(system_prompt, user_text) + ".json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(entry, ensure_ascii=False), encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.warning("LLMClient: cache write failed: %s", exc)

    def chat_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any] | str,
    ) -> dict[str, Any] | None:
        """Call the chat completions endpoint and parse JSON from the answer.

        Returns the parsed JSON dict, or None on ANY failure (SC-011).
        With the response cache enabled (T070), identical
        (system, payload) calls replay the recorded outcome without
        any HTTP call, making evaluation passes byte-reproducible.
        """
        user_text = (
            user_payload
            if isinstance(user_payload, str)
            else json.dumps(user_payload, ensure_ascii=False)
        )

        # Cache lookup first: a hit never touches the network and never
        # counts usage chars (cost stays real, SC-007).
        found, cached_payload = self._cache_lookup(system_prompt, user_text)
        if found:
            self.cache_hits += 1
            return cached_payload
        self.cache_misses += 1

        if transport_denied():
            # Replay: only the LLM transport is denied and counted; the
            # deterministic offline result is returned instead of a provider call.
            return None

        self.calls += 1
        self.prompt_chars += len(system_prompt) + len(user_text)

        if not self._base_url:
            logger.warning("LLMClient: no base_url configured; returning None")
            return None
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key

        url = self._base_url + "/chat/completions"
        try:
            with httpx.Client(timeout=self._timeout_s) as http:
                with self._limiter if self._limiter is not None else nullcontext():
                    resp = http.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                logger.warning(
                    "LLMClient: HTTP %s from %s", resp.status_code, url,
                )
                self._cache_store(system_prompt, user_text, None)
                return None
            data = resp.json()
        except Exception as exc:
            logger.warning("LLMClient: request to %s failed: %s", url, exc)
            self._cache_store(system_prompt, user_text, None)
            return None

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.warning("LLMClient: unexpected response shape from %s", url)
            self._cache_store(system_prompt, user_text, None)
            return None

        payload = _extract_json(content)
        self._cache_store(system_prompt, user_text, payload)
        if content:
            self.completion_chars += len(content)
        return payload
