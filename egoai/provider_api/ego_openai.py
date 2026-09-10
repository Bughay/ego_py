"""
Direct HTTP client for OpenAI-compatible chat-completions APIs.

Replaces the ``openai`` SDK with the project's own curl-style HTTP logic.
The implementation is stdlib-only (urllib) — zero third-party
dependencies — and the entire functionality lives in a single class,
:class:`EgoOpenAI`, which works in two modes:

* Client mode (``EgoOpenAI(api_key=..., base_url=..., timeout=...)``):
  sends requests and reproduces the SDK request-normalization behaviors the
  providers' ``_build_payload()`` rely on — ``payload["extra_body"]`` is
  merged into the top level of the JSON body, and top-level ``None`` values
  are omitted (the SDK strips them; e.g. ``reasoning_effort: None`` must not
  be sent as ``null``). Transient failures (408/409/429/5xx and connection
  errors) are retried twice with exponential backoff, mirroring the SDK's
  default ``max_retries=2``.

* Response mode (returned by ``create()``): the parsed JSON is wrapped in
  instances of the same class, so it can be consumed with the same attribute
  access the SDK objects provided — ``response.choices[0].message.content``,
  ``getattr(message, "reasoning_content", None)``, ``usage.total_tokens``.
  Missing keys resolve to ``None`` instead of raising, which matches the
  optional fields the SDK models define. Dict-style access is supported too:
  ``obj["key"]``, ``"key" in obj``, ``obj.get("key", default)``.

The API key is always supplied by the caller — the provider classes pass
``self._get_api_key()`` in; this class never reads the environment itself.

Usage:
    from egoai.provider_api import EgoOpenAI

    api = EgoOpenAI(api_key="sk-...")   # api_key=None raises RuntimeError
    response = api.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10000,
        temperature=0.5,
        reasoning_effort=None,
        extra_body={"thinking": {"type": "disabled"}},
    )
    print(response.choices[0].message.content)
    api.close()
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class EgoOpenAI:
    """Direct HTTP client for OpenAI-compatible chat-completions APIs.

    Mimics the one call path the providers use —
    ``client.chat.completions.create(**payload)`` — so an existing
    ``one_shot()`` body built around the OpenAI SDK can keep working
    unchanged against this object.

    The same class also wraps response JSON: ``create()`` returns an
    EgoOpenAI response node, and nested dicts/lists are further nodes of
    the same class, giving SDK-style attribute access.
    """

    # Statuses the SDK retries, plus our connection-level failure marker.
    _RETRYABLE_STATUSES = {408, 409, 429}
    _MAX_RETRIES = 2  # 1 initial attempt + 2 retries, like the SDK default

#-------------------------- magic methods --------------------------------------

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 600.0,
    ):
        # Response nodes reuse this same class; None marks this instance as a
        # client, a dict marks it as a wrapped response node.
        self._data: Optional[Dict[str, Any]] = None
        self._http = urllib.request.build_opener()
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def __getattr__(self, name: str) -> Any:
        # Internal attributes resolve normally, so a missing private name is
        # a genuine bug, never a JSON key lookup.
        if name.startswith("_"):
            raise AttributeError(name)
        if self._data is not None:
            # Response mode: present keys resolve (nested dicts wrapped on
            # access), absent keys fall back to None like the SDK's optional
            # fields.
            if name in self._data:
                return self._wrap(self._data[name])
            return None
        # Client mode: SDK-style namespace shim so
        # api.chat.completions.create(...) resolves. "chat" and "completions"
        # are not real attributes — they chain back to this same client.
        if name in ("chat", "completions"):
            return self
        raise AttributeError(name)

    def __getitem__(self, key: str) -> Any:
        self._require_data()
        return self._wrap(self._data[key])

    def __contains__(self, key: str) -> bool:
        self._require_data()
        return key in self._data

    def __repr__(self) -> str:
        if self._data is not None:
            return f"EgoOpenAI({self._data!r})"
        return f"EgoOpenAI(base_url={self._base_url!r}, timeout={self._timeout!r})"

    def __enter__(self) -> "EgoOpenAI":
        self._require_client()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup: never raise during interpreter shutdown.
        http = getattr(self, "_http", None)
        if http is not None and hasattr(http, "close"):
            try:
                http.close()
            except Exception:
                pass

#-------------------------- properties -----------------------------------------

    @property
    def api_key(self) -> str:
        self._require_client()
        return self._api_key

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        self._require_client()
        if value is None:
            raise RuntimeError(
                "api_key is required. Pass api_key=... to EgoOpenAI "
                "(the provider classes supply it via _get_api_key())."
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError("api_key must be a non-empty string")
        self._api_key = value

    @property
    def base_url(self) -> str:
        self._require_client()
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._require_client()
        if not isinstance(value, str) or not value.strip():
            raise ValueError("base_url must be a non-empty string")
        self._base_url = value.rstrip("/")

    @property
    def timeout(self) -> float:
        self._require_client()
        return self._timeout

    @timeout.setter
    def timeout(self, value: float) -> None:
        self._require_client()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("timeout must be a positive number of seconds")
        if value <= 0:
            raise ValueError("timeout must be a positive number of seconds")
        self._timeout = float(value)

    @property
    def http(self) -> Any:
        """The underlying urllib opener (read-only; use close() to release it)."""
        self._require_client()
        return self._http

#-------------------------- private helpers -------------------------------------

    def _require_client(self) -> None:
        if self._data is not None or self._http is None:
            raise TypeError(
                "this EgoOpenAI instance is a response object, not an API client"
            )

    def _require_data(self) -> None:
        if self._data is None:
            raise TypeError(
                "this EgoOpenAI instance is an API client, not a response object"
            )

    @classmethod
    def _from_data(cls, data: Dict[str, Any]) -> "EgoOpenAI":
        """Build a response node of this same class around a parsed JSON dict."""
        node = object.__new__(cls)
        node._data = data
        node._http = None
        node._api_key = None
        node._base_url = None
        node._timeout = None
        return node

    def _wrap(self, value: Any) -> Any:
        """Recursively wrap JSON dicts so attribute access works like SDK objects."""
        if isinstance(value, dict):
            return self.__class__._from_data(value)
        if isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Reproduce the OpenAI SDK's request normalization for a raw POST.

        ``extra_body`` is merged into the top level of the JSON body (the SDK
        does this; a raw POST would otherwise send a literal ``extra_body``
        field the API does not understand) and top-level ``None`` values are
        dropped (``reasoning_effort: None`` must not be sent as ``null``).
        """
        extra = payload.pop("extra_body", None) or {}
        body = {k: v for k, v in payload.items() if v is not None}
        return {**body, **extra}

    @staticmethod
    def _should_retry(status: int) -> bool:
        """True for transient failures the SDK retries (408/409/429/5xx/connect)."""
        return status == 0 or status in EgoOpenAI._RETRYABLE_STATUSES or status >= 500

    def _read_body(self, resp: Any) -> str:
        try:
            raw = resp.read()
        except Exception as exc:
            return f"(could not read response body: {exc})"
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    def _request_with_retries(self, url: str, data: bytes, headers: Dict[str, str]):
        """POST one request, retrying transient failures twice with backoff.

        Returns ``(status_code, response_text)``. Connection-level failures
        that survive every attempt raise RuntimeError directly.
        """
        retries = 0
        while True:
            try:
                resp = self._http.open(
                    urllib.request.Request(url, data=data, headers=headers, method="POST"),
                    timeout=self._timeout,
                )
                status = int(getattr(resp, "status", 200) or 200)
                text = self._read_body(resp)
            except urllib.error.HTTPError as exc:
                status = int(getattr(exc, "code", 0) or 0)
                text = self._read_body(exc)
            except (urllib.error.URLError, OSError) as exc:
                status = 0
                text = str(exc)

            if self._should_retry(status) and retries < self._MAX_RETRIES:
                retries += 1
                time.sleep(0.5 * (2 ** (retries - 1)))  # 0.5s, then 1.0s
                continue
            if status == 0:
                raise RuntimeError(
                    f"EgoOpenAI request failed after {retries + 1} attempt(s): {text}"
                )
            return status, text

#-------------------------- public methods --------------------------------------

    def create(self, **payload: Any) -> "EgoOpenAI":
        """POST one chat-completions request.

        Accepts the same kwargs the providers' ``_build_payload()`` produced
        for the OpenAI SDK (``model``, ``messages``, ``max_tokens``,
        ``temperature``, ``reasoning_effort``, ``extra_body``, ...) and
        applies the SDK's request normalization before sending. Transient
        failures are retried twice with backoff. Returns an EgoOpenAI
        response node with SDK-style attribute access.
        """
        self._require_client()
        body = self._normalize_payload(payload)

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        data = json.dumps(body).encode("utf-8")

        status, text = self._request_with_retries(url, data, headers)
        if status >= 400:
            raise RuntimeError(f"EgoOpenAI API error {status}: {text}")
        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"EgoOpenAI API returned invalid JSON: {exc}")
        return self._wrap(parsed)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style access on a response node, with a default for missing keys."""
        self._require_data()
        if key in self._data:
            return self._wrap(self._data[key])
        return default

    def close(self) -> None:
        """Release the client. urllib keeps no persistent connections, so this
        is a no-op kept for OpenAI-SDK parity."""
        self._require_client()
