"""Test doubles for the LLM wrappers.

These stand-ins let the unit tests exercise every layer of the framework
deterministically, without touching the network or needing real API keys:

- StrictLLM   : minimal concrete LLM whose __init__ passes everything
                straight through to LLM.__init__ — used to test the base
                class's own validation rules.
- FakeLLM     : concrete LLM whose one_shot() pops canned responses off a
                queue and records what it observed (memory snapshot,
                max_tokens, response_format) for later assertions.
- FakeClient /
  FakeMessage /
  FakeToolCall: mirrors of the OpenAI SDK response objects used by
                DeepseekLLM.one_shot / GrokLLM.one_shot.
"""
from egoai.llm.base import BaseLLM


class StrictLLM(BaseLLM):
    """Minimal concrete LLM used to test base-class validation directly.

    Unlike FakeLLM it does NOT inject default prompts, so constructor
    validation errors surface exactly as base.py defines them.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("model", "fake-model")
        super().__init__(**kwargs)

    def _get_api_key(self) -> str:
        return "fake-key"

    def _create_client(self):
        return None

    def _build_payload(self):
        return {}

    def one_shot(self):
        result = {"reasoning": None, "content": "", "tool_calls": []}
        self._print_output(result)
        return result


class FakeLLM(BaseLLM):
    """Deterministic LLM stub whose one_shot() pops canned responses off a queue.

    Records (in order) what each one_shot call observed:
      - seen_memories        : snapshot of the memory that was sent
      - seen_max_tokens      : max_tokens at call time
      - seen_response_formats: response_format at call time
    """

    def __init__(self, canned_responses=None, tokens_per_call=0, **kwargs):
        kwargs.setdefault("model", "fake-model")
        if "memory" not in kwargs:
            kwargs.setdefault("system_prompt", "You are a test assistant.")
            kwargs.setdefault("user_prompt", "Hello.")
        super().__init__(**kwargs)
        self.tokens_per_call = tokens_per_call
        self.canned_responses = list(canned_responses or [])
        self.one_shot_calls = 0
        self.seen_memories = []
        self.seen_max_tokens = []
        self.seen_response_formats = []

    def _get_api_key(self) -> str:
        return "fake-key"

    def _create_client(self):
        return None

    def _build_payload(self):
        return {"model": self.model, "messages": self.memory}

    def one_shot(self):
        self._manage_context()

        self.one_shot_calls += 1
        self.seen_memories.append([dict(m) for m in self.memory])
        self.seen_max_tokens.append(self.max_tokens)
        self.seen_response_formats.append(self.response_format)

        if self.canned_responses:
            canned = self.canned_responses.pop(0)
        else:
            canned = {"reasoning": None, "content": "", "tool_calls": []}
        # Mirror the real providers: append the assistant turn to memory.
        self.memory.append({"role": "assistant", "content": canned.get("content")})
        self._tokens_used += self.tokens_per_call
        self._print_output(canned)
        return canned


# ---------------------------------------------------------------------------
# Fake OpenAI-SDK-shaped objects for provider one_shot parsing tests.
# ---------------------------------------------------------------------------

class FakeFunction:
    """Mirrors message.tool_calls[i].function"""

    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # raw JSON string, like the real SDK


class FakeToolCall:
    """Mirrors message.tool_calls[i]"""

    def __init__(self, id, name, arguments):
        self.id = id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    """Mirrors response.choices[0].message"""

    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


class FakeUsage:
    """Mirrors response.usage (total_tokens required, prompt/completion optional)."""

    def __init__(self, total_tokens=0, prompt_tokens=0, completion_tokens=0):
        self.total_tokens = total_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeChoice:
    """Mirrors response.choices[0]"""

    def __init__(self, message):
        self.message = message


class FakeResponse:
    """Mirrors the OpenAI SDK response object."""

    def __init__(self, message, usage=None):
        self.choices = [FakeChoice(message)]
        self.usage = usage


class FakeCompletions:
    """Queue-backed stand-in for client.chat.completions."""

    def __init__(self, messages, usages=None):
        self._messages = list(messages)
        self._usages = list(usages or [])
        self.calls = []  # every payload passed to create()

    def create(self, **payload):
        self.calls.append(payload)
        message = self._messages.pop(0)
        usage = self._usages.pop(0) if self._usages else None
        return FakeResponse(message, usage)


class FakeClient:
    """Queue-backed stand-in for the whole provider client.

    Mirrors the OpenAI SDK layout the wrappers rely on:
        client.chat.completions.create(**payload)
    """

    def __init__(self, messages, usages=None):
        self.chat = type("FakeChat", (), {"completions": FakeCompletions(messages, usages)})()
