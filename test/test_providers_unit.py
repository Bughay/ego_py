"""Unit tests for DeepseekLLM and GrokLLM (provider-specific layers).

No network involved: FakeClient stands in for the OpenAI SDK client so we
can verify payload construction and one_shot response parsing offline.
Only the DEEPSEEK_API_KEY / XAI_API_KEY environment variables are set so
that the providers can be constructed; _get_api_key() raises a RuntimeError
when the key is missing instead of prompting.

Covered:
  1. Model allowlists (the "pick deepseek, grok or who" behaviour)
  2. reasoning_effort validation (incl. Grok's "xhigh")
  3. _build_payload shape (thinking on/off, temperature handling, optional fields)
  4. one_shot parsing (memory append, tool_calls, reasoning, return shape)
  5. API key resolution from the environment
"""
import os
import unittest

from ego_py import DeepseekLLM, GrokLLM
from test.fakes import FakeClient, FakeMessage, FakeToolCall, FakeUsage


class DeepseekLLMTests(unittest.TestCase):
    ALLOWED = ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp"]

    def setUp(self):
        self._old = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = "test-key"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = self._old

    def make(self, model="deepseek-v4-flash", **kw):
        kw.setdefault("system_prompt", "sys")
        kw.setdefault("user_prompt", "hi")
        return DeepseekLLM(model=model, **kw)

    # 1. model allowlist ----------------------------------------------------
    def test_allowed_models(self):
        for m in self.ALLOWED:
            self.assertEqual(self.make(model=m).model, m)

    def test_rejects_unknown_models(self):
        for bad in ("gpt-4o", "deepseek-v3", "", "grok-4.6"):
            with self.assertRaises(ValueError):
                self.make(model=bad)

    # 2. reasoning_effort ---------------------------------------------------
    def test_reasoning_effort_allowed(self):
        for level in (None, "low", "medium", "high"):
            self.assertEqual(self.make(reasoning_effort=level).reasoning_effort, level)

    def test_reasoning_effort_xhigh_rejected(self):
        # "xhigh" is Grok-only
        with self.assertRaises(ValueError):
            self.make(reasoning_effort="xhigh")

    # 3. payload ------------------------------------------------------------
    def test_payload_thinking_enabled(self):
        llm = self.make(reasoning_effort="low")
        payload = llm._build_payload()
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["messages"], llm.memory)
        self.assertIn("temperature", payload)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["extra_body"], {"thinking": {"type": "enabled"}})

    def test_payload_thinking_disabled_by_default(self):
        payload = self.make()._build_payload()
        self.assertIsNone(payload["reasoning_effort"])
        self.assertEqual(payload["extra_body"], {"thinking": {"type": "disabled"}})

    def test_payload_optional_fields(self):
        bare = self.make()._build_payload()
        self.assertNotIn("response_format", bare)
        self.assertNotIn("tools", bare)
        self.assertNotIn("tool_choice", bare)

        llm = self.make(
            response_format={"type": "json_object"},
            tools=[{"type": "function"}],
            tool_choice="auto",
        )
        payload = llm._build_payload()
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["tools"], [{"type": "function"}])
        self.assertEqual(payload["tool_choice"], "auto")

    # 4. one_shot parsing ---------------------------------------------------
    def test_one_shot_parses_response(self):
        llm = self.make()
        llm._client = FakeClient([FakeMessage(
            content="Hi there",
            reasoning_content="thinking hard",
            tool_calls=[FakeToolCall("call_1", "add", '{"a": 1, "b": 2}')],
        )])
        result = llm.one_shot()
        self.assertEqual(result["content"], "Hi there")
        self.assertEqual(result["reasoning"], "thinking hard")
        self.assertEqual(
            result["tool_calls"],
            [{"id": "call_1", "name": "add", "arguments": {"a": 1, "b": 2}}],
        )
        self.assertEqual(set(result.keys()), {"reasoning", "content", "tool_calls"})
        # assistant turn appended to memory with the raw arguments string
        last = llm.memory[-1]
        self.assertEqual(last["content"], "Hi there")
        self.assertEqual(last["tool_calls"][0]["function"]["arguments"], '{"a": 1, "b": 2}')

    def test_one_shot_no_tools(self):
        llm = self.make()
        llm._client = FakeClient([FakeMessage(content="ok")])
        result = llm.one_shot()
        self.assertEqual(result["tool_calls"], [])
        self.assertIsNone(result["reasoning"])
        self.assertNotIn("tool_calls", llm.memory[-1])

    def test_one_shot_malformed_arguments_fall_back(self):
        llm = self.make()
        llm._client = FakeClient([FakeMessage(
            content="",
            tool_calls=[FakeToolCall("call_1", "add", "not json")],
        )])
        result = llm.one_shot()
        self.assertEqual(result["tool_calls"][0]["arguments"], {})

    def test_one_shot_accumulates_usage(self):
        llm = self.make()
        llm._client = FakeClient([FakeMessage(content="ok")], usages=[FakeUsage(42)])
        self.assertEqual(llm.tokens_used, 0)
        llm.one_shot()
        self.assertEqual(llm.tokens_used, 42)

    def test_one_shot_without_usage_keeps_counter(self):
        llm = self.make()
        llm._client = FakeClient([FakeMessage(content="ok")])
        llm.one_shot()
        self.assertEqual(llm.tokens_used, 0)

    # 5. API key ------------------------------------------------------------
    def test_get_api_key_from_env(self):
        self.assertEqual(self.make()._get_api_key(), "test-key")

    def test_get_api_key_missing_raises(self):
        os.environ.pop("DEEPSEEK_API_KEY", None)
        with self.assertRaises(RuntimeError):
            self.make()._get_api_key()

    def test_get_api_key_empty_raises(self):
        os.environ["DEEPSEEK_API_KEY"] = ""
        with self.assertRaises(RuntimeError):
            self.make()._get_api_key()


class GrokLLMTests(unittest.TestCase):
    ALLOWED = ["grok-4.6", "grok-4.6-fast", "grok-4.6-non-reasoning"]

    def setUp(self):
        self._old = os.environ.get("XAI_API_KEY")
        os.environ["XAI_API_KEY"] = "test-key"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("XAI_API_KEY", None)
        else:
            os.environ["XAI_API_KEY"] = self._old

    def make(self, model="grok-4.6", **kw):
        kw.setdefault("system_prompt", "sys")
        kw.setdefault("user_prompt", "hi")
        return GrokLLM(model=model, **kw)

    # 1. model allowlist ----------------------------------------------------
    def test_allowed_models(self):
        for m in self.ALLOWED:
            self.assertEqual(self.make(model=m).model, m)

    def test_rejects_unknown_models(self):
        for bad in ("gpt-4o", "grok-4", "", "deepseek-v4-flash"):
            with self.assertRaises(ValueError):
                self.make(model=bad)

    # 2. reasoning_effort ---------------------------------------------------
    def test_reasoning_effort_allowed(self):
        for level in (None, "low", "medium", "high", "xhigh"):
            self.assertEqual(self.make(reasoning_effort=level).reasoning_effort, level)

    def test_reasoning_effort_invalid_rejected(self):
        with self.assertRaises(ValueError):
            self.make(reasoning_effort="extreme")

    # 3. payload ------------------------------------------------------------
    def test_default_temperature_is_one(self):
        self.assertEqual(self.make().temperature, 1.0)

    def test_payload_omits_temperature(self):
        payload = self.make()._build_payload()
        self.assertNotIn("temperature", payload)
        self.assertNotIn("reasoning_effort", payload)  # absent when None

    def test_payload_omits_temperature_even_when_set(self):
        # Grok fixes temperature at 1.0 and never sends it
        llm = self.make(temperature=0.7)
        self.assertEqual(llm.temperature, 0.7)  # stored...
        self.assertNotIn("temperature", llm._build_payload())  # ...but not sent

    def test_payload_reasoning_effort_included(self):
        payload = self.make(reasoning_effort="xhigh")._build_payload()
        self.assertEqual(payload["reasoning_effort"], "xhigh")

    def test_payload_optional_fields(self):
        bare = self.make()._build_payload()
        self.assertNotIn("response_format", bare)
        self.assertNotIn("tools", bare)
        self.assertNotIn("tool_choice", bare)

        llm = self.make(
            response_format={"type": "json_object"},
            tools=[{"type": "function"}],
            tool_choice="auto",
        )
        payload = llm._build_payload()
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["tools"], [{"type": "function"}])
        self.assertEqual(payload["tool_choice"], "auto")

    # 4. one_shot parsing ---------------------------------------------------
    def test_one_shot_parses_response(self):
        llm = self.make()
        llm._client = FakeClient([FakeMessage(
            content="Hi there",
            reasoning_content="grok reasoning",
            tool_calls=[FakeToolCall("call_1", "add", '{"a": 5, "b": 7}')],
        )])
        result = llm.one_shot()
        self.assertEqual(result["content"], "Hi there")
        self.assertEqual(result["reasoning"], "grok reasoning")
        self.assertEqual(
            result["tool_calls"],
            [{"id": "call_1", "name": "add", "arguments": {"a": 5, "b": 7}}],
        )
        self.assertEqual(set(result.keys()), {"reasoning", "content", "tool_calls"})
        last = llm.memory[-1]
        self.assertEqual(last["content"], "Hi there")
        self.assertEqual(last["tool_calls"][0]["function"]["arguments"], '{"a": 5, "b": 7}')

    def test_one_shot_no_tools(self):
        llm = self.make()
        llm._client = FakeClient([FakeMessage(content="ok")])
        result = llm.one_shot()
        self.assertEqual(result["tool_calls"], [])
        self.assertIsNone(result["reasoning"])
        self.assertNotIn("tool_calls", llm.memory[-1])

    def test_one_shot_accumulates_usage(self):
        llm = self.make()
        llm._client = FakeClient([FakeMessage(content="ok")], usages=[FakeUsage(7)])
        self.assertEqual(llm.tokens_used, 0)
        llm.one_shot()
        self.assertEqual(llm.tokens_used, 7)

    def test_one_shot_without_usage_keeps_counter(self):
        llm = self.make()
        llm._client = FakeClient([FakeMessage(content="ok")])
        llm.one_shot()
        self.assertEqual(llm.tokens_used, 0)

    # 5. API key ------------------------------------------------------------
    def test_get_api_key_from_env(self):
        self.assertEqual(self.make()._get_api_key(), "test-key")

    def test_get_api_key_missing_raises(self):
        os.environ.pop("XAI_API_KEY", None)
        with self.assertRaises(RuntimeError):
            self.make()._get_api_key()

    def test_get_api_key_empty_raises(self):
        os.environ["XAI_API_KEY"] = ""
        with self.assertRaises(RuntimeError):
            self.make()._get_api_key()


if __name__ == "__main__":
    unittest.main()
