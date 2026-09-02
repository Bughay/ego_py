"""Unit tests for the two factories.

Covers the creation abstractions, fully offline (placeholder API keys —
the provider clients are only constructed, never called):

  - LLM(model=...)          -> the right provider class (llm.py factory)
  - EgoAgent(agent=..., model=...) -> agent + provider combined (factory.py)
  - config forwarding       -> config={...} reaches providers and every
                               combined agent class
"""
import os
import tempfile
import unittest

from ego_py import EgoAgent, LLM
from ego_py.builtin_tools.file import build_file_tools
from ego_py.builtin_tools.math import build_math_tools
from ego_py.agent.planexecute import PlanExecuteAgent
from ego_py.agent.planreact import PlanReactAgent
from ego_py.agent.react import ReActAgent
from ego_py.llm.deepseek import DeepseekLLM
from ego_py.llm.grok import GrokLLM


def _set_keys():
    """Placeholder keys so provider clients can be built without prompting."""
    os.environ.setdefault("XAI_API_KEY", "test-key")
    os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")


class TestLLMFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _set_keys()

    def test_llm_returns_grok_instance(self):
        llm = LLM(model="grok-4.6", system_prompt="s", user_prompt="u")
        self.assertIsInstance(llm, GrokLLM)

    def test_llm_returns_deepseek_instance(self):
        llm = LLM(model="deepseek-v4-flash", system_prompt="s", user_prompt="u")
        self.assertIsInstance(llm, DeepseekLLM)

    def test_unknown_model_raises(self):
        with self.assertRaisesRegex(ValueError, "No provider registered"):
            LLM(model="gpt-4o", system_prompt="s", user_prompt="u")

    def test_non_string_model_raises(self):
        with self.assertRaises(ValueError):
            LLM(model=42, system_prompt="s", user_prompt="u")

    def test_direct_provider_still_works(self):
        llm = GrokLLM(model="grok-4.6", system_prompt="s", user_prompt="u")
        self.assertIsInstance(llm, GrokLLM)

    def test_llm_forwards_config_to_provider(self):
        config = {"session_path": "/some/dir"}
        llm = LLM(model="grok-4.6", system_prompt="s", user_prompt="u",
                  config=config)
        self.assertIsInstance(llm, GrokLLM)
        self.assertEqual(llm.config, config)

        llm = LLM(model="deepseek-v4-flash", system_prompt="s",
                  user_prompt="u", config=config)
        self.assertIsInstance(llm, DeepseekLLM)
        self.assertEqual(llm.config, config)


class TestEgoAgentFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _set_keys()

    def test_react_grok(self):
        agent = EgoAgent(agent="react", model="grok-4.6", max_tokens=100,
                         instruction="t")
        self.assertIsInstance(agent, ReActAgent)
        self.assertIsInstance(agent, GrokLLM)

    def test_react_deepseek(self):
        agent = EgoAgent(agent="react", model="deepseek-v4-flash",
                         max_tokens=100, instruction="t")
        self.assertIsInstance(agent, ReActAgent)
        self.assertIsInstance(agent, DeepseekLLM)

    def test_plan_execute_deepseek(self):
        agent = EgoAgent(agent="plan-execute", model="deepseek-v4-flash",
                         max_tokens=100, instruction="t")
        self.assertIsInstance(agent, PlanExecuteAgent)
        self.assertIsInstance(agent, DeepseekLLM)

    def test_plan_react_grok(self):
        agent = EgoAgent(agent="plan-react", model="grok-4.6",
                         max_tokens=100, instruction="t")
        self.assertIsInstance(agent, PlanReactAgent)
        self.assertIsInstance(agent, GrokLLM)

    def test_combined_class_name(self):
        agent = EgoAgent(agent="react", model="grok-4.6", max_tokens=100,
                         instruction="t")
        self.assertEqual(agent.__class__.__name__, "GrokLLMReActAgent")

    def test_reasoning_effort_forwarded_to_provider(self):
        """kwargs pass-through: every agent type reaches the provider."""
        for agent_kind in ("react", "plan-execute", "plan-react"):
            with self.subTest(agent=agent_kind):
                agent = EgoAgent(agent=agent_kind, model="deepseek-v4-flash",
                                 max_tokens=100, instruction="t",
                                 reasoning_effort="low")
                self.assertEqual(agent.reasoning_effort, "low")
                self.assertEqual(
                    agent._build_payload()["reasoning_effort"], "low"
                )

    def test_reasoning_effort_xhigh_forwarded_for_grok(self):
        agent = EgoAgent(agent="react", model="grok-4.6", max_tokens=100,
                         instruction="t", reasoning_effort="xhigh")
        self.assertEqual(agent.reasoning_effort, "xhigh")
        self.assertEqual(agent._build_payload()["reasoning_effort"], "xhigh")

    def test_reasoning_effort_invalid_rejected(self):
        with self.assertRaises(ValueError):
            EgoAgent(agent="react", model="deepseek-v4-flash",
                     max_tokens=100, instruction="t",
                     reasoning_effort="extreme")

    def test_unknown_agent_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown agent"):
            EgoAgent(agent="nope", model="grok-4.6", max_tokens=100)

    def test_config_forwarded_for_all_agent_kinds(self):
        """config passes through every agent/provider combination and lands
        on the combined instance as a dict of str -> str."""
        config = {"session_path": "/some/dir"}
        for agent_kind in ("react", "plan-execute", "plan-react"):
            for model in ("deepseek-v4-flash", "grok-4.6"):
                with self.subTest(agent=agent_kind, model=model):
                    agent = EgoAgent(agent=agent_kind, model=model,
                                     max_tokens=100, instruction="t",
                                     config=config)
                    self.assertEqual(agent.config, config)

    def test_unknown_model_raises(self):
        with self.assertRaisesRegex(ValueError, "No provider registered"):
            EgoAgent(agent="react", model="gpt-4o", max_tokens=100)

    def test_no_tools_registered_automatically(self):
        """directory must NOT auto-add file tools; registry is explicit-only."""
        agent = EgoAgent(agent="react", model="deepseek-v4-flash",
                         max_tokens=100, instruction="t")
        self.assertIsNone(agent.tool_registry)
        self.assertIsNone(agent.tools)

        with tempfile.TemporaryDirectory() as tmp:
            agent = EgoAgent(agent="react", model="deepseek-v4-flash",
                             max_tokens=100, instruction="t", directory=tmp)
            self.assertIsNone(agent.tool_registry)
            self.assertIsNone(agent.tools)

    def test_explicit_file_and_math_tools_registered(self):
        """Passing build_file_tools + build_math_tools by hand registers all 11."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = {**build_math_tools(), **build_file_tools(tmp)}
            agent = EgoAgent(agent="react", model="deepseek-v4-flash",
                             max_tokens=100, instruction="t",
                             directory=tmp, tool_registry=registry)
            self.assertEqual(
                set(agent.tool_registry),
                {"add", "subtract", "multiply", "divide",
                 "ls", "read_file", "write_file", "edit_file",
                 "delete", "glob", "grep"},
            )
            self.assertEqual(len(agent.tools), 11)

    def test_math_tools_execute(self):
        registry = build_math_tools()
        self.assertEqual(registry["add"](2, 3), 5)
        self.assertEqual(registry["subtract"](7, 4), 3)
        self.assertEqual(registry["multiply"](6, 6), 36)
        self.assertEqual(registry["divide"](10, 2), 5.0)
        self.assertEqual(registry["divide"](1, 0), "Error: division by zero")


class TestProviderConfigForwarding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _set_keys()

    def test_direct_providers_accept_config(self):
        llm = DeepseekLLM(model="deepseek-v4-flash", system_prompt="s",
                          user_prompt="u", config={"k": "v"})
        self.assertEqual(llm.config, {"k": "v"})

        llm = GrokLLM(model="grok-4.6", system_prompt="s", user_prompt="u",
                      config={"k": "v"})
        self.assertEqual(llm.config, {"k": "v"})


if __name__ == "__main__":
    unittest.main()
