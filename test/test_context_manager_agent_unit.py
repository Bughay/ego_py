"""Agent-level tests for config["context_manager"].

Verifies that the automatic context management wired into one_shot() actually
fires while an agent runs, using a FakeLLM-powered ReAct agent.
"""
import unittest

from egoai.agent.react import ReActAgent
from test.fakes import FakeLLM


class FakeReAct(ReActAgent, FakeLLM):
    """ReAct agent powered by canned FakeLLM responses."""


def noop() -> str:
    """No-op tool used to keep the ReAct loop running."""
    return "ok"


def tool_call(content=""):
    return {
        "reasoning": None,
        "content": content,
        "tool_calls": [{"id": "call_1", "name": "noop", "arguments": {}}],
    }


def answer(content):
    return {"reasoning": None, "content": content, "tool_calls": []}


class TestAgentContextManager(unittest.TestCase):
    def make(self, config, canned, tokens_per_call=0):
        return FakeReAct(
            model="fake-model",
            max_tokens=100,
            instruction="test",
            tool_registry={"noop": noop},
            canned_responses=list(canned),
            tokens_per_call=tokens_per_call,
            config=config,
        )

    def test_summarize_fires_during_run(self):
        agent = self.make(
            {"context_manager": {"summarize": 2, "max_iteration": None}},
            canned=[tool_call(), answer("summary"), answer("done")],
            tokens_per_call=2,
        )

        result = agent.run("do the task", max_steps=10)

        self.assertEqual(result["content"], "done")
        self.assertEqual(agent.memory[0]["role"], "system")
        self.assertEqual(agent.memory[1], {"role": "user", "content": "summary"})
        self.assertEqual(agent.tokens_used, 2)

    def test_trim_fires_during_run(self):
        agent = self.make(
            {"context_manager": {"summarize": None, "max_iteration": 2}},
            canned=[tool_call(), tool_call(), tool_call(), answer("done")],
        )

        agent.run("task that should get trimmed away", max_steps=10)

        contents = [m["content"] for m in agent.memory]
        self.assertNotIn("task that should get trimmed away", contents)
        self.assertEqual(agent.memory[0]["role"], "system")
        self.assertIn("done", contents)


if __name__ == "__main__":
    unittest.main()
