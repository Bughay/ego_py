"""Unit tests for config["print_output"] — pretty terminal output of one_shot().

Runs entirely offline using the canned FakeLLM double. Covers:

  1. default off        -> no output at all (absent key and explicit False)
  2. enabled            -> header + [Reasoning] + [Content] + [Tool calls]
  3. suppression        -> internal extract()/summarize() calls never print
  4. agent path         -> FakeReAct.run() prints the model turn when enabled
"""
import contextlib
import io
import unittest

from egoai.agent.react import ReActAgent
from test.fakes import FakeLLM


class FakeReAct(ReActAgent, FakeLLM):
    """ReAct agent powered by canned FakeLLM responses."""


def canned(reasoning="thinking...", content="the answer", tool_calls=None):
    return {
        "reasoning": reasoning,
        "content": content,
        "tool_calls": tool_calls or [],
    }


def capture(fn, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


# ------------------------- 1. default off ------------------------------------

class TestDefaultOff(unittest.TestCase):
    def test_no_config_prints_nothing(self):
        llm = FakeLLM(canned_responses=[canned()])
        _, out = capture(llm.one_shot)
        self.assertEqual(out, "")

    def test_explicit_false_prints_nothing(self):
        llm = FakeLLM(canned_responses=[canned()],
                      config={"print_output": False})
        _, out = capture(llm.one_shot)
        self.assertEqual(out, "")


# ------------------------- 2. enabled ----------------------------------------

class TestEnabled(unittest.TestCase):
    def test_prints_all_three_sections(self):
        tool_calls = [
            {"id": "call_1", "name": "subtract", "arguments": {"a": 20, "b": 4}}
        ]
        llm = FakeLLM(
            canned_responses=[canned(reasoning="need math", content="4",
                                     tool_calls=tool_calls)],
            config={"print_output": True},
        )
        result, out = capture(llm.one_shot)
        self.assertEqual(result["content"], "4")
        self.assertIn("=== FakeLLM output ===", out)
        self.assertIn("[Reasoning]\nneed math", out)
        self.assertIn("[Content]\n4", out)
        self.assertIn("[Tool calls]", out)
        self.assertIn("-> subtract(a=20, b=4)", out)

    def test_tool_only_response_omits_empty_sections(self):
        tool_calls = [
            {"id": "call_1", "name": "ls", "arguments": {"path": "."}}
        ]
        llm = FakeLLM(
            canned_responses=[canned(reasoning=None, content=None,
                                     tool_calls=tool_calls)],
            config={"print_output": True},
        )
        _, out = capture(llm.one_shot)
        self.assertIn("[Tool calls]", out)
        self.assertIn('-> ls(path=".")', out)
        self.assertNotIn("[Reasoning]", out)
        self.assertNotIn("[Content]", out)

    def test_empty_response_placeholder(self):
        llm = FakeLLM(canned_responses=[canned(reasoning=None, content=None)],
                      config={"print_output": True})
        _, out = capture(llm.one_shot)
        self.assertIn("(empty response)", out)


# ------------------------- 3. internal call suppression ----------------------

class TestInternalCallsSuppressed(unittest.TestCase):
    def test_summarize_does_not_print(self):
        llm = FakeLLM(canned_responses=[canned(content="a summary")],
                      config={"print_output": True})
        _, out = capture(llm.summarize, "some context", 100)
        self.assertEqual(out, "")

    def test_extract_does_not_print(self):
        llm = FakeLLM(canned_responses=[canned(content='{"a": 1}')],
                      config={"print_output": True})
        _, out = capture(llm.extract, {"a": "the a"})
        self.assertEqual(out, "")


# ------------------------- 4. agent path -------------------------------------

class TestAgentPath(unittest.TestCase):
    def test_run_prints_the_model_turn(self):
        agent = FakeReAct(
            model="fake-model", max_tokens=100,
            canned_responses=[canned(content="final answer")],
            config={"print_output": True},
        )
        _, out = capture(agent.run, "do the task")
        self.assertIn("FakeReAct output", out)
        self.assertIn("final answer", out)


if __name__ == "__main__":
    unittest.main()
