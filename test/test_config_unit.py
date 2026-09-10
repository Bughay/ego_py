"""Unit tests for the per-object config dict contract.

The legacy ``egoai.config.Config`` class is gone: the config schema now
lives in ``egoai/llm/config.py`` as the ``ConfigModel`` dataclass (covered
by test/test_config_model_unit.py), and every LLM-based object carries the
normalized plain ``config`` dict.

Runs entirely offline. Covers the per-object config wiring:

  - session saving is driven by the plain config dict on each LLM object
    (config["session_path"]), not by any Config class: an existing
    directory saves there; a missing key, a non-existent directory or
    conflicting paths raise RuntimeError; an empty workflow writes nothing;
    the dict is read at call time, so late mutations take effect.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from egoai.mlops import WorkflowSession
from test.fakes import FakeLLM


def simple_answer(content="done"):
    return {"reasoning": None, "content": content, "tool_calls": []}


def list_json_files(directory):
    return sorted(
        name for name in os.listdir(directory) if name.endswith(".json")
    )


# ------------------------- per-object config wiring --------------------------

class TestPerObjectConfigWiring(unittest.TestCase):
    """WorkflowSession takes no Config: the save directory comes from the
    config dict on each recorded LLM object."""

    def test_existing_session_path_saves_there(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLM(canned_responses=[simple_answer("hi")],
                          config={"session_path": tmp})
            with WorkflowSession(name="cfg") as session:
                llm.one_shot()

            self.assertEqual(session.directory, Path(tmp).resolve())
            self.assertTrue(session.path.is_file())
            self.assertEqual(str(session.path.parent), str(Path(tmp).resolve()))
            payload = json.loads(session.path.read_text(encoding="utf-8"))
            self.assertEqual(list(payload), ["1"])

    def test_missing_session_path_key_raises(self):
        llm = FakeLLM(canned_responses=[simple_answer("hi")])  # config={}
        with self.assertRaisesRegex(RuntimeError, "no 'session_path' key"):
            with WorkflowSession(name="nokey"):
                llm.one_shot()

    def test_nonexistent_session_path_raises_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no" / "such" / "dir"
            llm = FakeLLM(canned_responses=[simple_answer("hi")],
                          config={"session_path": str(missing)})
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                with WorkflowSession(name="missing"):
                    llm.one_shot()

            self.assertFalse(missing.exists())  # never auto-created

    def test_conflicting_session_paths_raise(self):
        with tempfile.TemporaryDirectory() as tmp_a, \
                tempfile.TemporaryDirectory() as tmp_b:
            llm_a = FakeLLM(canned_responses=[simple_answer("a")],
                            config={"session_path": tmp_a})
            llm_b = FakeLLM(canned_responses=[simple_answer("b")],
                            config={"session_path": tmp_b})
            with self.assertRaisesRegex(RuntimeError, "conflicting session_paths"):
                with WorkflowSession(name="conflict"):
                    llm_a.one_shot()
                    llm_b.one_shot()

    def test_empty_workflow_saves_nothing(self):
        with WorkflowSession(name="empty") as session:
            pass
        self.assertIsNone(session.directory)
        self.assertIsNone(session.path)
        self.assertEqual(session.steps, {})

    def test_object_config_read_at_call_time(self):
        """The session reads config["session_path"] from each object at
        record time, so setting it on the object before the call takes
        effect even when the session/decorator was created earlier."""
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLM(canned_responses=[simple_answer("ok")])  # {}
            decorated = WorkflowSession.capture("late-config")
            llm.config["session_path"] = tmp  # set before the call

            @decorated
            def workflow():
                llm.one_shot()

            workflow()
            self.assertEqual(len(list_json_files(tmp)), 1)


if __name__ == "__main__":
    unittest.main()
