"""Unit tests for ego_py.llm.models — the ConfigModel config schema.

Runs entirely offline. Covers:

  1. normalization      -> from_dict/to_dict round-trips; None -> {}; absent
                           keys stay absent; unknown str -> str keys survive
  2. validation         -> file must be a bool, agents.md/skills must be
                           path strings, known-key type violations raise
  3. context_manager    -> nested dict normalization unchanged
  4. file tools         -> config={"file": True} on an agent with a
                           directory loads build_file_tools into
                           tool_registry + the tools schema; False/absent
                           leaves tools untouched; no directory raises
  5. agents.md          -> recursive AGENTS.md scan injects every file's
                           contents into the system prompt (agents keep it
                           across run()); non-matching files ignored; empty
                           scan changes nothing
"""
import os
import tempfile
import unittest

from ego_py.agent.react import ReActAgent
from ego_py.llm.config import ConfigModel
from test.fakes import FakeLLM

FILE_TOOL_NAMES = {"ls", "read_file", "write_file", "edit_file",
                   "delete", "glob", "grep"}


class FakeReAct(ReActAgent, FakeLLM):
    """ReAct agent powered by canned FakeLLM responses."""


def simple_answer(content="done"):
    return {"reasoning": None, "content": content, "tool_calls": []}


def write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def schema_names(agent):
    return {tool["function"]["name"] for tool in (agent.tools or [])}


# ------------------------- 1. normalization ----------------------------------

class TestNormalization(unittest.TestCase):
    def test_none_becomes_empty_dict(self):
        self.assertEqual(ConfigModel.from_dict(None).to_dict(), {})

    def test_absent_keys_stay_absent(self):
        # Only keys the caller actually provided are emitted, so defaults
        # (file=False etc.) never materialize in the stored config.
        self.assertEqual(ConfigModel.from_dict({}).to_dict(), {})
        self.assertEqual(ConfigModel.from_dict({"file": False}).to_dict(),
                         {"file": False})
        self.assertEqual(ConfigModel.from_dict({"skills": None}).to_dict(),
                         {"skills": None})

    def test_full_round_trip(self):
        config = {
            "session_path": "/sessions",
            "skills": "/skills",
            "file": True,
            "agents.md": "/repo",
            "context_manager": {"summarize": 5},
            "env": "prod",
        }
        self.assertEqual(
            ConfigModel.from_dict(config).to_dict(),
            {
                "session_path": "/sessions",
                "skills": "/skills",
                "file": True,
                "agents.md": "/repo",
                "context_manager": {"summarize": 5, "max_iteration": None},
                "env": "prod",
            },
        )

    def test_unknown_str_keys_survive(self):
        self.assertEqual(ConfigModel.from_dict({"k": "v"}).to_dict(), {"k": "v"})

    def test_does_not_mutate_input(self):
        config = {"session_path": "/tmp/a"}
        ConfigModel.from_dict(config)
        self.assertEqual(config, {"session_path": "/tmp/a"})


# ------------------------- 2. validation -------------------------------------

class TestValidation(unittest.TestCase):
    def test_config_must_be_dict(self):
        for bad in ("nope", 42, ["a"], ("a",)):
            with self.subTest(config=bad):
                with self.assertRaisesRegex(TypeError, "config must be a dict"):
                    ConfigModel.from_dict(bad)

    def test_non_str_key_raises(self):
        with self.assertRaisesRegex(TypeError, "str keys to str values"):
            ConfigModel.from_dict({1: "x"})

    def test_unknown_key_value_must_be_str(self):
        for bad in (1, None, ["x"], {"k": "v"}):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(TypeError, "str keys to str values"):
                    ConfigModel.from_dict({"a": bad})

    def test_session_path_must_be_str(self):
        with self.assertRaisesRegex(TypeError, "session_path"):
            ConfigModel.from_dict({"session_path": None})

    def test_file_must_be_bool(self):
        for bad in (1, 0, "yes", None, [True]):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(TypeError, "file.*bool"):
                    ConfigModel.from_dict({"file": bad})

    def test_agents_md_must_be_path_string(self):
        for bad in (1, ["/repo"], {"p": "/repo"}):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(TypeError, "agents.md.*string path"):
                    ConfigModel.from_dict({"agents.md": bad})
        with self.assertRaisesRegex(ValueError, "non-empty path"):
            ConfigModel.from_dict({"agents.md": ""})

    def test_skills_still_validated(self):
        with self.assertRaisesRegex(TypeError, "skills.*string path"):
            ConfigModel.from_dict({"skills": 42})
        with self.assertRaisesRegex(ValueError, "non-empty path"):
            ConfigModel.from_dict({"skills": "   "})


# ------------------------- 3. context_manager --------------------------------

class TestContextManager(unittest.TestCase):
    def test_missing_keys_default_to_none(self):
        self.assertEqual(
            ConfigModel.from_dict({"context_manager": {"summarize": 100}}).to_dict(),
            {"context_manager": {"summarize": 100, "max_iteration": None}},
        )
        self.assertEqual(
            ConfigModel.from_dict({"context_manager": {}}).to_dict(),
            {"context_manager": {"summarize": None, "max_iteration": None}},
        )

    def test_non_dict_raises(self):
        for bad in ("x", 1, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(TypeError, "context_manager.*dict"):
                    ConfigModel.from_dict({"context_manager": bad})

    def test_bad_key_raises(self):
        with self.assertRaisesRegex(ValueError, "summarize.*max_iteration"):
            ConfigModel.from_dict({"context_manager": {"nope": 1}})

    def test_bad_value_raises(self):
        for bad in (0, -1, True, 1.5, "5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ConfigModel.from_dict({"context_manager": {"summarize": bad}})


# ------------------------- 4. file tools -------------------------------------

class TestFileTools(unittest.TestCase):
    def test_file_true_loads_tools_into_registry_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeReAct(model="fake-model", max_tokens=100,
                              directory=tmp, config={"file": True})
            self.assertTrue(FILE_TOOL_NAMES <= set(agent.tool_registry))
            self.assertTrue(FILE_TOOL_NAMES <= schema_names(agent))

    def test_file_false_leaves_tools_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeReAct(model="fake-model", max_tokens=100,
                              directory=tmp, config={"file": False})
            self.assertIsNone(agent.tool_registry)
            self.assertIsNone(agent.tools)

    def test_file_absent_leaves_tools_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeReAct(model="fake-model", max_tokens=100,
                              directory=tmp)
            self.assertIsNone(agent.tool_registry)
            self.assertIsNone(agent.tools)

    def test_file_true_merges_with_existing_registry(self):
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeReAct(model="fake-model", max_tokens=100,
                              directory=tmp, tool_registry={"add": add},
                              config={"file": True})
            self.assertEqual(set(agent.tool_registry), FILE_TOOL_NAMES | {"add"})
            self.assertEqual(schema_names(agent), FILE_TOOL_NAMES | {"add"})

    def test_file_true_without_directory_raises(self):
        with self.assertRaisesRegex(ValueError, "workspace directory"):
            FakeLLM(config={"file": True})

    def test_file_tools_are_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeReAct(model="fake-model", max_tokens=100,
                              directory=tmp, config={"file": True})
            result = agent.tool_registry["write_file"]("note.txt", "hello")
            self.assertIn("note.txt", result)
            self.assertEqual(agent.tool_registry["read_file"]("note.txt")
                             .splitlines()[-1], "hello")


# ------------------------- 5. agents.md ---------------------------------------

class TestAgentsMd(unittest.TestCase):
    def _make_tree(self):
        root = tempfile.mkdtemp()
        sub = os.path.join(root, "sub")
        os.makedirs(sub)
        write(os.path.join(root, "AGENTS.md"), "ROOT CONTEXT")
        write(os.path.join(sub, "AGENTS.md"), "SUB CONTEXT")
        write(os.path.join(sub, "other.md"), "IGNORED")
        write(os.path.join(root, "AGENT.md"), "ALSO IGNORED")
        return root

    def test_recursive_scan_injects_every_agents_md(self):
        root = self._make_tree()
        try:
            llm = FakeLLM(config={"agents.md": root})
            system = llm.memory[0]["content"]
            self.assertIn("ROOT CONTEXT", system)
            self.assertIn("SUB CONTEXT", system)
            self.assertIn("sub/AGENTS.md", system)  # relative path heading
            self.assertNotIn("IGNORED", system)
            self.assertNotIn("ALSO IGNORED", system)
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_empty_directory_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLM(config={"agents.md": tmp})
            self.assertEqual(llm._agents_prompt, "")
            self.assertEqual(llm.memory[0]["content"], "You are a test assistant.")

    def test_missing_directory_raises(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            FakeLLM(config={"agents.md": "/no/such/dir/anywhere"})

    def test_agent_keeps_agents_md_across_run(self):
        root = self._make_tree()
        try:
            agent = FakeReAct(model="fake-model", max_tokens=100,
                              config={"agents.md": root},
                              canned_responses=[simple_answer()])
            agent.run("task", max_steps=2)
            self.assertIn("ROOT CONTEXT", agent._react_prompt)
            self.assertIn("ROOT CONTEXT", agent.memory[0]["content"])
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_config_key_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLM(config={"agents.md": tmp})
            self.assertEqual(llm.config["agents.md"], tmp)


if __name__ == "__main__":
    unittest.main()
