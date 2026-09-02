"""Unit tests for ego_py.mlops.sessions.WorkflowSession.

Runs entirely offline: FakeLLM supplies canned model responses and all
session files are written into temporary directories, so no network or API
keys are needed. Covers:

  1. standalone LLM calls -> numbered llm_call steps
  2. agent run()           -> one agent_run step with metadata/memory/llm_calls
  3. a 3-agent workflow    -> one JSON with numbered steps "1", "2", "3"
  4. decorator API         -> one file per decorated workflow call
  5. nested run() calls    -> PlanReact still produces exactly ONE step
  6. two workflows         -> two separate session files
  7. exceptions            -> partial session saved, exception re-raised
  8. patch restoration     -> original methods restored after the session
  9. tool executions       -> tool_events recorded with observations
  10. standalone onestep() -> tool events attach to the llm_call step
  11. sanitization         -> PlanExecute dataclass -> plain dict
  12. decorator concurrency -> each call gets its own session file
  13. per-object config contract -> directory comes from each recorded
      object's config["session_path"]; missing/invalid/conflicting configs
      raise RuntimeError, and an empty workflow writes nothing.
"""
import json
import os
import tempfile
import threading
import unittest

from ego_py.agent.planexecute import PlanExecuteAgent
from ego_py.agent.planreact import PlanReactAgent
from ego_py.agent.react import ReActAgent
from ego_py.llm.base import BaseLLM
from ego_py.llm.deepseek import DeepseekLLM
from ego_py.mlops import WorkflowSession
from test.fakes import FakeLLM


# ------------------------- offline agent fixtures ---------------------------

class FakeReAct(ReActAgent, FakeLLM):
    """ReAct agent powered by canned FakeLLM responses."""


class FakePlanExecute(PlanExecuteAgent, FakeLLM):
    """Plan-Execute agent powered by canned FakeLLM responses."""


class FakePlanReact(PlanReactAgent, FakeLLM):
    """Plan-React agent powered by canned FakeLLM responses."""


class NoConfigLLM(FakeLLM):
    """FakeLLM whose config attribute is removed after construction, to
    simulate a recorded object that never got a config from BaseLLM."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.config


def add_tool(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def simple_answer(content="done"):
    return {"reasoning": None, "content": content, "tool_calls": []}


# planner answer + extraction JSON + one answer per execute step (2 steps)
PLAN_CANNED = [
    {"reasoning": None, "content": "Here is the plan.", "tool_calls": []},
    {"reasoning": None, "content": json.dumps(
        {"plan": "Do the task.", "execute_steps": ["First step.", "Second step."]}
    ), "tool_calls": []},
    simple_answer("first done"),
    simple_answer("second done"),
]


def list_json_files(directory):
    return sorted(
        name for name in os.listdir(directory) if name.endswith(".json")
    )


def load_session_file(directory, filename=None):
    names = list_json_files(directory)
    assert names, "no session file written"
    path = os.path.join(directory, filename or names[0])
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def make_react(canned, session_path):
    return FakeReAct(
        model="fake-model", max_tokens=100, instruction="test",
        canned_responses=list(canned),
        config={"session_path": session_path},
    )


def make_plan_execute(canned, session_path):
    return FakePlanExecute(
        model="fake-model", max_tokens=100, instruction="test",
        canned_responses=list(canned),
        config={"session_path": session_path},
    )


def make_plan_react(canned, session_path):
    return FakePlanReact(
        model="fake-model", max_tokens=100, instruction="test",
        canned_responses=list(canned),
        config={"session_path": session_path},
    )


# ------------------------- 1. standalone LLM calls --------------------------

class TestStandaloneLLMCall(unittest.TestCase):
    def test_single_llm_call_becomes_numbered_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLM(canned_responses=[simple_answer("hi")],
                          config={"session_path": tmp})
            with WorkflowSession(name="solo") as session:
                llm.one_shot()

            self.assertEqual(list(session.steps), ["1"])
            record = session.steps["1"]
            self.assertEqual(record["type"], "llm_call")
            self.assertEqual(record["step"], 1)
            self.assertEqual(len(record["llm_calls"]), 1)
            self.assertEqual(record["llm_calls"][0]["output"]["content"], "hi")
            self.assertTrue(record["memory"])
            self.assertIn("model", record["metadata"])

            payload = load_session_file(tmp)
            self.assertEqual(list(payload), ["1"])
            self.assertEqual(payload["1"]["type"], "llm_call")

    def test_manual_save_and_auto_save_share_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLM(canned_responses=[simple_answer("hi")],
                          config={"session_path": tmp})
            with WorkflowSession(name="manual") as session:
                llm.one_shot()
                session.save()
            self.assertEqual(len(list_json_files(tmp)), 1)


# ------------------------- 2. agent run() steps -----------------------------

class TestAgentRunStep(unittest.TestCase):
    def test_agent_run_creates_one_agent_run_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_react([simple_answer("final answer")], tmp)
            with WorkflowSession(name="one_agent") as session:
                agent.run("do the task", max_steps=5)

            self.assertEqual(list(session.steps), ["1"])
            record = session.steps["1"]
            self.assertEqual(record["type"], "agent_run")
            self.assertEqual(record["agent"], "FakeReAct")
            self.assertEqual(record["task"], "do the task")
            self.assertEqual(record["metadata"]["agent_type"], "react")
            self.assertEqual(record["metadata"]["max_steps"], 5)
            self.assertEqual(len(record["llm_calls"]), 1)
            self.assertEqual(record["memory"][0]["role"], "system")
            self.assertIn("result", record)


# ------------------------- 3. multi-agent workflow ---------------------------

class TestThreeAgentWorkflow(unittest.TestCase):
    def test_three_agents_become_steps_1_2_3(self):
        with tempfile.TemporaryDirectory() as tmp:

            @WorkflowSession.capture("three")
            def workflow():
                make_react([simple_answer("react done")], tmp).run("task")
                make_plan_execute(PLAN_CANNED, tmp).run("task")
                make_plan_react(PLAN_CANNED, tmp).run("task")

            workflow()

            payload = load_session_file(tmp)
            self.assertEqual(list(payload), ["1", "2", "3"])
            self.assertEqual(payload["1"]["metadata"]["agent_type"], "react")
            self.assertEqual(payload["2"]["metadata"]["agent_type"], "plan-execute")
            self.assertEqual(payload["3"]["metadata"]["agent_type"], "plan-react")
            self.assertEqual(payload["1"]["task"], "task")
            self.assertEqual(len(payload["2"]["llm_calls"]), 4)
            self.assertEqual(len(payload["3"]["llm_calls"]), 4)


# ------------------------- 4. decorator API ---------------------------------

class TestDecoratorAPI(unittest.TestCase):
    def test_decorator_writes_one_file_and_tags_function(self):
        with tempfile.TemporaryDirectory() as tmp:
            ran = []

            @WorkflowSession.capture("deco")
            def workflow():
                ran.append(True)
                make_react([simple_answer("ok")], tmp).run("task")

            workflow()

            self.assertEqual(ran, [True])
            self.assertEqual(workflow._workflow_session.name, "deco")
            self.assertEqual(len(list_json_files(tmp)), 1)
            payload = load_session_file(tmp)
            self.assertEqual(list(payload), ["1"])


# ------------------------- 5. nested run() deduplication --------------------

class TestNestedRuns(unittest.TestCase):
    def test_planreact_internal_runs_share_one_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_plan_react(PLAN_CANNED, tmp)
            with WorkflowSession(name="planreact") as session:
                agent.run("task")

            self.assertEqual(list(session.steps), ["1"])
            record = session.steps["1"]
            self.assertEqual(record["type"], "agent_run")
            # plan (2 LLM calls) + 2 ReAct sub-runs (1 call each) = 4 calls
            self.assertEqual(len(record["llm_calls"]), 4)


# ------------------------- 12. decorator concurrency -------------------------

class TestDecoratorConcurrency(unittest.TestCase):
    def test_decorated_workflow_called_concurrently_from_two_threads(self):
        """Two threads calling the same decorated function must each get
        their own session file, with no cross-thread state corruption."""
        with tempfile.TemporaryDirectory() as tmp:
            barrier = threading.Barrier(2)
            errors = []

            @WorkflowSession.capture("concurrent")
            def workflow():
                barrier.wait()
                make_react([simple_answer("ok")], tmp).run("task")

            def call():
                try:
                    workflow()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=call) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            files = list_json_files(tmp)
            self.assertEqual(len(files), 2)
            for filename in files:
                payload = load_session_file(tmp, filename)
                self.assertEqual(list(payload), ["1"])

    def test_decorated_workflow_called_twice_sequentially_writes_two_files(self):
        """Each call of a decorated workflow is an independent session:
        two calls must produce two files, both numbered from "1"."""
        with tempfile.TemporaryDirectory() as tmp:

            @WorkflowSession.capture("twice")
            def workflow():
                make_react([simple_answer("ok")], tmp).run("task")

            workflow()
            workflow()

            files = list_json_files(tmp)
            self.assertEqual(len(files), 2)
            for filename in files:
                payload = load_session_file(tmp, filename)
                self.assertEqual(list(payload), ["1"])


# ------------------------- 6. two workflows -> two sessions ------------------

class TestTwoWorkflows(unittest.TestCase):
    def test_two_decorated_workflows_write_two_files(self):
        with tempfile.TemporaryDirectory() as tmp:

            @WorkflowSession.capture("wf_a")
            def workflow_a():
                make_react([simple_answer("a")], tmp).run("task")

            @WorkflowSession.capture("wf_b")
            def workflow_b():
                make_react([simple_answer("b")], tmp).run("task")

            workflow_a()
            workflow_b()

            files = list_json_files(tmp)
            self.assertEqual(len(files), 2)
            self.assertTrue(files[0].startswith("wf_a-") or files[1].startswith("wf_a-"))
            self.assertTrue(files[0].startswith("wf_b-") or files[1].startswith("wf_b-"))
            for filename in files:
                payload = load_session_file(tmp, filename)
                self.assertEqual(list(payload), ["1"])


# ------------------------- 7. exception handling -----------------------------

class TestExceptionHandling(unittest.TestCase):
    def test_exception_saves_partial_session_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLM(canned_responses=[simple_answer("x")],
                          config={"session_path": tmp})
            with self.assertRaises(RuntimeError):
                with WorkflowSession(name="boom"):
                    llm.one_shot()
                    raise RuntimeError("kaboom")

            self.assertEqual(len(list_json_files(tmp)), 1)
            payload = load_session_file(tmp)
            self.assertEqual(list(payload), ["1"])
            self.assertEqual(payload["1"]["type"], "llm_call")

    def test_nested_sessions_are_rejected(self):
        with self.assertRaises(RuntimeError):
            with WorkflowSession(name="outer"):
                with WorkflowSession(name="inner"):
                    pass


# ------------------------- 8. patch restoration ------------------------------

class TestPatchRestoration(unittest.TestCase):
    def test_methods_restored_after_session(self):
        originals = {
            "FakeLLM.one_shot": FakeLLM.one_shot,
            "DeepseekLLM.one_shot": DeepseekLLM.one_shot,
            "BaseLLM._execute_tool_calls": BaseLLM._execute_tool_calls,
            "ReActAgent.run": ReActAgent.run,
            "PlanExecuteAgent.run": PlanExecuteAgent.run,
            "PlanReactAgent.run": PlanReactAgent.run,
        }
        with WorkflowSession(name="restore"):
            pass
        self.assertIs(FakeLLM.one_shot, originals["FakeLLM.one_shot"])
        self.assertIs(DeepseekLLM.one_shot, originals["DeepseekLLM.one_shot"])
        self.assertIs(BaseLLM._execute_tool_calls,
                      originals["BaseLLM._execute_tool_calls"])
        self.assertIs(ReActAgent.run, originals["ReActAgent.run"])
        self.assertIs(PlanExecuteAgent.run, originals["PlanExecuteAgent.run"])
        self.assertIs(PlanReactAgent.run, originals["PlanReactAgent.run"])

    def test_calls_outside_session_are_not_recorded(self):
        llm = FakeLLM(canned_responses=[simple_answer("quiet")])
        llm.one_shot()  # no session active -> recorder wrapper is bypassed
        # FakeLLM appended only its own assistant turn: system + user + assistant.
        self.assertEqual(len(llm.memory), 3)
        self.assertEqual(llm.memory[-1]["role"], "assistant")


# ------------------------- 9. tool executions --------------------------------

class TestToolEvents(unittest.TestCase):
    def test_tool_events_recorded_with_observations(self):
        canned = [
            {"reasoning": None, "content": "calling add",
             "tool_calls": [{"id": "c1", "name": "add",
                             "arguments": {"a": 1, "b": 2}}]},
            simple_answer("result is 3"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeReAct(
                model="fake-model", max_tokens=100, instruction="test",
                tool_registry={"add": add_tool}, canned_responses=canned,
                config={"session_path": tmp},
            )
            with WorkflowSession(name="tools") as session:
                agent.run("use the tool")

            record = session.steps["1"]
            self.assertEqual(len(record["tool_events"]), 1)
            event = record["tool_events"][0]
            self.assertEqual(event["tool"], "add")
            self.assertEqual(event["arguments"], {"a": 1, "b": 2})
            self.assertEqual(event["observation"], "3")
            self.assertTrue(any(m.get("role") == "tool" for m in record["memory"]))


# ------------------------- 10. standalone onestep() ---------------------------

class TestStandaloneOnestep(unittest.TestCase):
    def test_onestep_without_run_attaches_tools_to_llm_step(self):
        canned = [
            {"reasoning": None, "content": "calling add",
             "tool_calls": [{"id": "c1", "name": "add",
                             "arguments": {"a": 2, "b": 5}}]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeReAct(
                model="fake-model", max_tokens=100, instruction="test",
                tool_registry={"add": add_tool}, canned_responses=canned,
                config={"session_path": tmp},
            )
            with WorkflowSession(name="onestep") as session:
                agent.onestep()

            self.assertEqual(list(session.steps), ["1"])
            record = session.steps["1"]
            self.assertEqual(record["type"], "llm_call")
            self.assertEqual(record["tool_events"][0]["tool"], "add")
            self.assertEqual(record["tool_events"][0]["observation"], "7")


# ------------------------- 11. sanitization -----------------------------------

class TestSanitization(unittest.TestCase):
    def test_planexecute_dataclass_becomes_plain_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_plan_execute(PLAN_CANNED, tmp)
            with WorkflowSession(name="sanitize") as session:
                agent.run("task")

            record = session.steps["1"]
            # run() returned a PlanExecute dataclass -> now a plain dict
            self.assertIsInstance(record["result"]["plan"], dict)
            self.assertEqual(record["result"]["plan"]["execute_steps"],
                             ["First step.", "Second step."])
            self.assertEqual(len(record["step_memories"]), 2)

            payload = load_session_file(tmp)
            self.assertIsInstance(payload["1"]["result"]["plan"], dict)
            self.assertIsInstance(payload["1"]["step_memories"], list)


# ------------------------- 13. per-object config contract ---------------------

class TestConfigContract(unittest.TestCase):
    def test_object_without_config_raises(self):
        """A recorded object with no config attribute raises a descriptive
        RuntimeError instead of silently skipping saving."""
        llm = NoConfigLLM(canned_responses=[simple_answer("ok")])
        with self.assertRaisesRegex(RuntimeError, "no config attribute"):
            with WorkflowSession(name="noconfig"):
                llm.one_shot()

    def test_config_missing_session_path_raises(self):
        """config={} (the BaseLLM default) has no session_path key: the
        error names the missing key."""
        llm = FakeLLM(canned_responses=[simple_answer("ok")])
        with self.assertRaisesRegex(RuntimeError, "no 'session_path' key"):
            with WorkflowSession(name="nokey"):
                llm.one_shot()

    def test_empty_or_non_string_session_path_raises(self):
        # BaseLLM's constructor rejects non-str config VALUES up front, so
        # to exercise the session-layer check we mutate config after
        # construction (defense in depth).
        for bad in ("", "   ", None, 42):
            with self.subTest(session_path=bad):
                llm = FakeLLM(canned_responses=[simple_answer("ok")])
                llm.config["session_path"] = bad
                with self.assertRaisesRegex(RuntimeError, "session_path"):
                    with WorkflowSession(name="badpath"):
                        llm.one_shot()

    def test_nonexistent_session_path_raises_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "does", "not", "exist")
            llm = FakeLLM(canned_responses=[simple_answer("ok")],
                          config={"session_path": missing})
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                with WorkflowSession(name="missing"):
                    llm.one_shot()
            self.assertFalse(os.path.exists(missing))  # never auto-created

    def test_conflicting_session_paths_raise(self):
        """One workflow = one directory: a second object resolving to a
        different directory raises, and the partial session is saved."""
        with tempfile.TemporaryDirectory() as tmp_a, \
                tempfile.TemporaryDirectory() as tmp_b:
            llm_a = FakeLLM(canned_responses=[simple_answer("a")],
                            config={"session_path": tmp_a})
            llm_b = FakeLLM(canned_responses=[simple_answer("b")],
                            config={"session_path": tmp_b})
            with self.assertRaisesRegex(RuntimeError, "conflicting session_paths"):
                with WorkflowSession(name="conflict") as session:
                    llm_a.one_shot()
                    llm_b.one_shot()

            # the first call was recorded before the conflict surfaced
            self.assertEqual(list(session.steps), ["1"])
            payload = load_session_file(tmp_a)
            self.assertEqual(list(payload), ["1"])
            self.assertEqual(list_json_files(tmp_b), [])

    def test_empty_workflow_writes_nothing_and_raises_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with WorkflowSession(name="empty") as session:
                pass

            self.assertEqual(session.steps, {})
            self.assertIsNone(session.directory)
            self.assertIsNone(session.path)
            self.assertEqual(list_json_files(tmp), [])

    def test_empty_decorated_workflow_still_runs(self):
        ran = []

        @WorkflowSession.capture("empty-deco")
        def workflow():
            ran.append(True)

        workflow()
        self.assertEqual(ran, [True])
        self.assertIsNone(workflow._last_session.directory)
        self.assertIsNone(workflow._last_session.path)
        self.assertEqual(workflow._last_session.steps, {})


if __name__ == "__main__":
    unittest.main()
