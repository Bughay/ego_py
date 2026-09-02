"""Unit tests for agent_logic.llm.base.BaseLLM — the shared plumbing.

Runs entirely offline: FakeLLM supplies canned model responses so we can
test every behaviour of base.py deterministically:

  1. Constructor & memory/prompt validation
  2. Property validation (model, temperature, max_tokens, memory)
  3. Dunder / list-like behaviours
  4. Tool schema generation
  5. Tool execution
  6. extract()     — structured extraction
  7. classify()    — constrained classification (incl. retry loop)
  8. summarize()   — summarization
  9. trim_memory() — memory range cutting
  10. save_memory() — memory range getter (deep-copied snapshot)
  11. format_memory()/str() — clean print(agent) transcript rendering
  12. config attribute    — default {}, str->str validation, private copy
"""
import copy
import contextlib
import io
import json
import unittest

from ego_py.llm.base import BaseLLM
from test.fakes import FakeLLM, StrictLLM


# ------------------------- module-level tool fixtures ----------------------

def sample_tool(a: int, b: float = 1.0, name: str = "x",
                flag: bool = False, tags: list = None, note=None) -> str:
    """Add two numbers, with metadata."""
    return "ok"


def undoc_tool(x):
    return x


def weird_tool(cfg: dict):
    return cfg


def add(a: float, b: float) -> float:
    return a + b


def get_info() -> dict:
    return {"ok": True, "count": 3}


def get_list() -> list:
    return [1, 2, 3]


def boom():
    raise RuntimeError("boom")


# ------------------------- 1. constructor validation -----------------------

class TestConstructorValidation(unittest.TestCase):
    def test_memory_only_ok(self):
        llm = StrictLLM(memory=[{"role": "user", "content": "hi"}])
        self.assertEqual(llm.memory, [{"role": "user", "content": "hi"}])

    def test_prompts_ok(self):
        llm = StrictLLM(system_prompt="sys", user_prompt="hi")
        self.assertEqual(llm.memory, [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ])

    def test_nothing_raises(self):
        with self.assertRaisesRegex(ValueError, "Either provide 'memory' or both"):
            StrictLLM()

    def test_system_without_user_raises(self):
        with self.assertRaisesRegex(ValueError, "system prompt but not the user_prompt"):
            StrictLLM(system_prompt="sys")

    def test_user_without_system_raises(self):
        with self.assertRaisesRegex(ValueError, "Either provide 'memory' or both"):
            StrictLLM(user_prompt="hi")

    def test_memory_and_prompts_raises(self):
        with self.assertRaisesRegex(ValueError, "Provide only one"):
            StrictLLM(
                memory=[{"role": "user", "content": "hi"}],
                system_prompt="sys",
                user_prompt="hi",
            )


# ------------------------- 1b. config attribute --------------------------

class TestConfigAttribute(unittest.TestCase):
    def test_config_defaults_to_empty_dict(self):
        llm = FakeLLM()
        self.assertEqual(llm.config, {})
        self.assertIsInstance(llm.config, dict)

    def test_config_none_becomes_empty_dict(self):
        llm = FakeLLM(config=None)
        self.assertEqual(llm.config, {})

    def test_config_str_to_str_accepted(self):
        config = {"session_path": "/tmp/sessions", "env": "prod"}
        llm = FakeLLM(config=config)
        self.assertEqual(llm.config, config)

    def test_config_stored_as_private_copy(self):
        config = {"session_path": "/tmp/a"}
        llm = FakeLLM(config=config)
        config["session_path"] = "/tmp/b"
        config["new_key"] = "x"
        self.assertEqual(llm.config, {"session_path": "/tmp/a"})

    def test_config_non_dict_raises(self):
        for bad in ("nope", 42, ["a"], ("a",)):
            with self.subTest(config=bad):
                with self.assertRaisesRegex(TypeError, "config must be a dict"):
                    FakeLLM(config=bad)

    def test_config_non_str_key_raises(self):
        with self.assertRaisesRegex(TypeError, "str keys to str values"):
            FakeLLM(config={1: "x"})

    def test_config_non_str_value_raises(self):
        for bad in (1, None, ["x"], {"k": "v"}):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(TypeError, "str keys to str values"):
                    FakeLLM(config={"a": bad})

    def test_context_manager_dict_accepted(self):
        llm = FakeLLM(config={"session_path": "/tmp/s", "context_manager": {"summarize": 100, "max_iteration": 5}})
        self.assertEqual(
            llm.config["context_manager"],
            {"summarize": 100, "max_iteration": 5},
        )

    def test_context_manager_missing_keys_default_to_none(self):
        llm = FakeLLM(config={"context_manager": {"summarize": 100}})
        self.assertEqual(
            llm.config["context_manager"],
            {"summarize": 100, "max_iteration": None},
        )
        llm = FakeLLM(config={"context_manager": {}})
        self.assertEqual(
            llm.config["context_manager"],
            {"summarize": None, "max_iteration": None},
        )

    def test_context_manager_non_dict_raises(self):
        for bad in ("x", 1, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(TypeError, "context_manager.*dict"):
                    FakeLLM(config={"context_manager": bad})

    def test_context_manager_bad_key_raises(self):
        with self.assertRaisesRegex(ValueError, "summarize.*max_iteration"):
            FakeLLM(config={"context_manager": {"nope": 1}})

    def test_context_manager_bad_value_raises(self):
        for bad in (0, -1, True, 1.5, "5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    FakeLLM(config={"context_manager": {"summarize": bad}})


# ------------------------- 2. property validation --------------------------

class TestPropertyValidation(unittest.TestCase):
    def make(self):
        return FakeLLM()

    def test_model_must_be_nonempty_str(self):
        llm = self.make()
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            llm.model = ""
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            llm.model = 42

    def test_temperature_bounds(self):
        llm = self.make()
        llm.temperature = 0.0
        llm.temperature = 2.0
        for bad in (-0.1, 2.1):
            with self.assertRaisesRegex(ValueError, "between 0 and 2"):
                llm.temperature = bad

    def test_max_tokens_bounds(self):
        llm = self.make()
        llm.max_tokens = 1
        llm.max_tokens = 1_000_000
        with self.assertRaises(ValueError):
            llm.max_tokens = 0
        with self.assertRaises(ValueError):
            llm.max_tokens = -10
        with self.assertRaises(ValueError):
            llm.max_tokens = 1_000_001
        with self.assertRaises(TypeError):
            llm.max_tokens = "100"
        with self.assertRaises(TypeError):
            llm.max_tokens = 10.5

    def test_memory_shape(self):
        llm = self.make()
        with self.assertRaises(TypeError):
            llm.memory = "not a list"
        with self.assertRaises(ValueError):
            llm.memory = [{"role": "system"}]  # missing content
        llm.memory = [{"role": "user", "content": "ok"}]  # valid
        self.assertEqual(llm.memory, [{"role": "user", "content": "ok"}])


# ------------------------- 3. dunder / list behaviours ---------------------

class TestDunders(unittest.TestCase):
    def make(self):
        return FakeLLM(system_prompt="sys", user_prompt="hi")

    def test_len_index_iter(self):
        llm = self.make()
        self.assertEqual(len(llm), 2)
        self.assertEqual(llm[0], {"role": "system", "content": "sys"})
        self.assertEqual(llm[-1]["role"], "user")
        self.assertEqual([m["role"] for m in llm], ["system", "user"])

    def test_setitem_validates_shape(self):
        llm = self.make()
        llm[1] = {"role": "assistant", "content": "yo"}
        self.assertEqual(llm.memory[1]["content"], "yo")
        with self.assertRaisesRegex(ValueError, "'role' and 'content'"):
            llm[1] = "bad"

    def test_setitem_slice_accepts_workflow_shapes(self):
        # Every message shape the app produces must pass validation:
        # system/user, assistant (content=None allowed, key present),
        # tool (tool_call_id + content), assistant with tool_calls.
        llm = self.make()
        transcript = [
            {"role": "user", "content": "What is 20 - 4?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "subtract", "arguments": '{"a": 20, "b": 4}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "16"},
            {"role": "assistant", "content": "20 - 4 = 16."},
        ]
        llm[1:1] = transcript  # insert after the system prompt
        self.assertEqual(len(llm), 6)
        self.assertEqual(llm.memory[1]["role"], "user")
        self.assertIsNone(llm.memory[2]["content"])
        self.assertEqual(llm.memory[3]["tool_call_id"], "call_1")
        self.assertEqual(llm.memory[4], {"role": "assistant", "content": "20 - 4 = 16."})
        self.assertEqual(llm.memory[5], {"role": "user", "content": "hi"})

    def test_setitem_slice_transfer_between_agents(self):
        a1 = self.make()
        a2 = self.make()
        a1.memory.append({"role": "assistant", "content": "answer"})
        a2[1:2] = a1[2:3]  # hand a1's assistant turn over to a2
        self.assertEqual(a2.memory[1], {"role": "assistant", "content": "answer"})
        self.assertEqual(len(a2), 2)

    def test_setitem_slice_rejects_bad_items(self):
        llm = self.make()
        with self.assertRaisesRegex(ValueError, "'role' and 'content'"):
            llm[1:1] = [{"role": "user"}]  # missing content
        with self.assertRaisesRegex(ValueError, "'role' and 'content'"):
            llm[1:1] = ["not a dict"]
        # failed validation must not mutate memory
        self.assertEqual(len(llm), 2)

    def test_setitem_slice_requires_list(self):
        llm = self.make()
        with self.assertRaises(TypeError):
            llm[1:1] = "nope"
        with self.assertRaises(TypeError):
            llm[1:1] = {"role": "user", "content": "x"}  # dict is not a list
        self.assertEqual(len(llm), 2)

    def test_memory_list_ops_bypass_validation(self):
        # agent.memory[...] is the documented raw tier: direct list ops on
        # the live list run with no checks, on purpose.
        llm = self.make()
        llm.memory[0:1] = [{"role": "user"}]  # missing content — allowed
        self.assertEqual(llm.memory[0], {"role": "user"})
        self.assertEqual(len(llm), 2)

    def test_delitem(self):
        llm = self.make()
        del llm[1]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0]["role"], "system")

    def test_repr_str(self):
        llm = self.make()
        self.assertIn("FakeLLM(model='fake-model'", repr(llm))
        self.assertIn("memory_len=2", repr(llm))
        rendered = str(llm)
        self.assertIn("=== FakeLLM conversation (2 messages) ===", rendered)
        self.assertIn("[1] System prompt:\nsys", rendered)
        self.assertIn("[2] User:\nhi", rendered)


# ------------------------- 3b. format_memory / print ------------------------

class TestFormatMemory(unittest.TestCase):
    """Clean human-readable transcript rendering (str()/print(agent))."""

    TOOL_MEMORY = [
        {"role": "system", "content": "You are a calculator."},
        {"role": "user", "content": "What is 20 - 4?"},
        {
            "role": "assistant",
            "content": "Let me compute that.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "subtract", "arguments": '{"a": 20, "b": 4}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "16"},
        {"role": "assistant", "content": "20 - 4 = 16."},
    ]

    def make(self, memory):
        return FakeLLM(memory=memory)

    def test_header_and_order(self):
        out = self.make(self.TOOL_MEMORY).format_memory()
        self.assertTrue(out.startswith("=== FakeLLM conversation (5 messages) ==="))
        pos = out.index
        self.assertLess(pos("[1] System prompt:"), pos("[2] User:"))
        self.assertLess(pos("[2] User:"), pos("[3] Assistant:"))
        self.assertLess(pos("[3] Assistant:"), pos("[4] Tool call:"))
        self.assertLess(pos("[4] Tool call:"), pos("[5] Assistant:"))

    def test_tool_call_and_result_rendered(self):
        out = self.make(self.TOOL_MEMORY).format_memory()
        self.assertIn("    -> Tool call: subtract(a=20, b=4)", out)
        self.assertIn("[4] Tool call: subtract -> Result:\n16", out)
        self.assertIn("[5] Assistant:\n20 - 4 = 16.", out)

    def test_print_matches_str_and_format_memory(self):
        llm = self.make(self.TOOL_MEMORY)
        rendered = llm.format_memory()
        self.assertEqual(str(llm), rendered)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print(llm)
        self.assertEqual(buf.getvalue(), rendered + "\n")

    def test_unknown_tool_call_id_shown_raw(self):
        memory = [
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "ghost", "content": "nope"},
        ]
        out = self.make(memory).format_memory()
        self.assertIn("[2] Tool call: ghost -> Result:", out)

    def test_malformed_tool_arguments_rendered_verbatim(self):
        memory = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "subtract", "arguments": "not json"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "16"},
        ]
        out = self.make(memory).format_memory()
        self.assertIn("    -> Tool call: subtract(not json)", out)

    def test_plain_assistant_has_no_tool_lines(self):
        memory = [
            {"role": "assistant", "content": "Just text."},
        ]
        out = self.make(memory).format_memory()
        self.assertIn("[1] Assistant:\nJust text.", out)
        self.assertNotIn("-> Tool call", out)

    def test_non_destructive(self):
        llm = self.make(self.TOOL_MEMORY)
        snapshot = copy.deepcopy(llm.memory)
        llm.format_memory()
        self.assertEqual(llm.memory, snapshot)


# ------------------------- 4. tool schema generation -----------------------

class TestToolSchemaGeneration(unittest.TestCase):
    def test_full_signature(self):
        schema = BaseLLM._generate_tool_schema(sample_tool)
        self.assertEqual(schema["type"], "function")
        fn = schema["function"]
        self.assertEqual(fn["name"], "sample_tool")
        self.assertEqual(fn["description"], "Add two numbers, with metadata.")
        props = fn["parameters"]["properties"]
        self.assertEqual(props["a"]["type"], "number")      # int
        self.assertEqual(props["b"]["type"], "number")      # float
        self.assertEqual(props["name"]["type"], "string")
        self.assertEqual(props["flag"]["type"], "boolean")
        self.assertEqual(props["tags"]["type"], "array")
        self.assertEqual(props["note"]["type"], "string")   # untyped fallback
        # only 'a' has no default
        self.assertEqual(fn["parameters"]["required"], ["a"])

    def test_unknown_annotation_falls_back_to_string(self):
        schema = BaseLLM._generate_tool_schema(weird_tool)
        self.assertEqual(
            schema["function"]["parameters"]["properties"]["cfg"]["type"],
            "string",
        )

    def test_default_description(self):
        schema = BaseLLM._generate_tool_schema(undoc_tool)
        self.assertEqual(
            schema["function"]["description"],
            "Executes the undoc_tool operation.",
        )


# ------------------------- 5. tool execution -------------------------------

class TestToolExecution(unittest.TestCase):
    def setUp(self):
        self.llm = FakeLLM(tool_registry={
            "add": add,
            "get_info": get_info,
            "get_list": get_list,
            "boom": boom,
        })

    def test_scalar_result_stringified(self):
        obs = self.llm._execute_tool_calls(
            [{"name": "add", "arguments": {"a": 5, "b": 3}}]
        )
        self.assertEqual(obs, ["8"])

    def test_dict_result_serialised(self):
        obs = self.llm._execute_tool_calls(
            [{"name": "get_info", "arguments": {}}]
        )
        self.assertEqual(json.loads(obs[0]), {"ok": True, "count": 3})

    def test_list_result_serialised(self):
        obs = self.llm._execute_tool_calls(
            [{"name": "get_list", "arguments": {}}]
        )
        self.assertEqual(json.loads(obs[0]), [1, 2, 3])

    def test_unknown_tool(self):
        obs = self.llm._execute_tool_calls(
            [{"name": "nope", "arguments": {}}]
        )
        self.assertEqual(obs, ["Error: Unknown tool 'nope'"])

    def test_exception_captured(self):
        obs = self.llm._execute_tool_calls(
            [{"name": "boom", "arguments": {}}]
        )
        self.assertEqual(obs, ["Error executing boom: boom"])

    def test_order_preserved(self):
        obs = self.llm._execute_tool_calls([
            {"name": "add", "arguments": {"a": 1, "b": 1}},
            {"name": "add", "arguments": {"a": 2, "b": 2}},
        ])
        self.assertEqual(obs, ["2", "4"])

    def test_no_registry_raises(self):
        llm = FakeLLM()
        with self.assertRaisesRegex(RuntimeError, "No tool_registry"):
            llm._execute_tool_calls([])


# ------------------------- 6. extract --------------------------------------

SCHEMA = {
    "name": "the person's full name",
    "age": "age in years as an integer",
}


class TestExtract(unittest.TestCase):
    def make(self, canned):
        return FakeLLM(
            canned_responses=canned,
            system_prompt="sys",
            user_prompt="Tell me about Ada.",
        )

    def test_parses_clean_json(self):
        llm = self.make([{"content": '{"name": "Ada", "age": 42}'}])
        self.assertEqual(llm.extract(SCHEMA), {"name": "Ada", "age": 42})

    def test_strips_code_fences(self):
        llm = self.make([{"content": '```json\n{"name": "Ada", "age": 42}\n```'}])
        self.assertEqual(llm.extract(SCHEMA), {"name": "Ada", "age": 42})

    def test_invalid_json_returns_raw_string(self):
        llm = self.make([{"content": "not json"}])
        self.assertEqual(llm.extract(SCHEMA), "not json")

    def test_none_content_returns_empty_string(self):
        llm = self.make([{"content": None}])
        self.assertEqual(llm.extract(SCHEMA), "")

    def test_swaps_memory_and_response_format(self):
        llm = self.make([{"content": '{"name": "Ada", "age": 42}'}])
        original_memory = list(llm.memory)
        original_format = {"type": "text"}
        llm.response_format = original_format
        llm.extract(SCHEMA)
        # original state fully restored
        self.assertEqual(llm.memory, original_memory)
        self.assertEqual(llm.response_format, original_format)
        # the call itself saw a JSON response format and a temporary memory
        self.assertEqual(llm.seen_response_formats[0], {"type": "json_object"})
        sent = llm.seen_memories[0]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("structured extraction engine", sent[0]["content"])
        self.assertEqual(sent[-1], {"role": "user", "content": "Tell me about Ada."})

    def test_input_validation(self):
        llm = self.make([{"content": "{}"}])
        with self.assertRaisesRegex(ValueError, "non-empty dict"):
            llm.extract({})
        with self.assertRaises(TypeError):
            llm.extract({1: "x"})  # non-str key
        with self.assertRaisesRegex(TypeError, "example must be a str or None"):
            llm.extract(SCHEMA, example=123)
        with self.assertRaisesRegex(TypeError, "instruction must be a str or None"):
            llm.extract(SCHEMA, instruction=123)


# ------------------------- 7. classify -------------------------------------

SENTIMENT_SCHEMA = {
    "sentiment": {"description": "overall tone", "choices": ["positive", "negative"]},
}


class TestClassify(unittest.TestCase):
    def make(self, canned, **kw):
        kw.setdefault("system_prompt", "sys")
        kw.setdefault("user_prompt", "I love this product!")
        return FakeLLM(canned_responses=canned, **kw)

    def test_list_choices_ok(self):
        llm = self.make([{"content": '{"sentiment": "positive"}'}])
        self.assertEqual(llm.classify(SENTIMENT_SCHEMA), {"sentiment": "positive"})

    def test_range_string_choices(self):
        schema = {"urgency": {"description": "how urgent", "choices": "1-5"}}
        llm = self.make([{"content": '{"urgency": 3}'}])
        self.assertEqual(llm.classify(schema), {"urgency": 3})

    def test_range_dict_choices(self):
        schema = {"urgency": {"description": "how urgent", "choices": {"min": 1, "max": 5}}}
        llm = self.make([{"content": '{"urgency": 5}'}])
        self.assertEqual(llm.classify(schema), {"urgency": 5})

    def test_retry_then_success(self):
        llm = self.make([
            {"content": '{"sentiment": "happy"}'},     # invalid first attempt
            {"content": '{"sentiment": "positive"}'},  # corrected after feedback
        ])
        result = llm.classify(SENTIMENT_SCHEMA)
        self.assertEqual(result, {"sentiment": "positive"})
        self.assertEqual(llm.one_shot_calls, 2)
        feedback = llm.seen_memories[1][-1]["content"]
        self.assertIn("Your previous output was invalid", feedback)

    def test_persistent_failure_raises(self):
        llm = self.make([
            {"content": '{"sentiment": "happy"}'},
            {"content": '{"sentiment": "meh"}'},
            {"content": '{"sentiment": "angry"}'},
        ])
        with self.assertRaisesRegex(ValueError, "violates the classify schema"):
            llm.classify(SENTIMENT_SCHEMA)
        self.assertEqual(llm.one_shot_calls, 3)  # initial + 2 retries

    def test_input_validation(self):
        llm = self.make([{"content": "{}"}])
        with self.assertRaisesRegex(ValueError, "non-empty dict"):
            llm.classify({})
        with self.assertRaises(TypeError):
            llm.classify({"k": "not a dict"})
        with self.assertRaises(ValueError):
            llm.classify({"k": {}})  # missing description/choices
        with self.assertRaises(TypeError):
            llm.classify({"k": {"description": 1, "choices": ["a"]}})
        with self.assertRaises(ValueError):
            llm.classify({"k": {"description": "d", "choices": []}})
        with self.assertRaisesRegex(ValueError, "must look like '1-5'"):
            llm.classify({"k": {"description": "d", "choices": "abc"}})
        with self.assertRaises(ValueError):
            llm.classify({"k": {"description": "d", "choices": "5-1"}})
        with self.assertRaises(TypeError):
            llm.classify({"k": {"description": "d", "choices": {"min": 1.0, "max": 5}}})
        with self.assertRaises(TypeError):
            llm.classify({"k": {"description": "d", "choices": {"min": True, "max": 5}}})
        with self.assertRaisesRegex(TypeError, "example must be a str or None"):
            llm.classify(SENTIMENT_SCHEMA, example=1)
        with self.assertRaisesRegex(TypeError, "instruction must be a str or None"):
            llm.classify(SENTIMENT_SCHEMA, instruction=1)


class TestClassifyErrorsHelper(unittest.TestCase):
    """Direct tests of the _classify_errors static helper."""

    def schema(self):
        return {
            "sentiment": {"choices": ["positive", "negative"]},
            "score": {"choices": {"min": 1, "max": 5}},
        }

    def test_valid_output(self):
        self.assertEqual(
            BaseLLM._classify_errors({"sentiment": "positive", "score": 3}, self.schema()),
            [],
        )

    def test_not_a_dict(self):
        self.assertEqual(
            BaseLLM._classify_errors("nope", self.schema()),
            ["output must be a JSON object"],
        )

    def test_missing_and_unexpected_keys(self):
        errs = BaseLLM._classify_errors({"sentiment": "positive", "extra": 1}, self.schema())
        self.assertIn("missing key 'score'", errs)
        self.assertIn("unexpected key 'extra'", errs)

    def test_out_of_choices(self):
        errs = BaseLLM._classify_errors({"sentiment": "meh", "score": 9}, self.schema())
        self.assertIn("'sentiment' = 'meh' is not one of ['positive', 'negative']", errs)
        self.assertIn("'score' = 9 is outside [1, 5]", errs)

    def test_bool_rejected_for_int_range(self):
        errs = BaseLLM._classify_errors({"sentiment": "positive", "score": True}, self.schema())
        self.assertIn("'score' = True is not an integer", errs)


# ------------------------- 8. summarize ------------------------------------

class TestSummarize(unittest.TestCase):
    def make(self, canned):
        return FakeLLM(
            canned_responses=canned,
            system_prompt="sys",
            user_prompt="hi",
        )

    def test_returns_summary(self):
        llm = self.make([{"content": "Short summary."}])
        self.assertEqual(llm.summarize("A long text.", 500), "Short summary.")

    def test_swaps_memory_and_max_tokens(self):
        llm = self.make([{"content": "Summary."}])
        original_memory = list(llm.memory)
        original_max = llm.max_tokens
        llm.summarize("Context here.", 1234)
        self.assertEqual(llm.memory, original_memory)
        self.assertEqual(llm.max_tokens, original_max)
        self.assertEqual(llm.seen_max_tokens[0], 1234)
        sent = llm.seen_memories[0]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("summarization engine", sent[0]["content"])
        self.assertEqual(sent[1], {"role": "user", "content": "Context here."})

    def test_input_validation(self):
        llm = self.make([{"content": "Summary."}])
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            llm.summarize("", 100)
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            llm.summarize("   ", 100)
        with self.assertRaises(ValueError):
            llm.summarize(123, 100)
        with self.assertRaisesRegex(TypeError, "max_tokens must be an integer"):
            llm.summarize("ctx", True)
        with self.assertRaisesRegex(TypeError, "max_tokens must be an integer"):
            llm.summarize("ctx", 1.5)
        with self.assertRaisesRegex(ValueError, "positive"):
            llm.summarize("ctx", 0)


# ------------------------- 8b. context_manager -----------------------------

class TestContextManager(unittest.TestCase):
    def make(self, config, canned=None, tokens_per_call=0, memory=None):
        kwargs = {"config": config, "tokens_per_call": tokens_per_call}
        if memory is not None:
            kwargs["memory"] = memory
        else:
            kwargs["system_prompt"] = "sys"
            kwargs["user_prompt"] = "hello"
        return FakeLLM(canned_responses=canned, **kwargs)

    def test_disabled_by_default(self):
        llm = self.make({}, canned=[{"content": "a"}, {"content": "b"}],
                        tokens_per_call=100)
        llm.one_shot()
        llm.one_shot()
        self.assertEqual(llm.tokens_used, 200)
        self.assertEqual(len(llm.memory), 4)  # sys, user, a, b
        self.assertFalse(any(
            m[0]["role"] == "system" and "summarization engine" in m[0]["content"]
            for m in llm.seen_memories
        ))

    def test_summarize_triggers_at_token_threshold(self):
        llm = self.make(
            {"context_manager": {"summarize": 3, "max_iteration": None}},
            canned=[
                {"content": "a"},
                {"content": "b"},
                {"content": "summary"},
                {"content": "c"},
            ],
            tokens_per_call=2,
        )
        llm.one_shot()
        llm.one_shot()
        llm.one_shot()

        self.assertEqual(llm.tokens_used, 2)  # reset after summarize, then +2
        self.assertEqual(llm.memory[0]["role"], "system")
        self.assertEqual(llm.memory[1], {"role": "user", "content": "summary"})
        self.assertEqual(llm.memory[2], {"role": "assistant", "content": "c"})
        self.assertTrue(any(
            m[0]["role"] == "system" and "summarization engine" in m[0]["content"]
            for m in llm.seen_memories
        ))

    def test_max_iteration_triggers_trim(self):
        memory = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "t1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        llm = self.make(
            {"context_manager": {"summarize": None, "max_iteration": 2}},
            canned=[{"content": "a"}, {"content": "b"}],
            memory=memory,
        )
        llm.one_shot()
        llm.one_shot()

        contents = [m["content"] for m in llm.memory]
        self.assertNotIn("u1", contents)
        self.assertNotIn("a1", contents)
        self.assertEqual(llm.memory[0], {"role": "system", "content": "sys"})
        self.assertEqual(llm.memory[1], {"role": "tool", "content": "t1"})
        self.assertEqual(len(llm.memory), 6)  # sys + 4 kept + assistant b


# ------------------------- 9. trim_memory ----------------------------------

class TestTrimMemory(unittest.TestCase):
    def make(self):
        memory = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "m1"},
            {"role": "assistant", "content": "m2"},
            {"role": "user", "content": "m3"},
            {"role": "assistant", "content": "m4"},
        ]
        return FakeLLM(memory=memory)

    def test_trims_range_keeps_system(self):
        llm = self.make()
        result = llm.trim_memory([2, 4])
        self.assertIsNone(result)
        self.assertEqual(llm.memory, [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "m4"},
        ])

    def test_system_never_deleted_even_when_in_range(self):
        llm = self.make()
        llm.trim_memory([1, 3])  # range covers the system prompt's position
        self.assertEqual(llm.memory, [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "m3"},
            {"role": "assistant", "content": "m4"},
        ])

    def test_full_range_leaves_only_system(self):
        llm = self.make()
        llm.trim_memory([1, 5])
        self.assertEqual(llm.memory, [{"role": "system", "content": "sys"}])

    def test_input_validation(self):
        llm = self.make()
        with self.assertRaises(TypeError):
            llm.trim_memory([2])  # wrong length
        with self.assertRaises(TypeError):
            llm.trim_memory("24")
        with self.assertRaises(TypeError):
            llm.trim_memory([True, 3])
        with self.assertRaisesRegex(ValueError, "'from' must be >= 1"):
            llm.trim_memory([0, 2])
        with self.assertRaises(ValueError):
            llm.trim_memory([1, 6])  # beyond len(memory)
        with self.assertRaisesRegex(ValueError, "'from' must be <= 'to'"):
            llm.trim_memory([4, 2])


# ------------------------- 10. save_memory ---------------------------------

class TestSaveMemory(unittest.TestCase):
    def make(self):
        memory = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "m1"},
            {"role": "assistant", "content": "m2"},
            {"role": "user", "content": "m3"},
            {"role": "assistant", "content": "m4"},
        ]
        return FakeLLM(memory=memory)

    def test_returns_range_without_system(self):
        llm = self.make()
        saved = llm.save_memory([2, 4])
        self.assertEqual(saved, [
            {"role": "user", "content": "m1"},
            {"role": "assistant", "content": "m2"},
            {"role": "user", "content": "m3"},
        ])

    def test_skip_system_false_includes_system(self):
        llm = self.make()
        saved = llm.save_memory([1, 2], skip_system=False)
        self.assertEqual(saved, [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "m1"},
        ])

    def test_full_range_returns_all_non_system(self):
        llm = self.make()
        saved = llm.save_memory([1, 5])
        self.assertEqual(saved, [
            {"role": "user", "content": "m1"},
            {"role": "assistant", "content": "m2"},
            {"role": "user", "content": "m3"},
            {"role": "assistant", "content": "m4"},
        ])

    def test_non_destructive(self):
        llm = self.make()
        llm.save_memory([2, 4])
        self.assertEqual(len(llm.memory), 5)  # source untouched

    def test_snapshot_isolation(self):
        llm = self.make()
        saved = llm.save_memory([2, 3])
        saved[0]["content"] = "MUTATED"
        llm.memory.append({"role": "user", "content": "m5"})
        self.assertEqual(llm.memory[1], {"role": "user", "content": "m1"})
        self.assertEqual(len(saved), 2)  # later appends don't leak in

    def test_input_validation(self):
        llm = self.make()
        with self.assertRaises(TypeError):
            llm.save_memory([2])  # wrong length
        with self.assertRaises(TypeError):
            llm.save_memory("24")
        with self.assertRaises(TypeError):
            llm.save_memory([True, 3])
        with self.assertRaises(TypeError):
            llm.save_memory([1, 3], skip_system=1)
        with self.assertRaisesRegex(ValueError, "'from' must be >= 1"):
            llm.save_memory([0, 2])
        with self.assertRaises(ValueError):
            llm.save_memory([1, 6])  # beyond len(memory)
        with self.assertRaisesRegex(ValueError, "'from' must be <= 'to'"):
            llm.save_memory([4, 2])


if __name__ == "__main__":
    unittest.main()
