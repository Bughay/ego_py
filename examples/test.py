"""
System workflows — the runnable pipelines of the application.

Each decorated function is ONE complete workflow. Every LLM-based object
created inside a workflow carries its own plain config dict; the
WorkflowSession decorator reads each recorded object's
config["session_path"] at call time: the first recorded object's path must
be a non-empty string naming an EXISTING directory, and every later object
must resolve to the same directory. All conversation steps, LLM calls and
tool executions in the workflow are then recorded into a single numbered
JSON in that directory (one session file per workflow run):

    workflow_agents()        -> 3 agents (react, plan-execute, plan-react)
                                -> one session JSON with steps "1", "2", "3"
    workflow_plain_react()   -> no-tools ReAct agent
                                -> its own separate session JSON
    workflow_one_shot()      -> one raw LLM call
                                -> its own separate session JSON
    workflow_classify()      -> one classification call
                                -> its own separate session JSON
    workflow_summarize_add_subtract()    -> ReAct agent where summarize fires
    workflow_summarize_multiply_divide() -> ReAct agent where summarize fires

Run them via main.py, or directly:

    from examples import workflow_agents
    workflow_agents()

To add a new session: write another function here, decorate it with
@WorkflowSession.capture("<workflow_name>") and give every LLM(...) /
EgoAgent(...) inside it config={"session_path": <existing sessions dir>}.
"""
import json
import os

from egoai import EgoAgent, LLM, WorkflowSession
from egoai.builtin_tools.file import build_file_tools
from egoai.builtin_tools.math import build_math_tools

# Workspace the agents may work in (resolved relative to this file).
WORKSPACE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "test_agent")
)

# Existing sessions directory: every workflow object's config.session_path
# must point at an existing directory (None / non-existent -> RuntimeError
# at record time). Session JSONs land here.
SESSION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "sessions")
)

# The workflows only accept EXISTING directories (nothing is auto-created by
# WorkflowSession or the file tools), so make both dirs available up front.
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

TASK = (
    "Use the available math tools (add, subtract, multiply, divide) to "
    "compute ((20 - 4) * 5) / 2 + 3 step by step, then save the final "
    "result and your calculation steps into a file named result.txt in "
    "the workspace directory."
)

INSTRUCTION = "You are testing math and file operations inside the workspace."


# --------------------------------------------------------------------------- #
# Small print helpers: every example prints WHAT is being tested plus the     #
# system_prompt / user_prompt before the call, and the reply after it.        #
# --------------------------------------------------------------------------- #
def _describe(label, system_prompt, user_prompt):
    print("\n" + "=" * 70)
    print(f"TEST: {label}")
    print("-" * 70)
    print(f"system_prompt: {system_prompt!r}")
    print(f"user_prompt:   {user_prompt!r}")
    print("-" * 70)


def _reply(reply):
    print(f"reply: {reply!r}")


def _prompts_from_memory(memory):
    system_prompt = next(
        (m.get("content") for m in memory if m.get("role") == "system"), None
    )
    user_prompt = next(
        (m.get("content") for m in memory if m.get("role") == "user"), None
    )
    return system_prompt, user_prompt


@WorkflowSession.capture("llm_features")
def workflow_llm_features():
    """LLM-only workflow (sections 1-4 from the old main.py, kept for
    reference). Every raw one_shot/extract/classify/summarize call is saved
    as its own numbered step in one session JSON."""
    # ------------------------------------------------------------------ #
    # 1. one_shot — plain question, one LLM call.                        #
    # ------------------------------------------------------------------ #
    llm = LLM(
        model="deepseek-v4-flash",
        system_prompt="You are a helpful, precise assistant.",
        user_prompt="Explain quantum entanglement in exactly three sentences.",
        reasoning_effort="low",
        config={"session_path": SESSION_DIR},
    )
    system_prompt, user_prompt = _prompts_from_memory(llm.memory)
    _describe("one_shot — plain question", system_prompt, user_prompt)
    result = llm.one_shot()
    _reply(result)

    # ------------------------------------------------------------------ #
    # 2. extract — structured JSON extraction against a schema.          #
    # ------------------------------------------------------------------ #
    llm = LLM(
        model="deepseek-v4-flash",
        system_prompt="Extract data.",
        user_prompt=(
            "Ada Lovelace is 36 years old and lives in London. She is known "
            "for analytical engines, mathematics and early programming."
        ),
        config={"session_path": SESSION_DIR},
    )
    system_prompt, user_prompt = _prompts_from_memory(llm.memory)
    _describe("extract — structured JSON against a schema", system_prompt, user_prompt)
    data = llm.extract(
        schema={
            "name": "the person's full name",
            "age": "their age as an integer",
            "city": "the city they live in",
            "skills": "a list of their professional skills as strings",
        },
        example=json.dumps({"name": "Jane Doe", "age": 30, "city": "London",
                            "skills": ["python", "sql"]}),
    )
    _reply(data)

    # ------------------------------------------------------------------ #
    # 3. classify — constrained classification (one allowed value/key).  #
    # ------------------------------------------------------------------ #
    llm = LLM(
        model="deepseek-v4-flash",
        system_prompt="Classify messages.",
        user_prompt=(
            "Hi, I was charged twice for my subscription this month and I "
            "need this fixed today because my card was already overcharged."
        ),
        config={"session_path": SESSION_DIR},
    )
    system_prompt, user_prompt = _prompts_from_memory(llm.memory)
    _describe("classify — constrained classification", system_prompt, user_prompt)
    labels = llm.classify(
        schema={
            "sentiment": {"description": "overall sentiment of the message",
                          "choices": ["positive", "neutral", "negative"]},
            "urgency": {"description": "how urgent the request is",
                        "choices": "1-5"},
            "topic": {"description": "the main topic",
                      "choices": ["billing", "technical", "account", "other"]},
        },
        instruction="Classify this support message.",
    )
    _reply(labels)

    # ------------------------------------------------------------------ #
    # 4. summarize — compress a context into a token budget.             #
    # ------------------------------------------------------------------ #
    llm = LLM(
        model="deepseek-v4-flash",
        system_prompt="You are the agent under test.",
        user_prompt="Placeholder.",
        config={"session_path": SESSION_DIR},
    )
    context = (
        "[01] system: You are a coding agent.\n"
        "[02] user: Add a POST /api/v1/transfers endpoint to the backend.\n"
        "[03] assistant: Action: read_file('backend/app/routes.py')\n"
        "[04] tool: Observation: 200 lines, no transfers router yet.\n"
        "[05] assistant: Action: edit_file('backend/app/routes.py', add router)\n"
        "[06] tool: Observation: edit applied.\n"
        "[07] assistant: Action: run_tests('backend/tests/test_transfers.py')\n"
        "[08] tool: Observation: 6 passed, 0 failed."
    )
    system_prompt, user_prompt = _prompts_from_memory(llm.memory)
    _describe("summarize — compress a context into a token budget", system_prompt, context)
    summary = llm.summarize(
        context + "\n\nSummarize the work above in two concise paragraphs.",
        max_tokens=5000,
    )
    _reply(summary)


@WorkflowSession.capture("one_shot_workflow")
def workflow_one_shot():
    """One raw one_shot LLM call, saved as its own numbered session JSON."""
    llm = LLM(
        model="deepseek-v4-flash",
        system_prompt="You are a helpful, precise assistant.",
        user_prompt="Explain quantum entanglement in exactly three sentences.",
        reasoning_effort="low",
        config={"session_path": SESSION_DIR},
    )
    system_prompt, user_prompt = _prompts_from_memory(llm.memory)
    _describe("one_shot — plain question", system_prompt, user_prompt)
    result = llm.one_shot()
    _reply(result)


@WorkflowSession.capture("classify_workflow")
def workflow_classify():
    """One classification call, saved as its own numbered session JSON."""
    llm = LLM(
        model="deepseek-v4-flash",
        system_prompt="Classify messages.",
        user_prompt=(
            "Hi, I was charged twice for my subscription this month and I "
            "need this fixed today because my card was already overcharged."
        ),
        config={"session_path": SESSION_DIR},
    )
    system_prompt, user_prompt = _prompts_from_memory(llm.memory)
    _describe("classify — constrained classification", system_prompt, user_prompt)
    labels = llm.classify(
        schema={
            "sentiment": {"description": "overall sentiment of the message",
                          "choices": ["positive", "neutral", "negative"]},
            "urgency": {"description": "how urgent the request is",
                        "choices": "1-5"},
            "topic": {"description": "the main topic",
                      "choices": ["billing", "technical", "account", "other"]},
        },
        instruction="Classify this support message.",
    )
    _reply(labels)


@WorkflowSession.capture("agents_workflow")
def workflow_agents():
    """One workflow: run the three agent types.

    react -> plan-execute -> plan-react. The decorator saves everything into
    one session JSON with numbered steps "1", "2", "3" — each step holds the
    agent metadata, the run result, the full conversation memory and every
    LLM/tool call made inside that conversation.
    """
    # Tool registry built by hand: the math tools plus the workspace file
    # tools. Nothing is registered automatically — the agent gets exactly
    # the tools we pass in via tool_registry.
    tool_registry = {**build_math_tools(), **build_file_tools(WORKSPACE_DIR)}

    # ------------------------------------------------------------------ #
    # 5. ReAct agent — loop Reason -> Act -> Observe until done.         #
    # ------------------------------------------------------------------ #
    agent = EgoAgent(
        agent="react",
        model="deepseek-v4-flash",
        max_tokens=10000,
        instruction=INSTRUCTION,
        directory=WORKSPACE_DIR,
        tool_registry=tool_registry,
        reasoning_effort="low",  # thinking level, forwarded to the provider
        config={"session_path": SESSION_DIR},
    )
    _describe("react agent — Reason -> Act -> Observe", agent._react_prompt, TASK)
    result = agent.run(TASK, max_steps=10)
    _reply(result)
    print(agent)

    # ------------------------------------------------------------------ #
    # 6. Plan-Execute agent — plan first, then execute each step.        #
    # ------------------------------------------------------------------ #
    agent = EgoAgent(
        agent="plan-execute",
        model="deepseek-v4-flash",
        max_tokens=10000,
        instruction=INSTRUCTION,
        directory=WORKSPACE_DIR,
        tool_registry=tool_registry,
        config={"session_path": SESSION_DIR},
    )
    _describe("plan-execute agent — plan then execute", agent._planner_prompt, TASK)
    result = agent.run(TASK, max_steps=10)
    _reply(result)

    # ------------------------------------------------------------------ #
    # 7. Plan-React agent — plan, then a ReAct loop per step.            #
    # ------------------------------------------------------------------ #
    agent = EgoAgent(
        agent="plan-react",
        model="deepseek-v4-flash",
        max_tokens=10000,
        instruction=INSTRUCTION,
        directory=WORKSPACE_DIR,
        tool_registry=tool_registry,
        config={"session_path": SESSION_DIR},
    )
    _describe("plan-react agent — plan, then ReAct per step", agent._planner_prompt, TASK)
    result = agent.run(TASK, max_steps=10)
    _reply(result)


@WorkflowSession.capture("plain_react_workflow")
def workflow_plain_react():
    """Second workflow: a ReAct agent without tools — saved as its own,
    separate session JSON (this is how you record two distinct sessions:
    two decorated workflows)."""
    agent = EgoAgent(
        agent="react",
        model="deepseek-v4-flash",
        max_tokens=10000,
        instruction="You are a helpful API advisor. Answer questions directly.",
        config={"session_path": SESSION_DIR},
    )
    task = "In one paragraph, how does a JSON REST API typically signal errors?"
    _describe("plain react agent — no tools", agent._react_prompt, task)
    result = agent.run(task, max_steps=5)
    _reply(result)
    print(agent)


@WorkflowSession.capture("summarize_add_subtract_workflow")
def workflow_summarize_add_subtract():
    """Summarize example #1: a ReAct agent with math tools.

    The context_manager["summarize"] threshold is intentionally tiny so the
    accumulated real API token usage crosses it within a few one_shot calls;
    _auto_summarize then compresses the conversation and the agent keeps
    going. The task asks for two tool steps + a final answer, so the model
    is called three times and finishes on the third call.
    """
    agent = EgoAgent(
        agent="react",
        model="deepseek-v4-flash",
        max_tokens=10000,
        instruction="You are testing automatic summarization with math tools.",
        tool_registry=build_math_tools(),
        reasoning_effort="low",
        config={
            "session_path": SESSION_DIR,
            "context_manager": {"summarize": 250, "max_iteration": None},
        },
    )
    task = (
        "Use the math tools to add 10 and 5, then subtract 3 from that "
        "result, and tell me the final number."
    )
    _describe(
        "summarize example #1 — react agent (add -> subtract -> answer)",
        agent._react_prompt,
        task,
    )
    result = agent.run(task, max_steps=5)
    _reply(result)
    print(f"tokens_used after run: {agent.tokens_used}")
    print(agent)


@WorkflowSession.capture("summarize_multiply_divide_workflow")
def workflow_summarize_multiply_divide():
    """Summarize example #2: another ReAct agent with math tools.

    Same pattern as example #1 with a different two-step math task. The low
    summarize threshold again forces the context manager to kick in during
    the short multi-call run.
    """
    agent = EgoAgent(
        agent="react",
        model="deepseek-v4-flash",
        max_tokens=10000,
        instruction="You are testing automatic summarization with math tools.",
        tool_registry=build_math_tools(),
        reasoning_effort="low",
        config={
            "session_path": SESSION_DIR,
            "context_manager": {"summarize": 250, "max_iteration": None},
        },
    )
    task = (
        "Use the math tools to multiply 6 and 7, then divide that result "
        "by 2, and tell me the final number."
    )
    _describe(
        "summarize example #2 — react agent (multiply -> divide -> answer)",
        agent._react_prompt,
        task,
    )
    result = agent.run(task, max_steps=5)
    _reply(result)
    print(f"tokens_used after run: {agent.tokens_used}")
