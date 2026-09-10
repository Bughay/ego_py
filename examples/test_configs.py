"""
Config test workflows — every known config key, turned ON and OFF across
2 LLMs and 3 agents. Each object lives in its OWN decorated workflow, so
every workflow run saves its own numbered session JSON (the session
directory is read from each object's config["session_path"] at record time
by WorkflowSession).

All configs are written as plain dicts; BaseLLM validates and normalizes
every one through the ConfigModel schema from egoai/llm/config.py (the
single source of truth for the known config keys) at construction time:

    self.config = ConfigModel.from_dict(config).to_dict()

Combination matrix (workflow -> object -> config keys ON):

    workflow_llm_off      LLM             everything OFF (session_path only)
    workflow_llm_on       LLM             context_manager.summarize ON,
                                          skills ON, agents.md ON,
                                          extra metadata key; file OFF
    workflow_agent_off    react           everything OFF (session_path only)
    workflow_agent_file   plan-execute    file ON (auto file tools),
                                          context_manager.max_iteration ON,
                                          summarize/skills/agents.md OFF
    workflow_agent_all    plan-react      everything ON (file + skills +
                                          agents.md + summarize)

Run directly:

    python examples/test_configs.py
"""
import os
import sys

# Make `egoai` importable when the script is run directly from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from egoai import EgoAgent, LLM, WorkflowSession  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "example_directory"))
SKILLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "skills"))
SESSION_DIR = os.path.join(PROJECT_ROOT, "sessions")

# WorkflowSession never auto-creates directories: they must exist up front.
os.makedirs(SKILLS_DIR, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"CONFIG TEST: {title}")
    print("=" * 70)


def _show(obj, label: str) -> None:
    """Print the observable effects of an object's config."""
    print(f"\n[{label}] normalized config : {obj.config!r}")
    print(f"[{label}] skills_metadata   : {obj.skills_metadata!r}")
    tools = obj.tools or []
    names = [t["function"]["name"] for t in tools]
    print(f"[{label}] tool schemas      : {names or '(none)'}")
    print(f"[{label}] context_manager  : {obj.config.get('context_manager')!r}")


# --------------------------------------------------------------------------- #
# 1. LLM — all configs OFF                                                    #
# --------------------------------------------------------------------------- #
@WorkflowSession.capture("llm_off_workflow")
def workflow_llm_off():
    """LLM #1: only session_path (required for the session to save).

    No skills, no agents.md, no file tools, no context manager — the
    object behaves exactly like a plain dict-config LLM."""
    llm = LLM(
        model="deepseek-v4-flash",
        system_prompt="You are a helpful, brief assistant.",
        user_prompt="In one sentence, what is a config dict?",
        reasoning_effort="low",
        config={"session_path": SESSION_DIR},
    )
    _section("LLM — all configs OFF")
    _show(llm, "llm")
    result = llm.one_shot()
    print(f"[llm] reply: {result['content']!r}")


# --------------------------------------------------------------------------- #
# 2. LLM — skills + agents.md + summarize ON                                  #
# --------------------------------------------------------------------------- #
@WorkflowSession.capture("llm_on_workflow")
def workflow_llm_on():
    """LLM #2: configs ON.

    - skills:   use_skill tool is auto-loaded (see _show below).
    - agents.md: the repo AGENTS.md is injected into the system prompt.
    - context_manager.summarize = 200: as soon as 200 cumulative tokens
      are used, the conversation is compressed into a single summary.
    - file stays OFF (raw LLM objects have no directory) and an unknown
      metadata key ("owner") is preserved verbatim by ConfigModel."""
    llm = LLM(
        model="deepseek-v4-flash",
        system_prompt="You are a brief assistant under config test.",
        user_prompt="Answer in one short sentence.",
        reasoning_effort="low",
        config={
            "session_path": SESSION_DIR,
            "skills": SKILLS_DIR,
            "agents.md": PROJECT_ROOT,
            "context_manager": {"summarize": 200, "max_iteration": None},
            "owner": "test_configs",  # unknown key -> str metadata, kept
        },
    )
    _section("LLM — skills + agents.md + summarize ON (file OFF)")
    _show(llm, "llm")

    user_prompt = llm.memory[1]["content"]
    for i in range(1, 7):
        result = llm.one_shot()
        first_non_system = next(
            (m for m in llm.memory if m.get("role") != "system"), None
        )
        fired = (
            first_non_system is not None
            and first_non_system.get("content") != user_prompt
        )
        print(
            f"[llm] call {i}: tokens_used={llm.tokens_used}, "
            f"memory={len(llm.memory)} messages, "
            f"summarized={'YES' if fired else 'no'}"
        )
        if fired:
            break
    print(f"[llm] final memory ({len(llm.memory)} messages):")
    print(llm)


# --------------------------------------------------------------------------- #
# 3. react agent — all configs OFF                                            #
# --------------------------------------------------------------------------- #
@WorkflowSession.capture("agent_off_workflow")
def workflow_agent_off():
    """Agent #1 (react): only session_path.

    No directory, no tools, no context manager — the agent answers
    directly, and tool schemas stay empty (see _show below)."""
    agent = EgoAgent(
        agent="react",
        model="deepseek-v4-flash",
        max_tokens=10000,
        instruction="You are a helpful assistant. Answer directly.",
        reasoning_effort="low",
        config={"session_path": SESSION_DIR},
    )
    _section("react agent — all configs OFF")
    _show(agent, "agent")
    result = agent.run("In one sentence, what is the ReAct loop?", max_steps=3)
    print(f"[agent] reply: {result['content']!r}")
    print(f"[agent] memory: {len(agent.memory)} messages")


# --------------------------------------------------------------------------- #
# 4. plan-execute agent — file + max_iteration ON                             #
# --------------------------------------------------------------------------- #
@WorkflowSession.capture("agent_file_workflow")
def workflow_agent_file():
    """Agent #2 (plan-execute): config['file'] = True ON.

    The built-in file tools (ls, read_file, write_file, glob, grep, ...)
    are auto-loaded for the workspace directory — no hand-built
    tool_registry — and context_manager.max_iteration = 3 is ON (old
    non-system messages are trimmed once a step passes 3 one_shot turns).
    summarize / skills / agents.md stay OFF."""
    agent = EgoAgent(
        agent="plan-execute",
        model="deepseek-v4-flash",
        max_tokens=10000,
        instruction="You are testing the config['file'] auto file tools.",
        directory=WORKSPACE_DIR,
        reasoning_effort="low",
        config={
            "session_path": SESSION_DIR,
            "file": True,
            "context_manager": {"summarize": None, "max_iteration": 3},
        },
    )
    _section("plan-execute agent — file + max_iteration ON")
    _show(agent, "agent")

    task = (
        "Using the file tools, do this in the workspace: "
        "(1) list the directory with ls, "
        "(2) write a file named config_file_probe.txt containing the "
        "listing plus the sentence 'config file=True works', and "
        "(3) read the file back to confirm its contents."
    )
    result = agent.run(task, max_steps=8)
    print(f"[agent] plan: {result['plan'].plan!r}")
    replies = [r.get("content") for r in result["results"] if r]
    print(f"[agent] step replies: {replies!r}")
    print(f"[agent] memory: {len(agent.memory)} messages, "
          f"iterations={agent._iterations}, tokens_used={agent.tokens_used}")

    probe = os.path.join(WORKSPACE_DIR, "config_file_probe.txt")
    if os.path.isfile(probe):
        print(f"[agent] {probe} was written by the agent:")
        with open(probe, encoding="utf-8") as fh:
            print(fh.read())
    else:
        print("[agent] the model did not write config_file_probe.txt "
              "(tool execution is recorded in the session JSON)")


# --------------------------------------------------------------------------- #
# 5. plan-react agent — everything ON                                         #
# --------------------------------------------------------------------------- #
@WorkflowSession.capture("agent_all_workflow")
def workflow_agent_all():
    """Agent #3 (plan-react): every config key ON.

    file (auto file tools) + skills (use_skill tool) + agents.md (repo
    context injected into the prompts) + context_manager.summarize = 250
    (the conversation is compressed once 250 tokens accumulate)."""
    agent = EgoAgent(
        agent="plan-react",
        model="deepseek-v4-flash",
        max_tokens=10000,
        instruction=(
            "You are testing configs: file tools, the poetry_critic skill "
            "and automatic summarization, all enabled through config."
        ),
        directory=WORKSPACE_DIR,
        reasoning_effort="low",
        config={
            "session_path": SESSION_DIR,
            "file": True,
            "skills": SKILLS_DIR,
            "agents.md": PROJECT_ROOT,
            "context_manager": {"summarize": 250, "max_iteration": None},
        },
    )
    _section("plan-react agent — file + skills + agents.md + summarize ON")
    _show(agent, "agent")

    task = (
        "Load the poetry_critic skill, write a short 4-line poem about "
        "spring, critique it with the skill, then save both the poem and "
        "the critique into a file named config_all_probe.txt in the "
        "workspace directory."
    )
    result = agent.run(task, max_steps=8)
    print(f"[agent] plan: {result['plan'].plan!r}")
    final = (result["results"] or [{}])[-1]
    print(f"[agent] final step reply: {final.get('content')!r}")
    print(f"[agent] memory: {len(agent.memory)} messages, "
          f"tokens_used={agent.tokens_used}")

    probe = os.path.join(WORKSPACE_DIR, "config_all_probe.txt")
    if os.path.isfile(probe):
        print(f"[agent] {probe} was written by the agent:")
        with open(probe, encoding="utf-8") as fh:
            print(fh.read()[:2000])
    else:
        print("[agent] the model did not write config_all_probe.txt "
              "(tool execution is recorded in the session JSON)")


def main():
    from dotenv import load_dotenv

    load_dotenv()
    workflow_llm_off()
    workflow_llm_on()
    workflow_agent_off()
    workflow_agent_file()
    workflow_agent_all()


if __name__ == "__main__":
    main()
