AGENT.md
Project: Multi-Provider LLM Wrapper Framework
1. Project Overview
This project is a Python-based, object-oriented framework for wrapping various Large Language Model (LLM) APIs (e.g., DeepSeek, Grok, Anthropic Claude, OpenAI).
It provides a unified interface (one_shot, chat, tool execution) while allowing provider-specific customizations (e.g., DeepSeek's "thinking mode").

Core Philosophy:

DRY (Don't Repeat Yourself): All common logic (memory management, tool schema generation, interactive loops, and Pythonic list behaviors) lives in the abstract BaseLLM class.

Provider Agnosticism: Adding a new provider requires implementing only 4 abstract methods in a new subclass, inheriting all the "plumbing" automatically.

2. Architecture & File Structure
(Suggested structure based on the current codebase)

text
├── agent_logic
│   ├── agent
│   │   └── __init__.py
│   ├── __init__.py
│   ├── llm
│   │   ├── base.py
│   │   ├── deepseek.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── base.cpython-314.pyc
│   │       ├── deepseek.cpython-314.pyc
│   │       ├── __init__.cpython-314.pyc
│   │       └── newdeepseek.cpython-314.pyc
│   ├── mlops
│   │   ├── __init__.py
│   │   └── mlops.py
│   ├── __pycache__
│   │   └── __init__.cpython-314.pyc
│   └── system
│       ├── agentic_system
│       │   └── __init__.py
│       ├── __init__.py
│       └── workflows
│           └── __init__.py
├── AGENT.md
├── main.py
├── __pycache__
│   └── tools.cpython-314.pyc
└── tools.py

3. The BaseLLM Class (base_llm.py)
Role: The central "plumbing" warehouse. Do not instantiate this directly.

Responsibilities (Already Implemented):

Memory Initialization: Validates and stores system_prompt/user_prompt or a pre-existing memory list.

Property Validation: Enforces rules for model (non-empty string), max_tokens (positive, < 1M), and temperature (0–2).

Tool Management:

_generate_tools_schema(): Converts Python functions into OpenAPI-style JSON schemas using inspect.

_execute_tool_calls(): Executes local functions defined in the tool_registry.

Interactive Chat: chat() provides a built-in terminal conversation loop.

Pythonic Protocols: Implements __len__, __getitem__, __setitem__, __delitem__, __iter__, __repr__, and __str__ to allow the object to behave like a list of messages. __str__ (used by print(agent)) renders the full memory as a clean readable transcript via format_memory(). __setitem__ validates both scalar (agent[i] = {...}) and slice (agent[i:j] = [...]) assignment: every message must be a dict with 'role' and 'content' keys (content may be None, extra keys like tool_calls/tool_call_id allowed). Direct list operations on the live list (agent.memory[i:j] = ...) are the intentional raw tier with no validation.

Abstract Methods (Require Implementation by Subclasses):

_get_api_key()

_create_client()

_build_payload()

one_shot()

4. Implementing a New Provider (Subclass Contract)
To add a new LLM provider (e.g., "Grok", "Claude"), create a new Python file and a class that inherits from BaseLLM.
You must implement the following 4 methods. The rest is handled automatically.

Method	Purpose & Requirements
_get_api_key(self) -> str	Retrieve the API key. Check os.environ for the specific variable name (e.g., XAI_API_KEY). If missing, prompt the user for input and save it to the environment. Return the key as a string.
_create_client(self)	Instantiate the provider's HTTP client (e.g., OpenAI(api_key=..., base_url=...) or a custom requests session). Returns the client object (stored as self._client).
_build_payload(self) -> dict	Build the exact request dictionary the API expects. Start with the common fields (model, messages, max_tokens, temperature) and conditionally add provider-specific fields (e.g., reasoning_effort for DeepSeek, top_p for Anthropic).
one_shot(self) -> dict	The core execution logic. Call self._build_payload(), send the request via self._client, parse the response. Crucial: Append the assistant's message (including tool_calls if present) to self.memory before returning. Return a standardized dictionary: {"reasoning": str/None, "content": str, "tool_calls": list}.
Important: If the new provider has unique parameters (e.g., top_k), define them as standard properties with getters/setters inside the subclass, then include them in _build_payload.

5. Extending Tools
To allow the LLM to call custom Python functions:

Define your Python functions in a separate module (e.g., tools.py) or use the built-in tools shipped with the framework.

Pass a tool_registry dictionary mapping string names to the function objects into the constructor.

Nothing is registered automatically: passing `directory` only scopes the agent prompts to that workspace. If you want file access, add the built-in file tools yourself.

Built-in tools live in agent_logic/agent/builtin_tools/:
- file.py: build_file_tools(directory) -> ls, read_file, write_file, edit_file, delete, glob, grep
- math.py: build_math_tools() -> add, subtract, multiply, divide

The framework automatically generates the JSON schemas and handles execution via _execute_tool_calls.

Example (file + math tools merged by hand):

python
from agent_logic.agent.builtin_tools.file import build_file_tools
from agent_logic.agent.builtin_tools.math import build_math_tools

registry = {**build_math_tools(), **build_file_tools("/abs/workspace")}
agent = EgoAgent(agent="react", model="deepseek-v4-flash", max_tokens=10000,
                 directory="/abs/workspace", tool_registry=registry)
6. Usage Pattern (Entry Point)
When writing the main application, instantiate the specific provider class directly.

Initialization:

python
from deepseek_llm import DeepseekLLM

agent = DeepseekLLM(
    model="deepseek-v4-flash",
    system_prompt="You are a math tutor.",
    user_prompt="Explain quantum physics.",
    temperature=0.7,
    reasoning_effort="high"  # DeepSeek-specific
)
Calling the Agent:

agent.one_shot(): Sends the conversation and returns the assistant's response dict.

agent.chat(): Starts an interactive terminal session.

agent.memory: Access/modify the conversation history directly.

print(agent) / agent.format_memory(): Render the entire memory as a clean transcript — System prompt / User / Assistant turns plus every Tool call (name, arguments, result) — for terminal testing.

len(agent), agent[0]: Use standard Python list behaviors.

Item/slice assignment (validated tier):

    agent[i] = {"role": "assistant", "content": "..."}   # single message; raises ValueError unless it's a dict with role + content
    agent[i:j] = [{"role": "user", "content": "..."}, ...]  # slice; requires a list, every item checked with the same rule
    agent2[i:j] = agent1[i:j]                              # inter-agent transfer; passes instantly — real memories are always well-formed

Raw tier (no validation, on purpose): agent.memory returns the live list, so plain list operations mutate memory directly without any checks — agent.memory[0:1] = [{"role": "user"}], del agent.memory[2]. Use only when you deliberately want raw control.

Cross-agent transfers share dict objects (shallow): agent2[i:j] = agent1[i:j] aliases the same dicts, so later in-place mutations of one agent's messages leak into the other. Use save_memory() (deep copy) or copy.deepcopy(...) when the two agents must stay independent.

Memory getter/setter pair (BaseLLM, inherited by every LLM and agent):

    agent.trim_memory([from, to])  # setter — deletes that 1-based range
                                   # in place; system prompts are kept.
    carry = agent.save_memory([from, to])
                                   # getter — deep-copies that range
                                   # (system prompts excluded by default)
                                   # without touching the source memory.

Handoff recipe — carry one conversation into another object:

python
carry = agent1.save_memory([1, len(agent1.memory)])
agent2.memory = [{"role": "system", "content": agent2._react_prompt}, *carry]


7. Environment Variables
The following API keys are required in the environment depending on which
provider you are using. The provider raises a RuntimeError when the key is
missing (no interactive prompt; a commented-out CLI fallback exists in the
provider modules):

DeepSeek: DEEPSEEK_API_KEY

Grok (xAI): XAI_API_KEY

OpenAI: OPENAI_API_KEY

Anthropic: ANTHROPIC_API_KEY

(These are sourced via os.environ in the respective subclass's _get_api_key method).

8. Developer Guidelines (Rules for AI Agents)
When contributing to this repository, adhere to these strict guidelines:

Never edit BaseLLM for provider-specific logic. If you need a new parameter specific to a single API, add it to the subclass, not the parent.

Standardize the one_shot return dict. It must contain exactly the keys reasoning, content, and tool_calls. Keep tool_calls parsed as a list of dicts with id, name, and arguments (dict).

Don't override inherited methods unless absolutely necessary. Methods like _execute_tool_calls, chat, and the dunders are perfectly generic and should not be changed in subclasses.

Always call super().__init__(...) in the subclass's __init__ before setting provider-specific attributes.

9. MLOps Sessions (WorkflowSession)
The MLOps layer (ego_py/mlops) records complete workflows into session logs — one numbered JSON per wrapped workflow. Full documentation: ego_py/mlops/sessions.py.

`WorkflowSession` itself takes NO Config. Instead, every LLM-based object (LLM(...), EgoAgent(...), providers, agents — all inherit it from BaseLLM) carries its own plain `config` dict (str -> str by default, plus the optional nested `config["context_manager"]` dict). The session directory is read from each recorded object's `config["session_path"]` at the moment its run()/one_shot()/tool call fires:

- the first recorded object's `config["session_path"]` must be a non-empty string naming an EXISTING directory -> sessions are saved there (nothing is auto-created);
- a missing/invalid config raises RuntimeError immediately with a message saying exactly what is wrong (never silently skipped);
- every later recorded object in the same workflow must resolve to the SAME directory (one session = one directory), otherwise RuntimeError.

```python
from ego_py import EgoAgent, WorkflowSession

SESSION_DIR = "/existing/sessions/dir"

@WorkflowSession.capture("agents_workflow")
def workflow_agents():
    # any number of agents and raw LLM calls...
    agent1 = EgoAgent(agent="react", model="deepseek-v4-flash",
                      max_tokens=10000, instruction="...",
                      config={"session_path": SESSION_DIR})
    agent2 = EgoAgent(agent="plan-execute", model="deepseek-v4-flash",
                      max_tokens=10000, instruction="...",
                      config={"session_path": SESSION_DIR})
    agent1.run(task)   # becomes step "1" in the session JSON
    agent2.run(task)   # becomes step "2"
```

Calling workflow_agents() saves <SESSION_DIR>/agents_workflow-<timestamp>.json with numbered steps "1", "2" — each step contains the agent metadata, the run result, the full conversation memory, every LLM call and every tool execution. To save two separate sessions, write two decorated workflow functions. Each call of a decorated function gets a fresh session, so calling the same workflow twice (or concurrently from two threads) always yields separate session files. A workflow that records no objects produces no file and no error.

Rules:
- Workflows live in system/ as functions decorated with @WorkflowSession.capture(...); main.py imports and runs them.
- WorkflowSession patches one_shot/_execute_tool_calls/run at runtime and restores them on exit — never edit agent or provider code to add logging.
- Do not nest WorkflowSession in the same thread (it raises RuntimeError); use separate decorated workflows instead.
- Threads: contextvars are NOT inherited by newly spawned threading.Thread objects, so worker threads record nothing unless the context is propagated (threading.Thread(target=contextvars.copy_context().run, args=(fn,))).

### config["context_manager"] (automatic context management)

Any LLM-based object can opt into automatic context management by adding a nested dict to its `config`:

```python
config={
    "session_path": SESSION_DIR,
    "context_manager": {
        "summarize": 30000,     # token threshold; None = disabled
        "max_iteration": 10,    # one_shot iteration threshold; None = disabled
    },
}
```

- `summarize` (int|None): when cumulative `tokens_used` reaches this value, the current conversation is compressed into a single summary message via `BaseLLM.summarize` (2000-token summary budget) and the counter resets.
- `max_iteration` (int|None): when the number of `one_shot` turns reaches this value, `trim_memory` drops old non-system messages, keeping the system prompt(s) and the most recent 4 non-system messages; the iteration counter resets.
- Both `None` (the default) disables context management entirely. The checks run at the top of every provider `one_shot()` call.

### config["file"] and config["agents.md"]

Two more recognized config keys, validated by `ConfigModel` in
`ego_py/llm/config.py` (the single source of truth for the config schema):

```python
config = {
    "session_path": SESSION_DIR,
    "file": True,                     # bool, default False
    "agents.md": "/path/to/context",  # str | None
}
```

- `file` (bool): when `True`, `build_file_tools(<workspace directory>)` is
  loaded automatically into `tool_registry` + the tools schema (the same way
  the `use_skill` tool is auto-loaded for skills). The workspace comes from
  the agent's `directory=` parameter; a raw `LLM(...)` object (which has no
  directory) raises `ValueError`. When `False` (the default) nothing changes.
- `agents.md` (str): a directory scanned **recursively**; the contents of
  every file named exactly `AGENTS.md` found under it are concatenated and
  injected into the system prompt (agents keep the block across run()/plan()
  memory rebuilds). No matches -> the prompt is unchanged.

