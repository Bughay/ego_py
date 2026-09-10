# LLM & EgoAgent — User Guide

The framework has two entry points:

- **`LLM`** — a raw, single model call with memory, tools and structured output.
- **`EgoAgent`** — an agent that solves a whole task, picking one of four strategies and combining it with a provider.

Both are factories: the `model` string decides the provider (`deepseek*` → `DeepseekLLM`, `grok*` → `GrokLLM`) and, for `EgoAgent`, the `agent` string decides the strategy. Everything else is passed straight through as constructor parameters.

This guide has three parts:

1. **The objects and their purpose** — what `LLM`, `EgoAgent` and the four agent strategies actually are.
2. **Parameters and what they are for** — every constructor parameter and every `config` key.
3. **Tips for using each parameter** — when to raise temperature, how to size `max_tokens`, when thinking helps, and so on.

---

## 1. The objects and their purpose

### 1.1 `LLM` — one conversation, one model call

```python
from egoai import LLM

llm = LLM(
    model="deepseek-v4-flash",
    system_prompt="You are a precise assistant.",
    user_prompt="Explain quantum entanglement in three sentences.",
)
result = llm.one_shot()   # -> {"reasoning": ..., "content": ..., "tool_calls": [...]}
```

`LLM` returns a provider instance (`DeepseekLLM` or `GrokLLM`). It is the smallest usable object in the framework: it holds a conversation in `memory`, sends it to the provider's API on `one_shot()`, appends the assistant reply to memory and returns a normalized result dict. It also carries all the structured-output helpers (`extract`, `classify`, `summarize`) and the memory helpers, because everything lives in the shared `BaseLLM` base class.

Use `LLM` when you want **one step**: one question, one generation, one extraction, one classification, one summary — or when you want to drive the conversation manually by calling `one_shot()` repeatedly.

### 1.2 `EgoAgent` — a task solver built on top of `LLM`

```python
from egoai import EgoAgent
from egoai.builtin_tools.file import build_file_tools

workspace = "/abs/path/to/workspace"

agent = EgoAgent(
    agent="react",
    model="deepseek-v4-flash",
    max_tokens=10000,
    instruction="You are a careful coding assistant.",
    directory=workspace,
    tool_registry=build_file_tools(workspace),
)
result = agent.run("Find the config file and summarize what it contains.", max_steps=20)
print(result["content"])
```

`EgoAgent` combines an agent strategy class with a provider class and hands the task to `run()`. An agent is still an `LLM` underneath — same memory, same tools, same `one_shot()`, same config — but instead of you calling `one_shot()` once, the agent calls it in a loop, executes the tools the model asks for, appends the observations back into memory, and keeps going until the model stops calling tools (or the step limit is reached).

Use `EgoAgent` when a task needs **multiple turns, tool use, or decomposition**.

### 1.3 The four agent strategies

| `agent=` | What it does | Best for |
|---|---|---|
| `"react"` | One Reason → Act → Observe loop. The model thinks, calls tools, sees the observations, and repeats until it produces a final answer with no tool calls. | A single self-contained task that needs tools (edit a file, search a repo, compute something). |
| `"plan-execute"` | The planner LLM writes a plan first (extracted into a structured `PlanExecute` object), then each plan step is executed sequentially with the executor prompt. | Tasks that are easier to break into ordered steps than to solve in one loop. Cheap: no per-step tool loop unless the executor asks. |
| `"plan-react"` | Plan first, then run a **full ReAct loop** for every plan step. Each step gets the overall plan as context. | Multi-step tasks where each step itself needs tools and several turns. |
| `"agent-swarm"` | Same as `plan-react`, but the plan steps run **concurrently**, one fresh agent per step in a thread pool. | Independent steps where wall-clock time matters and the steps do not conflict. |

### 1.4 What every object shares (inherited from `BaseLLM`)

- **Memory** — `agent.memory` is a live list of message dicts. The object also behaves like a list: `len(agent)`, `agent[i]`, `agent[i] = {...}`, `agent[i:j] = [...]`, `del agent[i]`, iteration, and `print(agent)` (a formatted transcript via `format_memory()`).
- **Tooling** — a `tool_registry` of plain Python functions is converted into JSON schemas and executed automatically; tool results are appended to memory as `tool` messages.
- **Structured workflows** —
  - `extract(schema, example=None, instruction=None)` → JSON dict parsed against a description schema (raw string if parsing fails).
  - `classify(schema, example=None, instruction=None)` → JSON dict with every key constrained to its allowed values/range; retries twice on invalid output, then raises `ValueError`.
  - `summarize(context, max_tokens)` → summary string. Does **not** touch the current memory or the object's `max_tokens`.
- **Memory surgery** — `trim_memory([from, to])` deletes a 1-based inclusive range in place (system prompts are never deleted); `save_memory([from, to], skip_system=True)` deep-copies a range without mutating the source.
- **Automatic context management** — opt-in through `config["context_manager"]` (summarize at a token threshold, trim after N turns).

### 1.5 Methods and their return shapes

| Method | Returns |
|---|---|
| `LLM.one_shot()` | `{"reasoning": str \| None, "content": str \| None, "tool_calls": [{"id", "name", "arguments": dict}]}`. `content` can be `None` when the model only asked for tools. |
| `ReActAgent.onestep()` | The `one_shot()` dict plus `"observations": list[str] \| None` (the tool results just executed). |
| `ReActAgent.run(task, max_steps=50)` | The final `onestep()` dict. |
| `PlanExecuteAgent.plan(task)` | A `PlanExecute(plan: str, execute_steps: list[str])` dataclass. |
| `PlanExecuteAgent.run(task, max_steps=10)` | `{"plan": PlanExecute, "results": [one_shot dict per step]}`. |
| `PlanReactAgent.run(task, max_steps=50)` | `{"plan": PlanExecute, "results": [...], "memories": [full conversation per step]}`. |
| `AgentSwarm.run(task, max_steps=50)` | Same as `plan-react`: `{"plan", "results", "memories"}`. |

`run()` rebuilds memory from the agent's system prompt and the new task, so it always starts a fresh conversation. `onestep()` continues the current memory instead.

### 1.6 API keys

API keys are read from the environment when the provider client is created (at construction time). A missing key raises `RuntimeError` immediately:

| Provider | Environment variable |
|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` |
| Grok (xAI) | `XAI_API_KEY` |

Load them before constructing the object (e.g. `from dotenv import load_dotenv; load_dotenv()`). Keys are never accepted as constructor arguments.

> Note: the `BaseLLM` docstring mentions an "interactive chat loop", but this version of the codebase has no `chat()` method. The public surface is `one_shot()`, `extract()`, `classify()`, `summarize()` on every object, plus `plan()` / `onestep()` / `run()` on agents.

---

## 2. Parameters and what they are for

All parameters below are accepted by `LLM(...)`; the agent-only ones are accepted by `EgoAgent(...)` in addition. Parameters are validated at construction time (property setters in `BaseLLM`/providers; `config` through `ConfigModel` in `egoai/llm/config.py`).

### 2.1 Conversation setup

| Parameter | Type / accepted values | Default | Purpose |
|---|---|---|---|
| `system_prompt` | `str` | — | The role / instructions / context, sent as the first `system` message. |
| `user_prompt` | `str` | — | The first `user` message. |
| `memory` | `list[dict]` | — | A ready-made conversation to start from. Each item must be a dict with `role` and `content` (extra keys such as `tool_calls` / `tool_call_id` are allowed). |

**Rule:** provide **either** both prompts **or** `memory`, never a mix.

- Only `system_prompt` (no `user_prompt`) → `ValueError`.
- Only `user_prompt` → `ValueError`.
- `memory` together with either prompt → `ValueError`.

### 2.2 Generation

| Parameter | Type / accepted values | Default | Purpose |
|---|---|---|---|
| `model` | `str` (required). DeepSeek: `deepseek-v4-flash`, `deepseek-v4-pro`, `deepseek-v4-flash-vision-exp`. Grok: `grok-4.6`, `grok-4.6-fast`, `grok-4.6-non-reasoning`. | — | Selects the provider (by prefix) and the exact model. Also decides which API key is required. |
| `max_tokens` | `int`, `1 … 1_000_000` | `10000` | Cap on the model's **generated** tokens. |
| `temperature` | `float`, `0 … 2` | `0.5` (DeepSeek) / `1.0` (Grok) | Sampling randomness. Grok stores the value but does not send it (see tips). |
| `reasoning_effort` | DeepSeek: `None \| "low" \| "medium" \| "high"`. Grok: `None \| "low" \| "medium" \| "high" \| "xhigh"`. | `None` | Thinking depth. On DeepSeek, setting it enables thinking mode; `None` disables it. |
| `response_format` | object, e.g. `{"type": "json_object"}` | `None` | Forces an output format supported by the provider. |

### 2.3 Tools

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `tool_registry` | `dict[str, callable]` | `None` | Tools the model may call. The key is the tool name, the value is any plain Python function. The framework reads the signature and docstring and generates the OpenAPI-style schema. |
| `tools` | `list[dict]` | `None` | Raw tool schemas. When provided, schema auto-generation from `tool_registry` is skipped. |
| `tool_choice` | provider-specific value, e.g. `"auto"` | `None` | Tool-selection policy, forwarded to the API when set. |

When a tool is called, the framework looks it up in `tool_registry`, runs it with the parsed arguments, converts the result to a string (dicts/lists via `json.dumps`), and makes it available both as an observation and as a `tool` message in memory. Unknown names and exceptions become error strings, not crashes.

### 2.4 Agent-only parameters

| Parameter | Type / accepted values | Default | Purpose |
|---|---|---|---|
| `agent` | `"react" \| "plan-execute" \| "plan-react" \| "agent-swarm"` (required by `EgoAgent`) | — | Picks the strategy. |
| `instruction` | `str` | `None` | Extra instructions appended to the agent's base prompt(s). |
| `directory` | `str` — an existing directory, no `..` segments | current working directory | Scopes the agent prompts to a workspace. Also the workspace used when `config["file"]` is set. |
| `memory` | `list[dict]` | `None` | **react only** — a custom starting conversation. |
| `max_workers` | `int > 0` | `5` | **agent-swarm only** — thread-pool size and the worker cap advertised to the planner. |
| `max_steps` | `int > 0` — argument of `run()` | `50` for `react` / `plan-react` / `agent-swarm`; `10` for `plan-execute` (per plan step) | Caps how many model turns a tool loop may take before it is forced to stop. |

### 2.5 The `config` dictionary

`config` is a single plain dict carried by every object. It is validated and normalized once at construction by `ConfigModel` (unknown keys must still be `str → str`).

| Key | Type / values | Default | Purpose |
|---|---|---|---|
| `session_path` | `str` — an existing directory | absent | Directory where `WorkflowSession` records this object's calls. Required in practice when the workflow is decorated. |
| `file` | `bool` or `"read-only"` | `False` | `True` auto-loads `build_file_tools(directory)`; `"read-only"` auto-loads only `ls`, `read_file`, `glob`, `grep`. |
| `skills` | `str` — a directory | absent | Directory of `.md` skill files. Skills are discovered at init, advertised in the system prompt, and loadable through the auto-injected `use_skill` tool. |
| `agents.md` | `str` — a directory | absent | Scanned **recursively**; every file named exactly `AGENTS.md` has its contents concatenated and injected into the system prompt, under a heading with its relative path. |
| `context_manager.summarize` | `int > 0` or `None` | `None` (off) | When cumulative `tokens_used` reaches this value, the conversation is compressed into a single summary message (2000-token budget) and the counter resets. |
| `context_manager.max_iteration` | `int > 0` or `None` | `None` (off) | After this many `one_shot()` turns, old non-system messages are trimmed, keeping the system prompt(s) and the most recent 4 non-system messages; the counter resets. |
| any other key | `str → str` | — | Free metadata (e.g. owner, tags), preserved verbatim. |

There is no other way to enable sessions, skills, `AGENTS.md` or context management — everything is declared here.

### 2.6 Built-in tool builders

Nothing is registered automatically from `directory` alone (except through `config["file"]`). These builders exist so you can pass a registry explicitly:

| Builder | Tools |
|---|---|
| `build_file_tools(directory)` | `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, `bash` |
| `build_read_only_file_tools(directory)` | `ls`, `read_file`, `glob`, `grep` |
| `build_math_tools()` | `add`, `subtract`, `multiply`, `divide` |

They return plain dicts, so they can be merged:

```python
from egoai.builtin_tools.file import build_file_tools
from egoai.builtin_tools.math import build_math_tools

registry = {**build_math_tools(), **build_file_tools("/abs/workspace")}
```

File tools are bound to the workspace: relative paths resolve against it, `..` is rejected, paths that would escape the workspace are rejected, and `bash` refuses absolute/home paths and `..` tokens.

---

## 3. Tips for using each parameter

### 3.1 `model`

- Matching is by prefix (`deepseek*`, `grok*`) but the exact name is checked against the provider allowlist. A typo raises `ValueError` at construction — good, fail early.
- `deepseek-v4-flash` / `grok-4.6-fast` — cheap and fast; the right default for classification, extraction, short answers, and most agent tool loops.
- `deepseek-v4-pro` / `grok-4.6` — use when the task needs deeper reasoning, long context comprehension, or better planning.
- `deepseek-v4-flash-vision-exp` — use when the prompt contains images.
- `grok-4.6-non-reasoning` — use when you want the fastest Grok responses and no thinking overhead.
- Provider choice is also an operational choice: different API keys, rate limits and outages. For critical workflows, being able to switch `deepseek-v4-flash` → `grok-4.6-fast` by changing one string is the point.

### 3.2 `system_prompt` / `user_prompt` / `memory`

- Use the two prompts for a fresh one-off call; use `memory` when you already have a conversation (a saved transcript, a hand-off from another agent, or a restored session).
- Since you cannot pass both, the common pattern for "continue an existing conversation" is: construct with `memory=...`, or construct normally and then set `agent.memory = [...]` (the setter validates that every message is a dict with `role` and `content`).
- Keep the `system` prompt stable and put task-specific content in the `user` message. Agents rebuild memory from their own prompts on every `run()` — manual `memory` edits survive only until the next run.
- A system prompt with explicit output constraints ("answer in exactly 3 bullets", "return only JSON") reliably beats hoping the model infers the format.

### 3.3 `max_tokens`

This caps **output only**. It never limits the input/context, and it counts reasoning tokens too on thinking models, so leave headroom when `reasoning_effort` is set.

Practical sizes:

| Task | Sensible `max_tokens` |
|---|---|
| Classify / route / one-word answers | 50–200 |
| JSON extraction (`extract`) | 300–1000 |
| Chat / explanation answer | 500–1500 |
| Summaries | 500–2000 |
| Code generation | 4000–8000 |
| Agents (`EgoAgent`) | 8000–16000 |

- Too small is worse than too large: the model gets cut off mid-JSON and `extract`/`classify` parsing fails. If you see truncated-looking output, raise `max_tokens` first.
- For agents, remember a turn can contain reasoning + a large tool-call payload; 10000 is a reasonable floor, not a ceiling.
- The internal auto-summarizer always uses a 2000-token budget; `summarize(context, max_tokens)` temporarily overrides the object's `max_tokens` for that one call and restores it afterwards.

### 3.4 `temperature`

| Range | Use it for |
|---|---|
| `0.0 – 0.3` | Deterministic work: classification, extraction, factual Q&A, code, anything you will parse or test. |
| `0.4 – 0.7` | Balanced assistants (DeepSeek's default 0.5 sits here). Good general-purpose range. |
| `0.8 – 1.2` | Creative work: brainstorming, marketing copy, poetry, alternative suggestions. |
| `> 1.2` | Deliberate randomness. Expect surprises, including incoherence; useful for idea generation only. |

- For `extract`/`classify` chains, set temperature to 0.0–0.2: the added randomness buys nothing and hurts schema compliance.
- For agents doing file edits or multi-step planning, lower is safer (0.2–0.5): fewer inconsistent tool arguments.
- **Grok caveat:** Grok stores whatever `temperature` you pass but never sends it in the payload; xAI applies its own fixed behavior. On Grok, control output with the prompt and `reasoning_effort`, not temperature.

### 3.5 `reasoning_effort`

Thinking is a latency/cost dial: more effort means more internal reasoning tokens before the answer.

| Level | When to use |
|---|---|
| `None` | Simple formatting, routing, short answers, high-volume calls. On DeepSeek this explicitly disables thinking mode. |
| `"low"` | Routine tool use, "find the file and report it", light classification. |
| `"medium"` | General-purpose agents and multi-step tasks — a good default when you want quality without the slowest path. |
| `"high"` | Math, planning, debugging, multi-constraint reasoning, architecture decisions. |
| `"xhigh"` (Grok only) | The hardest problems, when latency does not matter. |

- Turn thinking **off** (`reasoning_effort=None` on DeepSeek) for the classification/extraction/summarization calls inside a workflow: the helpers are prompt-constrained and do not need it.
- Thinking tokens are billed and count toward `tokens_used`, so they can trigger `config["context_manager"]["summarize"]` earlier than expected.
- When the provider returns it, the thinking text is available in `result["reasoning"]` — useful for debugging why an agent chose a tool.

### 3.6 `response_format`

- Set `response_format={"type": "json_object"}` when you will parse JSON yourself from a raw `one_shot()`.
- You rarely need it with `extract()` / `classify()`: both force JSON internally for their own call and temporarily switch tools off, so tool chatter cannot pollute the structured answer.
- Do not combine a forced JSON format with tool calling in the same call — pick one: structured answer or tool use.

### 3.7 `tool_registry`

- Write a precise docstring on every tool: it becomes the description the model reads. "Add two numbers together and return their sum." is exactly right; "add" is not.
- Annotate parameters (`int`, `float`, `str`, `bool`, `list`): annotations become the schema types (`int`/`float` → number, `str` → string, `bool` → boolean, `list` → array). Unannotated parameters fall back to `string`. Parameters without defaults become required.
- Return simple values. The framework stringifies results; dicts and lists are JSON-encoded. Errors thrown by a tool are caught and returned to the model as `Error executing <tool>: ...`, so the agent can recover.
- Keep tools narrow and safe. They are plain in-process Python functions — the model can call anything you register. For read-only analysis register `build_read_only_file_tools(...)`; only register `bash`/write access when the agent genuinely needs it.
- You can merge registries freely: `{**build_math_tools(), **build_file_tools(dir)}`. Later keys win on name collisions.
- Tool schemas are generated only when `tools` is `None`; if you pass `tools`, make sure the names exist in `tool_registry` or calls will return "Unknown tool".

### 3.8 `tools` and `tool_choice`

- Use `tools` only when you need schemas the auto-generator cannot express (e.g. nested objects, enums, custom descriptions). It fully replaces generated schemas, so keep it in sync with the registry.
- `tool_choice` is the provider's policy knob — typically `"auto"`. Set it only if the provider documents forcing (`"required"`) or forbidding (`"none"`) tool use. Leaving it `None` sends nothing.

### 3.9 `instruction` (agents)

- Think "house rules", not "task". The task goes to `run(task)`; `instruction` says who the agent is and how it should behave.
- It is appended to the planner **and** executor prompts (and the ReAct prompt), so write it once and it applies to the whole strategy.
- Keep it short (1–5 sentences) and concrete: role, constraints, conventions ("all paths relative to the workspace", "never touch `.env`", "report results as a markdown list").
- Do not repeat what the built-in prompt already says (e.g. "you can call tools") — repetition wastes context and can confuse the planner.

### 3.10 `directory` (agents)

- Always pass an explicit absolute path; relying on the current working directory makes behavior depend on where the program was launched.
- It scopes the prompts ("All work must stay inside this directory") and is the workspace used by `config["file"]`. It does **not** register tools by itself.
- The directory must already exist and must not contain `..`; construction raises `ValueError` otherwise. Create workspaces with `os.makedirs(..., exist_ok=True)` before constructing the agent.
- Use one dedicated project directory per agent. Pointing agents at `$HOME` or a repo root with secrets is how accidental writes happen.

### 3.11 `agent` (strategy choice)

- Start with `"react"`. It is the simplest loop and handles the majority of tool-using tasks.
- Choose `"plan-execute"` when the work is naturally a checklist and each step is one shot — it is cheaper than `plan-react` because it does not loop per step.
- Choose `"plan-react"` when steps are interdependent and each needs tools and several turns.
- Choose `"agent-swarm"` only when plan steps are genuinely independent (different files, different research questions). Every step runs in a fresh agent with fresh memory in its own thread, so steps cannot share context, and conflicting steps (two agents editing one file) will race.
- `plan-execute` and `plan-react` expose `plan(task)` separately — call it if you want to inspect or approve the plan before execution.

### 3.12 `max_steps`

- One step ≈ one model call plus its tool executions, so `max_steps` is effectively your cost/stall guard.
- Defaults: 50 for `react`, `plan-react`, `agent-swarm` (per step), 10 for `plan-execute` (per plan step).
- Long, deep tool chains (large refactors, multi-file research) may need more. Conversely, drop it to 5–10 for quick tasks to stop a confused agent from looping.
- If an agent keeps hitting the limit, raising `max_steps` is often the wrong fix — improve `instruction`, narrow the task, or check the tool descriptions first. Repeated identical tool calls mean the model is not learning from observations.

### 3.13 `max_workers` (agent-swarm)

- Set it to roughly the number of independent plan steps, bounded by your provider rate limits. Default is 5.
- Too low and independent steps queue up, losing the concurrency benefit. Too high and you may hit API rate limits or spawn many agents that all touch the same resources.
- The planner is explicitly told to plan at most `max_workers` tasks and to avoid conflicting tasks, so the value influences the plan itself, not just the thread pool.
- Concurrency increases total token throughput per minute — monitor cost when you raise it.

### 3.14 `config["session_path"]` and `WorkflowSession`

```python
import os
from egoai import EgoAgent, WorkflowSession

SESSION_DIR = "/abs/path/to/sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

@WorkflowSession.capture("my_workflow")
def workflow():
    agent = EgoAgent(agent="react", model="deepseek-v4-flash",
                     max_tokens=10000, instruction="...",
                     config={"session_path": SESSION_DIR})
    agent.run("Do the thing.")
```

- The directory must exist; nothing is auto-created. A missing or invalid path fails loudly at record time.
- All recorded objects in one workflow must use the **same** `session_path`; the first recorded object decides it.
- Record sessions in a per-project directory (`sessions/`) and date/archive them — each run writes one JSON containing every LLM call, tool execution and conversation.
- Sessions are the debugging tool of the framework: when an agent does something odd, read the session JSON instead of adding print statements.

### 3.15 `config["file"]`

- `file=True` for build/refactor agents that must create and edit files; `file="read-only"` for analysis, review, and audit agents that must never mutate the workspace.
- It requires a workspace: agents get one from `directory` (default: cwd); a raw `LLM(..., config={"file": True})` raises `ValueError` because it has no directory. Create the directory first.
- This is equivalent to passing `tool_registry=build_file_tools(directory)` — use the config key when you want it declared with the rest of the behavior, or the explicit registry when you need to merge tools.
- File tools are workspace-confined: `..` and escaping paths are rejected, and `bash` refuses absolute/home paths and runs with a timeout. Still, `file=True` includes `delete` and `bash` — grant it only to agents you trust with the directory.

### 3.16 `config["skills"]`

- A skill is one `.md` file with YAML frontmatter:

```markdown
---
name: csv-cleaning
description: Use when a CSV has inconsistent delimiters or encoding issues.
---
# CSV cleaning
Step-by-step instructions for the model...
```

- `name` and `description` are required; files without valid frontmatter are silently skipped (one bad file cannot break the agent).
- The scan is one level (non-recursive), so keep all skill files directly in the directory.
- Only the name + description are injected into the system prompt — cheap. The full text is loaded on demand via the auto-registered `use_skill` tool, so a skill directory costs almost nothing until the model uses one.
- Write descriptions as triggers ("Use when ..."), not summaries. The model picks skills by description.

### 3.17 `config["agents.md"]`

- The directory is scanned recursively for files named exactly `AGENTS.md`; their contents are concatenated with relative-path headings and injected into the system prompt.
- Best practice: keep the coding conventions, build commands and repository rules in an `AGENTS.md` at the workspace root; point `config["agents.md"]` at the workspace or repo root.
- Contents stay in the prompt across `run()`/`plan()` memory rebuilds, so the agent never "forgets" the house rules.
- Watch the size: every call pays for the full text. If the combined `AGENTS.md` files are huge, the agent's token usage (and thus auto-summarization) will rise.

### 3.18 `config["context_manager"]`

- Both switches are off by default; nothing happens to memory unless you enable them.

```python
config = {
    "context_manager": {
        "summarize": 30000,   # compress history when cumulative tokens hit 30k
        "max_iteration": 10,  # trim after 10 one_shot turns
    },
}
```

- `summarize` — set it for long-running chats where the thread must keep going indefinitely. The conversation is replaced by a single summary (2000-token budget) plus the system prompt, so exact details are lost but the gist survives. Typical values: 20k–60k. Setting it too low (e.g. below one normal turn) makes it fire constantly and thrash the context.
- `max_iteration` — set it for agent tool loops that accumulate many short messages. It keeps the newest 4 non-system messages, so it drops tool chatter while preserving recent context. Must be greater than 4 for anything to be trimmed; typical value 10.
- Both checks run at the top of every provider `one_shot()` call, and both reset their counters after firing. During internal structured calls (`extract`, `classify`, `summarize`) they are suppressed, so helpers never compress their own temporary conversations.
- If your workflow is short and single-shot, leave both off — the memory never grows enough to matter.

### 3.19 `config` free metadata

- Any unknown key must map to a string and is preserved exactly as written (`"owner": "team-x"`). Use it for ownership tags, experiment IDs, or routing metadata; it never reaches the model.
- A non-string value on an unknown key raises `TypeError` at construction — the config is validated eagerly, so typos surface immediately rather than mid-workflow.

### 3.20 Memory helpers and list behavior

- `print(agent)` renders the whole conversation as a readable transcript (system, user, assistant, every tool call with name/arguments and result). Use it before and after a run — it is the fastest way to see what the agent actually did.
- `agent[i:j]` returns message dicts. Slice **assignment** between agents aliases the same dicts; use `save_memory()` (deep copy) when the two conversations must stay independent:

```python
carry = agent1.save_memory([1, len(agent1.memory)])   # deep copy, source untouched
agent2.memory = [{"role": "system", "content": agent2._react_prompt}, *carry]
```

- Ranges for `trim_memory` / `save_memory` are 1-based and inclusive; system prompts are always preserved by trimming and excluded by `save_memory` by default.
- Direct list operations on `agent.memory` (e.g. `agent.memory[0:1] = [...]`) bypass validation on purpose; assignment through the object (`agent[i] = ...`, `agent[i:j] = [...]`) validates that every message is a dict with `role` and `content`.

---

## Quick reference

**Raw LLM call**

```python
from dotenv import load_dotenv
from egoai import LLM

load_dotenv()

llm = LLM(
    model="deepseek-v4-flash",          # provider + model
    system_prompt="You are a precise assistant.",
    user_prompt="List 3 uses of a hash map.",
    max_tokens=800,                     # output cap only
    temperature=0.3,                    # low = deterministic
    reasoning_effort=None,              # fastest; set "high" for hard reasoning
)
print(llm.one_shot()["content"])
```

**Agent with tools**

```python
import os
from egoai import EgoAgent
from egoai.builtin_tools.file import build_file_tools

workspace = "/abs/path/to/workspace"
os.makedirs(workspace, exist_ok=True)

agent = EgoAgent(
    agent="react",                      # "plan-execute" | "plan-react" | "agent-swarm"
    model="deepseek-v4-flash",
    max_tokens=10000,
    instruction="You are a careful coding assistant. Never delete files.",
    directory=workspace,
    tool_registry=build_file_tools(workspace),
    temperature=0.2,
    reasoning_effort="medium",
    config={
        "session_path": "/abs/path/to/sessions",   # existing dir
        "context_manager": {"summarize": 30000, "max_iteration": 10},
    },
)
result = agent.run("Create hello.py that prints 'hello' and run it.", max_steps=20)
print(result["content"])
```
