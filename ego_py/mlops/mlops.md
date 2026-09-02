# MLOps — `WorkflowSession`

Production session capture for the agent framework. `WorkflowSession` records
**everything** that happens inside one wrapped workflow — every LLM call,
every tool execution, every agent conversation — into a **single numbered
JSON file** in the directory given by the workflow's `Config.session_path`
(see *Configuration* in section 2.4).

```json
{
  "1": { "type": "agent_run", "agent": "...", "task": "...",
         "metadata": {...}, "result": {...}, "memory": [...],
         "llm_calls": [...], "tool_events": [...] },
  "2": {...},
  "3": {...}
}
```

Each numbered key is **one conversation** — one `agent.run()` call, or one
standalone LLM call. If a workflow runs three agents (react, plan-execute,
plan-react), the file contains exactly steps `"1"`, `"2"`, `"3"`, and each
step contains the complete internal trace (planning calls, extraction calls,
tool calls) of that agent. The goal is full traceability: you can replay each
exact step of a multi-LLM / multi-agent workflow, with its metadata and full
memory, in order.

Location: `ego_py/mlops/sessions.py` — exported as
`from ego_py import WorkflowSession`. Session locations are read from the
workflow's `Config.session_path` (`ego_py.config`).

---

## 1. Why this exists

Running several agents in one function mixes many model calls into one
output. When something goes wrong (or when you want to evaluate quality),
you need to answer:

- Which agent ran, in which order, with which config?
- What exactly did the model send and receive at each step?
- Which tools were called, with what arguments and observations?
- What was the conversation memory at the end of each step?

`WorkflowSession` answers all of that with zero plumbing inside the agents:
you decorate a workflow function, and the entire interaction is captured and
saved as one JSON when the function finishes.

---

## 2. Quick start

### 2.1 Decorator (recommended)

```python
from ego_py.config import Config
from ego_py import WorkflowSession

config = Config(session_path="/existing/sessions/dir")

@WorkflowSession.capture("agents_workflow", config=config)   # one session name per workflow
def workflow_agents():
    agent1 = EgoAgent(agent="react", model="deepseek-v4-flash", ...)
    agent1.run(task)              # -> step "1"
    agent2 = EgoAgent(agent="plan-execute", ...)
    agent2.run(task)              # -> step "2"
    agent3 = EgoAgent(agent="plan-react", ...)
    agent3.run(task)              # -> step "3"

workflow_agents()
# -> <session_path>/agents_workflow-<UTC timestamp>-<id>.json
```

If `config.session_path` is `None` or does not exist, the workflow runs
normally but no session is recorded at all.

The instance form works too:

```python
@WorkflowSession("agents_workflow", config=config)
def workflow_agents(): ...
```

In both forms the decorating instance is only a **template** (name +
config): every call of the decorated function creates a fresh
`WorkflowSession`, so calling the same workflow twice — or from two threads
at once — always produces separate session files, never shared state.

### 2.2 Context manager

```python
with WorkflowSession("my_workflow", config=config) as session:
    agent.run(task)
    llm.one_shot()               # raw LLM call also recorded

print(session.steps)             # {"1": {...}, "2": {...}}
print(session.path)              # path of the written JSON
```

### 2.3 Multiple sessions

One decorated function = one session file. To record two separate sessions,
write two workflow functions:

```python
@WorkflowSession.capture("workflow_a", config=config)
def workflow_a(): ...

@WorkflowSession.capture("workflow_b", config=config)
def workflow_b(): ...
```

This is the convention used in `system/test.py`: `workflow_agents`,
`workflow_plain_react` and `workflow_llm_features` are three independent
sessions; `main.py` imports and runs them.

### 2.4 Configuration — the workflow's `Config`

Session JSONs go to the directory given by the workflow's
`Config.session_path` (`ego_py/config`, exported as
`from ego_py.config import Config`). `Config` holds exactly two path
attributes, both `None` by default: `session_path` and `agents_path`
(reserved for the agents' workspace — keep it `None` for now).

```python
from ego_py.config import Config

config = Config()                                  # both paths None -> no saving
config.session_path = "/existing/sessions/dir"     # enable saving there
```

**Saving rule:** sessions are saved only when `session_path` is set AND the
directory exists. `None` or a non-existent path disables saving (and
recording) entirely — nothing is created on the fly. The decorator re-reads
`config.session_path` at **call time**, so setting it after decoration but
before running the workflow takes effect.

---

## 3. API reference

```python
class WorkflowSession:
    def __init__(self, name=None, config=None)
    @classmethod
    def capture(cls, name=None, config=None) -> WorkflowSession
    def __call__(self, func)                    # use as a decorator
    def __enter__(self) -> WorkflowSession      # context manager
    def __exit__(self, exc_type, exc, tb) -> bool
    def save(self) -> Optional[str]             # write (or re-write) the JSON
```

| Member | Description |
| --- | --- |
| `name` | Workflow name used in the file name. Sanitized to `[A-Za-z0-9_-]`; default `"workflow"`. |
| `config` | `ego_py.config.Config` instance. Its `session_path` decides where (and whether) session JSONs are saved: the directory must exist, otherwise nothing is saved at all. A `None` config, or `None` session_path, disables saving. Re-read on every `__enter__`, so mutations before a call take effect. |
| `session_id` | Random UUID identifying this run (also embedded in every step's metadata). |
| `steps` | Ordered dict of the numbered records (`{"1": ..., "2": ...}`). |
| `path` | `Path` of the written JSON (set by the first `save()`). |
| `error` | `{"type", "message"}` when the wrapped workflow raised. |
| `save()` | Writes the current steps. Safe to call mid-workflow; the automatic save on exit re-writes the same file. Returns `None` when nothing was recorded. |

The location comes from the workflow's `Config.session_path` (see section
2.4): `None` or a non-existent directory means no sessions are saved. The
decorator template re-reads `config.session_path` at **call time**, so a
value set before running the workflow takes effect.

Notes:

- **Importing** a decorated function records nothing. Capture starts only
  when the function is *called* (the decorator wraps the call in a session).
- **Exceptions**: if the wrapped workflow raises, a partial session (all
  steps completed so far) is still saved, the error is stored on
  `session.error` and the exception is re-raised.
- **Nested sessions** in the same thread raise `RuntimeError` — wrap two
  workflows separately instead of nesting them.
- **Threads**: recording is thread-safe, but note that `contextvars` are
  per-thread and are **not inherited by newly spawned threads** — each
  `threading.Thread` starts with an empty context. Two threads calling the
  *same decorated workflow* are safe: the decorator creates one fresh
  session per call, so each thread records into its own file (steps are
  numbered in completion order under a lock). If a workflow spawns worker
  threads that make LLM/tool calls and you want them recorded into the
  active session, propagate the context explicitly:

  ```python
  import contextvars
  import threading

  with WorkflowSession("wf") as session:
      ctx = contextvars.copy_context()          # carries the active session
      t = threading.Thread(target=ctx.run, args=(worker,))
      t.start(); t.join()
  ```

---

## 4. Session file reference

File: `<session_path>/<name>-<YYYYmmdd-HHMMSS>-<id8>.json`, written with
`json.dump(indent=2, ensure_ascii=False)`.

Top level: **only numbered keys** `"1"`, `"2"`, `"3"`, … in the order the
conversations completed.

### 4.1 Step record (agent conversation)

```json
{
  "step": 1,
  "type": "agent_run",
  "agent": "DeepseekLLMReActAgent",
  "task": "compute ((20 - 4) * 5) / 2 + 3, save result.txt",
  "metadata": {
    "session_id": "…", "session_name": "agents_workflow",
    "agent_class": "DeepseekLLMReActAgent",
    "agent_type": "react",              // react | plan-execute | plan-react | null
    "model": "deepseek-v4-flash",
    "max_tokens": 10000, "temperature": 0.5,
    "response_format": null, "tool_choice": null,
    "reasoning_effort": "low",
    "tools": ["add", "subtract", "multiply", "divide", "ls", "read_file", "write_file"],
    "instruction": "You are testing math and file operations…",
    "directory": "/…/test_agent",
    "max_steps": 10,
    "started_at": "2026-…T…Z", "finished_at": "2026-…T…Z", "duration_s": 41.2,
    "error": {"type": "ValueError", "message": "…"}   // present only on failure
  },
  "result": { /* sanitized agent.run() return: plan / results / memories / content */ },
  "memory": [ /* full final conversation: [{role, content, tool_calls…}, …] */ ],
  "step_memories": [ /* plan-execute: one memory snapshot per executor step */ ],
  "llm_calls": [ … ],      // see 4.2
  "tool_events": [ … ]     // see 4.3
}
```

### 4.2 LLM call record (one API call)

```json
{
  "call": 1,
  "timestamp": "2026-…T…Z",
  "provider": "DeepseekLLM",
  "model": "deepseek-v4-flash",
  "settings": {
    "max_tokens": 10000, "temperature": 0.5,
    "response_format": null, "tool_choice": null,
    "reasoning_effort": "low", "tools": ["add", "ls", …]
  },
  "input_message_count": 4,
  "output": {
    "content": "The result is 43.",
    "reasoning": null,
    "tool_calls": [{"id": "…", "name": "add", "arguments": {"a": 40, "b": 3}}]
  },
  "appended_messages": [ {"role": "assistant", "content": "…"} ],
  "duration_s": 0.93,
  "error": { "type": "…", "message": "…" }   // present only on failure
}
```

`appended_messages` is the delta the call added to memory (usually the
assistant turn, including `tool_calls` when the model requested tools), so
the full memory of the step can be reconstructed from the first snapshot +
successive deltas without storing the whole memory per call.

### 4.3 Tool event (one executed tool call)

```json
{
  "tool": "add",
  "arguments": {"a": 40, "b": 3},
  "observation": "43",
  "timestamp": "2026-…T…Z",
  "duration_s": 0.001
}
```

`duration_s` is the batch total for all tools executed in the same call.

### 4.4 Standalone LLM step

A raw `one_shot()` / `extract()` / … outside any `agent.run()` becomes its
own numbered step: `type: "llm_call"`, `task: null`, with the same
`metadata`, `memory` (memory *after* the call), `llm_calls` (one entry) and
`tool_events` fields. This is how `workflow_llm_features` records pure LLM
workflows.

---

## 5. How it works under the hood

The framework requires **no changes** to `BaseLLM`, the providers or the
agents. `WorkflowSession` captures everything through **runtime method
patching** that lives only for the duration of the wrapped workflow.

### 5.1 The three capture hooks

When a session starts (`__enter__`), it temporarily replaces three kinds of
methods on the *classes* (not instances), so every instance — including
classes built later by the `EgoAgent`/`LLM` factories — is covered:

| Hook | Patched on | Captures |
| --- | --- | --- |
| `one_shot()` | every concrete provider class (discovered as `BaseLLM` subclasses that define their own `one_shot`: `DeepseekLLM`, `GrokLLM`, future providers) | every raw LLM API call → `llm_calls` |
| `_execute_tool_calls()` | `BaseLLM` (shared by all agents/providers) | every tool execution + its observation → `tool_events` |
| `run()` | `ReActAgent`, `PlanExecuteAgent`, `PlanReactAgent` | conversation boundaries → numbered steps |

Discovery of providers is generic (`BaseLLM.__subclasses__()` walk), so a
new provider is automatically instrumented without touching the MLOps layer.

### 5.2 Routing via ContextVars

The patched wrappers are **generic**: they look up the active session from a
module-level `contextvars.ContextVar` (`_ACTIVE_SESSION`) and the active
step from another (`_ACTIVE_STEP`). When no session is active, the wrapper
calls the original method untouched — normal program behaviour is identical
outside workflows. ContextVars isolate state per thread/context, so
concurrent workflows never see each other's session; the decorator creates
one fresh `WorkflowSession` per call, which keeps concurrent calls of the
same decorated function independent. Newly spawned threads start with an
empty context, so propagate it explicitly (see the *Threads* note above)
when worker threads must be recorded.

### 5.3 Conversation steps and nested runs

- A `run()` call with **no active step** opens a new step record, pushes it
  into `_ACTIVE_STEP`, executes the original `run()`, then finalizes:
  metadata (started/finished/duration), sanitized result, deep-copied final
  `memory`, `_step_memories` (plan-execute), and all LLM/tool events that
  occurred while the step was active. The step is numbered
  (`"1"`, `"2"`, …) in completion order.
- A `run()` call **inside another run** (e.g. `PlanReactAgent.run` calling
  `ReActAgent.run` per plan step) does *not* open a new step — it executes
  and its LLM/tool events flow into the outer step. One agent conversation
  therefore always equals one numbered record, even though plan-react makes
  many internal model calls.
- A `one_shot()` with no active step becomes a standalone `llm_call` step;
  a following tool execution (e.g. `agent.onestep()` without `run()`) is
  attached to that standalone record.

### 5.4 Patch lifecycle: refcounted registry

Patches are tracked in a module-level registry
`{(class, attr): {"original": fn, "refcount": n}}` guarded by a lock:

1. First session patches the class method and stores the original
   (refcount 1).
2. Concurrent sessions only increment the refcount — never double-wrap.
3. On `__exit__` each session decrements; the **last one out** restores the
   original method and removes the registry entry.

Restoration happens in a `finally` block, so even a failing workflow leaves
the framework exactly as it found it.

### 5.5 Sanitization & persistence

Before writing, records are sanitized recursively: `PlanExecute` dataclasses
→ plain dicts, datetimes → ISO strings, unknown objects → `str()` fallback;
memory and arguments are **deep-copied** at capture time so later mutation
by the agent cannot corrupt the log. The file is written on `__exit__`
(partial logs included on error) and reuses the same path on manual
`save()` calls.

### 5.6 End-to-end flow

```
@WorkflowSession.capture("agents_workflow", config=config)
def workflow_agents():
    react.run(task)        ─┐
    plan_execute.run(task) ─┤
    plan_react.run(task)   ─┘

workflow_agents()
        │
        ▼
__enter__: patch one_shot / _execute_tool_calls / run
           set _ACTIVE_SESSION (ContextVar)
        │
        ▼  react.run(...)  -> step "1": {metadata, result, memory, llm_calls, tool_events}
        ▼  plan-execute.run(...) -> step "2": {..., step_memories, 4 llm_calls}
        ▼  plan-react.run(...)   -> step "3": {..., 4 llm_calls (nested runs merged)}
        │
        ▼
__exit__: restore all original methods (refcount -> 0)
          write <session_path>/agents_workflow-<ts>-<id>.json
```

---

## 6. Conventions & best practices

- Workflows live in `system/` as functions decorated with
  `@WorkflowSession.capture("<workflow_name>", config=config)` — each
  workflow owns its own `Config` whose `session_path` points at an existing
  sessions directory; `main.py` imports and runs them. One function = one
  session file.
- Give workflows descriptive names — the name is the file prefix.
- Keep agent construction inside the workflow function where possible; it
  makes each session self-contained (though instances created outside the
  workflow are also captured, since patching happens at session start).
- Use the numbered steps as your debugging entry point: `memory` replays the
  conversation, `llm_calls` shows every model round-trip, `tool_events`
  shows every observation.

## 7. Limitations

- **No nested sessions** in the same thread (deliberate: it would make step
  ownership ambiguous). Use two workflows.
- Calls made inside **newly spawned threads** are not recorded unless the
  context is propagated (see the *Threads* note in section 3) —
  `threading.Thread` starts with an empty context and does not inherit the
  active session.
- `run()` is the step boundary. Other public entry points (`plan()`,
  `onestep()`) are recorded as LLM-call/tool sequences rather than named
  conversation steps.
- Custom agents should subclass the existing agents so they inherit the
  instrumented `run()`; custom providers are discovered automatically.
- Logs include full memories — large conversations produce large JSON files
  by design (traceability over size).
