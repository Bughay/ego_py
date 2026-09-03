# ego_py

A tiny, workflow-first Python framework that abstracts away LLMs and agents so you can focus on **what you're building**, not on API plumbing.

## The idea

Most LLM libraries make you think about providers, payloads, and message formats. This one doesn't. Everything from a raw model call to a tool-using agent or a whole multi-agent pipeline is just an object you talk to:

```python
from ego_py import EgoAgent, LLM

llm = LLM(model="deepseek-v4-flash",
          system_prompt="You are a helpful assistant.",
          user_prompt="Explain quantum entanglement in three sentences.")
print(llm.one_shot()["content"])

agent = EgoAgent(agent="react", model="deepseek-v4-flash",
                 max_tokens=10000, instruction="Be precise.")
print(agent.run("What is 2 + 2?"))
```

That's it. `EgoAgent` picks the agent strategy (`react`, `plan-execute`, `plan-react`, `agent-swarm`) and the provider (`deepseek*`, `grok*`) for you: one factory, one constructor, no boilerplate.

## How this idea turns into problem solving

You describe the problem; the object turns it into a solution. `EgoAgent` picks a strategy (`react`, `plan-execute`, `plan-react`, `agent-swarm`) and a provider (`deepseek*`, `grok*`), runs the loop of LLM calls, tool executions, memory, and hands back the answer. A raw `LLM` does the same for single calls. The problem goes in, the solution comes out; that's the whole idea.

## Workflow-first

The library treats a **workflow**, a sequence of LLM calls and agent steps that accomplish one real task, as the primary unit. You write a plain Python function, decorate it, and the entire run (every call, every tool execution, every conversation) is recorded automatically as a numbered session JSON:

```python
from ego_py import EgoAgent, LLM, WorkflowSession

@WorkflowSession.capture("research_workflow")
def research():
    writer = LLM(model="deepseek-v4-flash",
                 user_prompt="Write a short poem about the sea.",
                 config={"session_path": "/existing/sessions/dir"})
    poem = writer.one_shot()["content"]

    critic = EgoAgent(agent="plan-execute", model="deepseek-v4-flash",
                      max_tokens=10000, instruction="You are a literary critic.",
                      config={"session_path": "/existing/sessions/dir"})
    return critic.run(f"Critique this poem: {poem}")

research()  # -> /existing/sessions/dir/research_workflow-<timestamp>.json
```

Chaining stages is just normal Python. Running a workflow once, twice, or from a thread always produces its own session file, perfect for debugging, auditing, and MLOps.

## How to use the library

There are exactly two entry points: `LLM` and `EgoAgent`. Both are factories: the `model` string picks the provider (`deepseek*` or `grok*`), the `agent` string picks the strategy. Everything else is passed through as parameters, so one constructor carries the whole API.

### `LLM`: the raw model call, all parameters

```python
from ego_py import LLM

llm = LLM(
    # provider
    model="deepseek-v4-flash",      # "deepseek*" -> DeepseekLLM, "grok*" -> GrokLLM
    reasoning_effort="high",        # thinking level: "low" | "medium" | "high"
                                    # (Grok also allows "xhigh")

    # conversation setup — either both prompts, or a ready-made memory list
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain quantum entanglement in three sentences.",
    # memory=[{"role": "system", "content": "..."},
    #         {"role": "user",   "content": "..."}],

    # generation
    max_tokens=10000,               # response size cap
    temperature=0.5,                # sampling temperature (Grok default 1.0)
    response_format={"type": "json_object"},   # force JSON output

    # tools
    tool_registry={"add": add, "subtract": subtract},  # dict: name -> function
    # tools=[...],                  # raw schema list (overrides the generated schema)
    # tool_choice="auto",           # tool selection policy

    # config — see "One config dict rules everything" below
    config={
        "session_path": None,   # where sessions are recorded
        "skills": None,         # directory of .md skill files
        "file": False,          # auto-load file tools
        "agents.md": None,      # inject AGENTS.md contents into the prompt
        "context_manager": {
            "summarize": None,      # auto-compress history at this token count
            "max_iteration": None,  # auto-trim old turns every N calls
        },
    },
)

result = llm.one_shot()            # -> {"reasoning": str|None, "content": str, "tool_calls": [...]}
```

`tool_registry` is a dict of functions with no config objects, no wrapper classes. The key is the name the model sees, the value is any plain Python function; the framework reads the signature and docstring and turns it into the tool's JSON schema for you. That's why the built-ins are one-liners: `build_file_tools("/workspace")` returns exactly such a dict (`ls`, `read_file`, `write_file`, and friends), `build_math_tools()` the same for the arithmetic helpers, and since they're plain dicts you can merge them: `{**build_math_tools(), **build_file_tools("/workspace")}`.

### The built-in structured workflows: `extract`, `classify`, `summarize`

Every `LLM` (and every agent) carries three ready-made workflows: you describe a schema and get back parsed JSON, no prompt engineering:

```python
llm = LLM(model="deepseek-v4-flash",
          system_prompt="You read customer emails.",
          user_prompt="I want a refund, this is outrageous!!!")

# classify — one allowed value per key, validated and auto-retried
llm.classify({
    "sentiment": {"description": "Overall tone of the email",
                  "choices": ["positive", "negative", "neutral"]},
    "urgency":   {"description": "How urgent, 1 = calm, 5 = furious",
                  "choices": "1-5"},                  # or {"min": 1, "max": 5}
})   # -> {"sentiment": "negative", "urgency": 5}

# extract — free-form JSON from a description schema
llm.extract({"name": "The customer's name as written",
             "amount": "The refund amount they ask for as a number"})
# -> {"name": "...", "amount": ...}

# summarize — compress any text down to a token budget
llm.summarize(long_document_text, max_tokens=500)     # -> short summary string
```

### `EgoAgent`: one factory, three strategies, all parameters

```python
from ego_py import EgoAgent
from ego_py.builtin_tools.file import build_file_tools

agent = EgoAgent(
    # strategy + provider
    agent="react",                  # "react" | "plan-execute" | "plan-react" | "agent-swarm"
    model="deepseek-v4-flash",      # "deepseek*" or "grok*"

    # agent parameters
    max_tokens=10000,               # required — response size cap
    instruction="Be precise.",      # extra instructions appended to the agent prompts
    directory="/abs/workspace",     # scopes the agent prompts to this workspace
    tool_registry=build_file_tools("/abs/workspace"),  # dict: name -> function
    # memory=[...],                 # react only — custom starting conversation

    # provider parameters, forwarded straight through
    reasoning_effort="high",
    temperature=0.5,
    response_format=None,
    # tools=[...], tool_choice="auto",
    config={
        "session_path": None,
        "skills": None,
        "file": False,
        "agents.md": None,
        "context_manager": {
            "summarize": None,
            "max_iteration": None,
        },
    },
)

result = agent.run("Create todo.txt with today's plan", max_steps=50)
# -> {"reasoning": ..., "content": ..., "tool_calls": [...]}
```

The three strategies (plus a parallel variant):

| `agent=` | What it does | Best for |
|---|---|---|
| `react` | One tool-using loop: think → act → observe, repeated until the task is done | A single self-contained task with tools |
| `plan-execute` | Plans first, then executes each plan step sequentially | Tasks that must be broken into ordered steps |
| `plan-react` | Plans first, then runs a full ReAct loop per plan step | Multi-step tasks where each step needs tools |
| `agent-swarm` | `plan-react`, but the plan steps run concurrently, one fresh agent per step | Independent steps where wall-clock time matters |

### Every possible workflow is these built-ins composed

There is no graph DSL and no pipeline class. A workflow is a plain Python function that creates objects, calls their built-in methods, and passes results around: `one_shot()`, `extract()`, `classify()`, `summarize()`, and `agent.run()` can be chained in any order:

```python
def support_workflow(ticket_text: str) -> str:
    # 1. pull structured facts out of the raw text
    reader = LLM(model="deepseek-v4-flash",
                 system_prompt="You read support tickets.",
                 user_prompt=ticket_text)
    facts = reader.extract({"customer": "the customer's name",
                            "request": "what they are asking for"})

    # 2. classify to decide routing
    triage = LLM(model="deepseek-v4-flash",
                 system_prompt="You triage tickets.",
                 user_prompt=str(facts))
    labels = triage.classify({
        "queue": {"description": "Team to route to",
                  "choices": ["billing", "tech", "shipping"]},
        "priority": {"description": "1 = low, 3 = high",
                     "choices": "1-3"},
    })

    # 3. let an agent actually solve it (tools, multi-step reasoning)
    solver = EgoAgent(agent="react", model="deepseek-v4-flash",
                      max_tokens=10000,
                      instruction="You resolve support tickets.",
                      config={"session_path": "/existing/sessions/dir"})
    answer = solver.run(f"Queue: {labels['queue']}, facts: {facts}")

    # 4. compress the result for the boss
    digest = LLM(model="grok-4.6-fast",
                 system_prompt="You write one-paragraph executive summaries.",
                 user_prompt=answer["content"])
    return digest.one_shot()["content"]
```

The same object can be reused across stages too: conversation history is just `agent.memory`, a list you can slice and hand to the next object (`agent2[1:3] = agent1[1:3]`, `len(agent)`, `agent[0]`). Wrap the function in `@WorkflowSession.capture(...)` and the whole run is recorded as one session JSON.

## One config dict rules everything

Every object carries a single plain `config` dict. No subclassing, no builders: behavior is declared where you create the object:

```python
config = {
    "session_path": "/existing/sessions/dir",  # where sessions are recorded
    "file": True,                              # auto-load file tools (ls, read, write, edit...)
    "skills": "/path/to/skills",               # directory of .md skill files
    "agents.md": "/repo",                      # inject every AGENTS.md found under here into the prompt
    "context_manager": {
        "summarize": 30000,      # auto-compress history when tokens hit 30k
        "max_iteration": 10,     # auto-trim old turns every 10 calls
    },
}

agent = EgoAgent(agent="react", model="deepseek-v4-flash",
                 max_tokens=10000, directory="/workspace",
                 config=config)
```

The config is validated once, up front, by a single schema, so a typo fails immediately instead of silently misbehaving. Every key is optional; absent keys stay absent and change nothing.

| Key | Type | What it does |
|---|---|---|
| `session_path` | `str` | Directory where `WorkflowSession` saves the session JSON for this object's calls. When recording, it must name an **existing** directory: the first recorded object in a workflow decides it, and every later object must resolve to the same one. |
| `file` | `bool` (default `False`) | When `True`, the built-in file tools (`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`) are auto-loaded into the object's `tool_registry` and tools schema. The workspace comes from the object's `directory=` parameter: a raw `LLM(...)` without a directory raises `ValueError`. |
| `skills` | `str` or `None` | Directory of `.md` skill files. The skills are discovered at init time, advertised in the system prompt, and loadable by the model through the auto-injected `use_skill` tool. |
| `agents.md` | `str` or `None` | Directory scanned **recursively**; the contents of every file named exactly `AGENTS.md` found under it are concatenated and injected into the system prompt (agents keep the block across `run()`/`plan()` memory rebuilds). |
| `context_manager` | `dict` or `None` | Automatic context management with two independent switches, each `None` by default (= disabled): `summarize` triggers when cumulative token usage reaches this value, the conversation is compressed into a single summary message (2000-token budget) and the counter resets. `max_iteration` triggers after this many `one_shot` turns, old non-system messages are trimmed, keeping the system prompt(s) and the 4 most recent messages. |

Any **other** key must be plain `str -> str` metadata and is preserved verbatim, so you can attach your own tags to any object without touching the schema.

## What you get for free

### Built-in LLM functionalities: deliberately minimal

Every object exposes exactly five verbs: `one_shot()`, `extract()`, `classify()`, `summarize()` on every LLM, and `run()` on every agent. That's the whole API. It is minimal on purpose: one verb to talk, three verbs to structure output, one verb to solve a task. Every bigger workflow such as research pipelines, triage systems, multi-stage agents is just these verbs composed in plain Python (see "How to use the library" above). Fewer verbs means less to learn, less to debug, and nothing to fight when you chain stages.

### Context management: memory is a list

Conversation history lives in `agent.memory`, and the object itself behaves like a Python list of messages, so inspecting and editing context uses syntax you already know:

```python
len(agent)      # -> 6   how many messages are in the conversation
agent[0]        # -> {"role": "system", "content": "You are..."}
agent[-1]       # -> {"role": "assistant", "content": "The answer is 4."}
```

Assignment is validated (every message must be a dict with `role` and `content`), and slices work between objects:

```python
agent[0] = {"role": "system", "content": "New instructions"}  # replace one message
agent[1:3] = [{"role": "user", "content": "A fresh turn"}]    # replace two messages with one
del agent[2]    # drop a single message; for i in agent: ... also works
```

On top of the list syntax, two built-in helpers manage memory with simple 1-based ranges: `trim_memory()` (setter) and `save_memory()` (getter):

```python
agent.trim_memory([1, 10])        # deletes messages 1-10 in place; system prompts always kept
carry = agent.save_memory([1, 4]) # -> deep copy of messages 1-4 (system prompts skipped),
                                  #    source memory untouched — carry them somewhere else
print(agent)                      # -> the whole memory rendered as a clean readable transcript
```

### Manage memory of workflows: conversations are just arrays of messages

Because memory is a plain list of message dicts, moving a conversation between objects is a slice assignment with no serialization, no special transfer API:

```python
carry = agent1.save_memory([1, len(agent1.memory)])  # grab the entire conversation
agent2.memory = [{"role": "system", "content": agent2._react_prompt}, *carry]

agent2[1:3] = agent1[1:3]        # slice transfer between agents, no copying needed
```

`save_memory()` returns a deep copy, so `agent1` keeps talking without affecting `agent2`; a plain slice shares the dicts instead (use it when aliasing is fine). This is what makes workflow management simple: an agent's entire state is one array, and handing state to the next stage is one list operation. Chain two agents or ten, the pattern never changes.

- **Automatic tooling**: pass plain Python functions as a `tool_registry`; schemas are generated and tool calls executed for you. Built-in math and file tools included.
- **Session recording**: `WorkflowSession` patches nothing into your agent code; it wraps your workflow and logs it.
- **Automatic context compression**: summarization and trimming, opt-in via config (`config["context_manager"]`, see above).

## How the library manages context

Every LLM and every agent keeps its conversation in `memory`, a plain list of message dicts. Managing that context means two things: **shrinking** it so the prompt never grows out of control, and **moving** it so one object can hand its conversation to the next. Both are controlled from the one place that rules everything: the `config` dict.

Nothing happens unless you turn it on. `config["context_manager"]` has two independent switches, both `None` (= off) by default:

```python
llm = LLM(model="deepseek-v4-flash",
          system_prompt="You are a support bot.",
          config={
              "context_manager": {
                  "summarize": 30000,   # compress history when tokens hit 30k
                  "max_iteration": 10,  # trim old turns every 10 calls
              },
          })
```

- **`summarize`** triggers when cumulative token usage reaches this number, the whole conversation so far is compressed into a single summary message (2000-token budget) and the counter resets. Thousands of tokens of back-and-forth become one short paragraph.
- **`max_iteration`** triggers after this many `one_shot()` turns, `trim_memory()` drops the old non-system messages, keeping the system prompt(s) and the 4 most recent messages, and the counter resets. The conversation keeps going with fresh room.

You set them once at object creation; the checks run automatically at the top of every call. Your workflow code never changes; the config does the work.

## Memory management technique through code

The config switches above manage context *automatically* for long conversations. But most workflows need something simpler and more explicit: take what one LLM learned and hand it to the next. Since memory is just a list, that's plain Python.

Imagine the entire workflow is a site creation: a client LLM interviews the customer, a plan-execute agent turns the conversation into a plan, and a react agent builds the site.

### Step 1: keep the good parts: summarize, extract, or just `[]`

The client LLM goes through a whole conversation. The next stage needs the gist, the facts, or the raw history. One verb for each, and for the memory itself, just brackets, no helper:

```python
client = LLM(model="deepseek-v4-flash",
             system_prompt="You interview the customer about their website.",
             user_prompt="I need a website for my bakery.")
client.one_shot()   # you ask, customer answers, memory grows

# a) summarize — the conversation compressed into prose
short = client.summarize(client.format_memory(), max_tokens=500)   # -> one short string

# b) extract — only the structured facts you care about
facts = client.extract({"name": "the bakery's name",
                        "pages": "the pages the site needs",
                        "style": "the look they want"})
# -> {"name": "...", "pages": "...", "style": ...}

# c) the memory itself — index the object like a list
memory = client[1:]   # the ENTIRE conversation after the system prompt
```

`summarize()` gives you prose, `extract()` gives you fields, `client[...]` gives you the messages themselves. Pick the shape the next stage needs.

### Step 2: `[]` into the plan-execute agent, answer into the react agent

Take the entire conversation and put it into the plan-execute agent as its `self.memory`, one assignment:

```python
planner = EgoAgent(agent="plan-execute", model="deepseek-v4-flash",
                   max_tokens=10000, directory="/abs/workspace",
                   instruction="Plan the website build from the conversation.")
planner.memory = [{"role": "system", "content": planner._planner_prompt}, *memory]
answer = planner.run("Plan the build from the conversation above.")
```

Then post the answer to a react agent with the instruction "create this website", memory through `[]` only, again:

```python
from ego_py.builtin_tools.file import build_file_tools

builder = EgoAgent(agent="react", model="deepseek-v4-flash",
                   max_tokens=10000, instruction="Create this website.",
                   directory="/abs/workspace",
                   tool_registry=build_file_tools("/abs/workspace"))
print(builder.run(f"Here is the plan:\n{planner[1:]}"))
```

The whole flow is `object[]` all the way: `client[1:]` carries the conversation in, `planner[1:]` carries the answer out. The memory is just an array, so the brackets are the whole API.

## Philosophy
- **simplicity** - we have only a few things you can do, but you can build anyting from it.
- **Workflows first.** Build the pipeline you actually want; the framework records it.
- **Simplicity over surface area.** Four provider methods, one factory, one config dict.
- **DRY.** All plumbing (memory, tools, sessions) lives in one base class; a new provider is ~4 methods.

## Quick reference

| You want... | You write |
|---|---|
| One model call | `LLM(model=..., user_prompt=...).one_shot()` |
| A tool-using agent | `EgoAgent(agent="react", tool_registry=...)` |
| Structured output | `llm.extract(schema={...})` or `llm.classify(schema={...})` |
| A recorded pipeline | `@WorkflowSession.capture("name")` + `config["session_path"]` |
| Automatic history trimming | `config["context_manager"] = {"summarize": 30000}` |

Environment keys: `DEEPSEEK_API_KEY`, `XAI_API_KEY` (Grok).
