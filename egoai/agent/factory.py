"""Agent factory — one class that creates every kind of agent.

    from egoai import EgoAgent

    from egoai.builtin_tools.file import build_file_tools

    agent = EgoAgent(agent="react", model="deepseek-v4-flash",
                     max_tokens=10000, directory="/path/to/workspace",
                     tool_registry=build_file_tools("/path/to/workspace"),
                     reasoning_effort="low")
    result = agent.run(task, max_steps=10)

`EgoAgent` takes the same parameters as the agents, plus one extra: `agent`.

    agent="react"         -> ReActAgent
    agent="plan-execute"  -> PlanExecuteAgent
    agent="plan-react"    -> PlanReactAgent
    agent="agent-swarm"    -> AgentSwarm (plan steps run concurrently;
                              accepts max_workers)

`model` decides the provider:
    "deepseek*"  -> DeepseekLLM    (e.g. DeepseekLLMReActAgent)
    "grok*"      -> GrokLLM         (e.g. GrokLLMReActAgent)

Agent-specific parameters:
    max_tokens            required; response size cap
    instruction           optional extra instructions appended to the prompts
    directory             optional absolute workspace path (scopes the
                          agent prompts to that workspace); defaults to
                          the current working directory
    tool_registry         optional dict of tools; nothing is auto-added —
                          pass build_file_tools(directory) yourself to give
                          the agent file access
    memory                react only; custom starting conversation

Parameters forwarded straight through to the provider (DeepseekLLM / GrokLLM):
    reasoning_effort      thinking level
        DeepSeek: "low" | "medium" | "high"
        Grok:     "low" | "medium" | "high" | "xhigh"
    temperature           sampling temperature (DeepSeek default 0.5,
                          Grok default 1.0)
    response_format       forced output format, e.g. {"type": "json_object"}
    tools                 raw tool schema list (overrides the generated schema)
    tool_choice           tool selection policy, e.g. "auto"
    config                dict of str -> str settings attached to the
                          instance (default {}); WorkflowSession reads
                          config["session_path"] to decide where session
                          JSONs are saved (see egoai/mlops). May also
                          contain config["context_manager"] =
                          {"summarize": int|None, "max_iteration": int|None}
                          for automatic context summarization/trimming,
                          config["skills"] (directory of .md skill files),
                          config["file"] (bool or "read-only"; True
                          auto-loads the full built-in file tools for the
                          workspace directory, "read-only" loads only
                          ls/read_file/glob/grep) and
                          config["agents.md"] (directory scanned recursively;
                          every AGENTS.md file's contents are injected into
                          the system prompt). Validated and normalized by
                          ConfigModel (egoai/llm/config.py).
"""
from egoai.agent.planexecute import PlanExecuteAgent
from egoai.agent.planreact import PlanReactAgent
from egoai.agent.planreactasync import AgentSwarm
from egoai.agent.react import ReActAgent
from egoai.llm.deepseek import DeepseekLLM
from egoai.llm.grok import GrokLLM


class EgoAgent:
    """Same parameters as the agents, plus one extra: `agent`."""

    def __new__(cls, agent, model, *args, **kwargs):
        # --- step 1: check `agent` -> pick the agent class ----------------
        if agent == "react":
            agent_cls = ReActAgent
        elif agent == "plan-execute":
            agent_cls = PlanExecuteAgent
        elif agent == "plan-react":
            agent_cls = PlanReactAgent
        elif agent == "agent-swarm":
            agent_cls = AgentSwarm
        else:
            raise ValueError(f"Unknown agent {agent!r}; "
                             f"use 'react', 'plan-execute', 'plan-react' "
                             f"or 'agent-swarm'")

        # --- step 2: check `model` -> pick the provider class --------------
        if model.startswith("deepseek"):
            provider_cls = DeepseekLLM
        elif model.startswith("grok"):
            provider_cls = GrokLLM
        else:
            raise ValueError(f"No provider registered for model {model!r}")

        # --- step 3: combine the two and build the object -----------------
        Combined = type(f"{provider_cls.__name__}{agent_cls.__name__}",
                        (agent_cls, provider_cls), {})
        return Combined(model=model, *args, **kwargs)
