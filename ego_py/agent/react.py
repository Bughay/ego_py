"""
ReAct (Reason-Act-Observe) agent.

Inherits all LLM plumbing (memory, tool registry, one_shot) from
agent_logic.llm.base.BaseLLM and adds the classic ReAct loop on top of it:

    onestep() -> one Reason -> Act -> Observe cycle
    run()     -> starts a fresh conversation for a task, then repeats
                 onestep() until the model stops calling tools (or
                 max_steps is reached)

The agent system prompt is REACT_SYSTEM_PROMPT + the injected instruction.

Passing `directory` (an absolute path) only scopes the system prompt to that
workspace — it does NOT register any tools. File tools must be passed
explicitly via `tool_registry`, e.g. with build_file_tools(directory):

Usage: mix with a concrete provider, e.g.

    from agent_logic.agent.builtin_tools.file import build_file_tools

    class GrokReAct(ReActAgent, GrokLLM):
        pass

    agent = GrokReAct(model="grok-4.6", max_tokens=10000,
                      instruction="You are helping a data analyst.",
                      directory="/absolute/path/to/workspace",
                      tool_registry=build_file_tools("/absolute/path/to/workspace"))
    result = agent.run("Find the config file and tell me what it contains.")
"""
from typing import Any, Callable, Dict, List, Optional

from ego_py.builtin_tools.file import validate_directory
from ego_py.llm.base import BaseLLM


class ReActAgent(BaseLLM):
    """
    ReAct agent built directly on top of the LLM base class.

    One call to onestep() performs exactly one Reason -> Act -> Observe cycle:
    the model responds (reason), any requested tools are executed (act) and
    their results are appended to memory as observations (observe).

    Parameters:
        model, max_tokens     required; passed to the provider
        instruction           optional extra instructions appended to the system prompt
        directory             optional absolute workspace path (scopes the system
                              prompt to it; no tools are registered automatically)
        tool_registry         optional dict of tools; pass build_file_tools(directory)
                              yourself if the agent should have file access
        memory                optional custom starting conversation

    Parameters forwarded to the provider via **kwargs:
        reasoning_effort      thinking level ("low" | "medium" | "high" on DeepSeek,
                              plus "xhigh" on Grok)
        temperature           sampling temperature
        response_format       forced output format, e.g. {"type": "json_object"}
        tools                 raw tool schema list
        tool_choice           tool selection policy, e.g. "auto"
    """

    REACT_SYSTEM_PROMPT = (
        "You are a ReAct (Reason-Act-Observe) agent. Solve the user's task by "
        "alternating between reasoning and tool use:\n\n"
        "1. Thought: reason about the current situation and decide what to do next.\n"
        "2. Action: call one or more tools when you need information or to take an action.\n"
        "3. Observation: the tool results are given back to you, then you keep reasoning.\n\n"
        "Repeat this loop until the task is fully solved, then stop calling tools "
        "and give the final answer."
    )

#-------------------------- magic methods --------------------------------------

    def __init__(
        self,
        model: str,
        max_tokens: int,
        instruction: Optional[str] = None,
        directory: Optional[str] = None,
        tool_registry: Optional[Dict[str, Callable]] = None,
        memory: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ):
        # When another agent (e.g. PlanExecuteAgent) sits earlier in the MRO,
        # the instruction/directory have already been stored on the instance.
        if instruction is None:
            instruction = getattr(self, "_instruction", None)
        if instruction is not None and not isinstance(instruction, str):
            raise TypeError("instruction must be a str or None")
        if directory is None:
            directory = getattr(self, "_directory", None)
        if directory is not None:
            directory = validate_directory(directory)
        self._directory = directory

        injected = f"\n\nAdditional instructions:\n{instruction}" if instruction else ""
        self._react_prompt = self.REACT_SYSTEM_PROMPT + injected
        if directory:
            self._react_prompt += (
                f"\n\nWorkspace directory: {directory}\n"
                "You may operate on files only inside this directory."
            )

        if memory is None:
            memory = [{"role": "system", "content": self._react_prompt}]

        super().__init__(
            model=model,
            max_tokens=max_tokens,
            memory=memory,
            tool_registry=tool_registry,
            **kwargs,
        )

        # Skills and AGENTS.md context (discovered by BaseLLM from
        # config["skills"] / config["agents.md"]) are appended to the stored
        # prompt as well: run() rebuilds memory from self._react_prompt, so
        # this is what keeps them alive across runs. The initial memory was
        # already patched by BaseLLM.__init__.
        self._react_prompt += self._skills_prompt + self._agents_prompt
        if self.memory and self.memory[0].get("role") == "system":
            self.memory[0]["content"] = self._react_prompt

#-------------------------- public methods --------------------------------------

    def onestep(self) -> Dict[str, Any]:
        """
        Run one ReAct step: call the model, execute any requested tools and
        append the observations to memory. Returns the model response plus
        the observations.
        """
        result = self.one_shot()

        observations = None
        if result["tool_calls"]:
            observations = self._execute_tool_calls(result["tool_calls"])
            for tool_call, observation in zip(result["tool_calls"], observations):
                self.memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": observation,
                    }
                )

        content = result.get("content")

        return {
            "content": content,
            "reasoning": result.get("reasoning"),
            "tool_calls": result["tool_calls"],
            "observations": observations,
        }

    def run(self, task: str, max_steps: int = 50) -> Dict[str, Any]:
        """
        Start a fresh ReAct conversation for `task` and repeat onestep()
        until the model stops calling tools or max_steps is reached.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        self._tokens_used = 0
        self._iterations = 0

        self.memory = [
            {"role": "system", "content": self._react_prompt},
            {"role": "user", "content": task},
        ]

        final = None
        for _ in range(max_steps):
            final = self.onestep()
            if not final["tool_calls"]:
                break
        return final
