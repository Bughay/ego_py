"""
Plan-Execute agent.

Inherits all LLM plumbing from agent_logic.llm.base.BaseLLM and implements the
plan-then-execute strategy:

    1. plan(task) -> the planner LLM writes a plan, then extract() pulls out
                     the structured PlanExecute{plan: str, execute_steps: list[str]}
    2. run(task)  -> plan(task), then executes each step of the plan one by
                     one with the executor system prompt

The planner/executor system prompts are extended with the injected instruction.

Passing `directory` (an absolute path) scopes the plan and the execution to
that workspace: the directory is injected into both prompts. File tools are
NOT registered automatically — pass them explicitly via `tool_registry`
(e.g. build_file_tools(directory)) so the executor can read, write, list and
search files during step execution.

Usage: mix with a concrete provider, e.g.

    class GrokPlanExecute(PlanExecuteAgent, GrokLLM):
        pass
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ego_py.builtin_tools.file import validate_directory
from ego_py.llm.base import BaseLLM


@dataclass
class PlanExecute:
    """Structured plan produced by the planning step."""
    plan: str
    execute_steps: List[str]


class PlanExecuteAgent(BaseLLM):
    """
    Plan-Execute agent built directly on top of the LLM base class.

    plan() asks the planner LLM for a plan and extracts it into a PlanExecute
    object; run() then executes every step of that plan one by one.

    Parameters:
        model, max_tokens     required; passed to the provider
        instruction           optional extra instructions appended to both prompts
        directory             optional absolute workspace path (scopes the prompts
                              to it; no tools are registered automatically)
        tool_registry         optional dict of tools; pass build_file_tools(directory)
                              yourself if the executor should have file access

    Parameters forwarded to the provider via **kwargs (in chained agents such
    as PlanReactAgent they also flow through ReActAgent.__init__):
        reasoning_effort      thinking level ("low" | "medium" | "high" on DeepSeek,
                              plus "xhigh" on Grok)
        temperature           sampling temperature
        response_format       forced output format, e.g. {"type": "json_object"}
        tools                 raw tool schema list
        tool_choice           tool selection policy, e.g. "auto"
    """

    PLANNER_SYSTEM_PROMPT = (
        "You are a planning agent. Given the user's task, produce a clear "
        "step-by-step plan:\n\n"
        "- Break the task down into concrete, ordered, self-contained steps.\n"
        "- Each step must be a single, actionable instruction that can be executed on its own.\n"
        "- Number the steps in the order they must be executed.\n"
        "- Output only the plan, nothing else."
    )

    EXECUTOR_SYSTEM_PROMPT = (
        "You are an execution agent. You receive the overall plan plus a single "
        "step from it. Execute that step to the best of your ability and output "
        "only the result of the step."
        "\n\nYou can call tools when you need to interact with the filesystem "
        "or take other actions. Use the tools to complete the step, then give "
        "your final result without calling tools."
    )

#-------------------------- magic methods --------------------------------------

    def __init__(
        self,
        model: str,
        max_tokens: int,
        instruction: Optional[str] = None,
        directory: Optional[str] = None,
        tool_registry: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ):
        # When another agent (e.g. PlanReactAgent) sits later in the MRO, the
        # instruction/directory may already be stored on the instance.
        if instruction is None:
            instruction = getattr(self, "_instruction", None)
        if instruction is not None and not isinstance(instruction, str):
            raise TypeError("instruction must be a str or None")
        if directory is None:
            directory = getattr(self, "_directory", None)
        if directory is not None:
            directory = validate_directory(directory)

        self._instruction = instruction
        self._directory = directory
        self._step_memories: List[List[Dict[str, str]]] = []

        injected = f"\n\nAdditional instructions:\n{instruction}" if instruction else ""
        self._planner_prompt = self.PLANNER_SYSTEM_PROMPT + injected
        self._executor_prompt = self.EXECUTOR_SYSTEM_PROMPT + injected
        if directory:
            note = (
                f"\n\nWorkspace directory: {directory}. "
                "All work must stay inside this directory."
            )
            self._planner_prompt += note
            self._executor_prompt += note

        memory = [{"role": "system", "content": self._planner_prompt}]
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            memory=memory,
            tool_registry=tool_registry,
            **kwargs,
        )

        # Skills and AGENTS.md context (discovered by BaseLLM from
        # config["skills"] / config["agents.md"]) are appended to BOTH
        # stored prompts as well: plan() and _execute_step() rebuild memory
        # from them, so this is what keeps them alive across runs. The
        # initial memory was already patched by BaseLLM.__init__.
        self._planner_prompt += self._skills_prompt + self._agents_prompt
        self._executor_prompt += self._skills_prompt + self._agents_prompt
        if self.memory and self.memory[0].get("role") == "system":
            self.memory[0]["content"] = self._planner_prompt

#-------------------------- private helpers -------------------------------------

    @staticmethod
    def _normalize_plan(extracted: Any) -> PlanExecute:
        """Validate/coerce the extract() output into a PlanExecute object."""
        if not isinstance(extracted, dict):
            raise ValueError(
                f"Plan extraction failed: expected a dict, got {type(extracted).__name__}"
            )
        plan = extracted.get("plan")
        steps = extracted.get("execute_steps")
        if plan is None or steps is None:
            raise ValueError("Plan extraction failed: missing 'plan' or 'execute_steps'")

        plan = str(plan).strip()
        if not isinstance(steps, list):
            raise ValueError("Plan extraction failed: 'execute_steps' must be a list")
        steps = [str(step).strip() for step in steps]
        steps = [step for step in steps if step]

        if not plan or not steps:
            raise ValueError("Plan extraction failed: 'plan' and 'execute_steps' must be non-empty")
        return PlanExecute(plan=plan, execute_steps=steps)

    def _execute_step(self, step: str, plan: PlanExecute, max_steps: int = 100) -> Dict[str, Any]:
        """Execute a single step of the plan with the executor prompt.

        The executor may call tools: each requested tool is executed and its
        observation appended to memory, repeating until the model stops
        calling tools or max_steps is reached.
        """
        self._tokens_used = 0
        self._iterations = 0

        self.memory = [
            {"role": "system", "content": self._executor_prompt},
            {
                "role": "user",
                "content": f"Overall plan:\n{plan.plan}\n\nExecute this step:\n{step}",
            },
        ]

        result = None
        for _ in range(max_steps):
            result = self.one_shot()
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
            else:
                break

        # Keep a snapshot of the full step conversation so callers can inspect
        # the complete execution history after run() finishes.
        self._step_memories.append([dict(msg) for msg in self.memory])
        return result

    def _validate_max_steps(self, max_steps: int) -> None:
        """Validate the max_steps argument shared by run()."""
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

#-------------------------- public methods --------------------------------------

    def plan(self, task: str) -> PlanExecute:
        """Have the planner LLM write a plan, then extract the structured version."""
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        self.memory = [
            {"role": "system", "content": self._planner_prompt},
            {"role": "user", "content": task},
        ]

        # The planner may call the registered tools (e.g. to inspect the
        # workspace). Execute them in a short loop so every assistant
        # tool_calls message is answered with tool messages before extract()
        # reuses the conversation — otherwise the provider rejects the next
        # request ("tool_calls must be followed by tool messages").
        for _ in range(8):
            result = self.one_shot()
            if not result["tool_calls"]:
                break
            observations = self._execute_tool_calls(result["tool_calls"])
            for tool_call, observation in zip(result["tool_calls"], observations):
                self.memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": observation,
                    }
                )

        extracted = self.extract(
            schema={
                "plan": "a short string summarizing the overall plan",
                "execute_steps": (
                    "a list of strings, where each string is one concrete step "
                    "of the plan, in execution order"
                ),
            },
            instruction="Convert the plan produced above into this structured format.",
        )
        plan = self._normalize_plan(extracted)
        return plan

    def run(self, task: str, max_steps: int = 10) -> Dict[str, Any]:
        """Plan the task, then execute each step of the plan one by one.

        max_steps caps how many tool-using turns the executor may take per
        plan step.
        """
        self._validate_max_steps(max_steps)
        self._tokens_used = 0
        self._iterations = 0
        plan = self.plan(task)
        self._step_memories = []
        results = []
        for step in plan.execute_steps:
            results.append(self._execute_step(step, plan, max_steps=max_steps))
        return {"plan": plan, "results": results}
