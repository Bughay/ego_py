"""
Plan-React agent.

Combines the two other agents:

    1. PlanExecuteAgent.plan() -> structured PlanExecute{plan, execute_steps}
    2. for every step of the plan, a full ReAct loop (ReActAgent.run())
       executes it

The class only adds run(), the single function that drives the whole loop;
everything else is inherited from PlanExecuteAgent and ReActAgent.

The `directory` parameter flows through both parents: the planner/executor
prompts are scoped to the workspace. `directory` defaults to the current
working directory when omitted. File tools are not auto-registered —
pass them explicitly via `tool_registry` (e.g. build_file_tools(directory))
if the execution steps must operate on files.

Each ReAct execution step receives the overall plan as context in its user
message, so steps are executed with knowledge of the full plan rather than
in isolation. After every step the full conversation memory of that ReAct
loop is snapshotted and returned in the "memories" key of run(), so the
complete execution history can be inspected.

Usage: mix with a concrete provider, e.g.

    class GrokPlanReact(PlanReactAgent, GrokLLM):
        pass
"""
from typing import Any, Dict, List

from egoai.agent.planexecute import PlanExecuteAgent
from egoai.agent.react import ReActAgent


class PlanReactAgent(PlanExecuteAgent, ReActAgent):
    """Plan first, then execute every step of the plan with a ReAct loop.

    Inherits the full constructor parameter set from both parents: the
    agent-specific parameters (max_tokens, instruction, directory,
    tool_registry) and the provider parameters forwarded via **kwargs
    (reasoning_effort, temperature, response_format, tools, tool_choice).
    """

#-------------------------- public methods --------------------------------------

    def run(self, task: str, max_steps: int = 50) -> Dict[str, Any]:
        """Plan the task, then run a full ReAct loop for each step of the plan.

        Returns {"plan", "results", "memories"} where memories is a list of
        the full conversation memory (list of message dicts) of each step's
        ReAct loop, in execution order.
        """
        self._validate_max_steps(max_steps)
        self._tokens_used = 0
        self._iterations = 0
        plan = self.plan(task)
        results = []
        memories: List[List[Dict[str, str]]] = []
        for step in plan.execute_steps:
            results.append(
                ReActAgent.run(
                    self,
                    f"Overall plan:\n{plan.plan}\n\nExecute this step:\n{step}",
                    max_steps=max_steps,
                )
            )
            memories.append([dict(msg) for msg in self.memory])
        return {"plan": plan, "results": results, "memories": memories}
