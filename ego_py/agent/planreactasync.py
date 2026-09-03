"""
Plan-React-Async agent.

Identical to PlanReactAgent — plan first, then one ReAct loop per plan
step — except the plan steps run concurrently: each step gets its own
fresh react agent and runs in its own thread.

Usage: mix with a concrete provider, e.g.

    class GrokPlanReactAsync(PlanReactAsyncAgent, GrokLLM):
        pass

or via the factory:

    from ego_py import EgoAgent
    agent = EgoAgent(agent="agent-swarm", model="deepseek-v4-flash",
                     max_tokens=10000, max_workers=5, ...)
    result = agent.run(task, max_steps=10)
"""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from ego_py.agent.planreact import PlanReactAgent
from ego_py.agent.react import ReActAgent


class AgentSwarm(PlanReactAgent):

    PLANNER_SYSTEM_PROMPT_ADDITION = (
        "\n\n# KEEP NOTE:\n"
        "- YOU ARE AN ASYNCHRONOUS PLANNING AGENT MEANING THAT ALL YOUR "
        "TASKS WILL BE EXECUTED IN PARALLEL.\n"
        "- Create a maximum of {max_workers} workers.\n"
        "- DO NOT PLAN TASKS THAT WOULD CONFLICT TOGETHER; YOUR GOAL IS TO "
        "SAVE TIME THROUGH CONCURRENCY, MAKE SURE THE EXECUTOR AGENTS "
        "DON'T CONFLICT WITH EACH OTHER."
    )

    def __init__(self, model, max_tokens, max_workers: int = 5, **kwargs):
        self.max_workers = max_workers
        super().__init__(model=model, max_tokens=max_tokens, **kwargs)
        self._planner_prompt += self.PLANNER_SYSTEM_PROMPT_ADDITION.format(
            max_workers=self.max_workers
        )
        if self.memory and self.memory[0].get("role") == "system":
            self.memory[0]["content"] = self._planner_prompt

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @max_workers.setter
    def max_workers(self, value: int):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("max_workers must be an integer")
        if value <= 0:
            raise ValueError("max_workers must be positive")
        self._max_workers = value

    def _spawn_react_agent(self):
        """Fresh react instance of the same provider+agent class.

        New memory, new provider client, fresh token counters — one
        instance per thread, never shared.
        """
        return self.__class__(
            model=self.model,
            max_tokens=self.max_tokens,
            max_workers=self.max_workers,
            instruction=getattr(self, "_instruction", None),
            directory=getattr(self, "_directory", None),
            tool_registry=self.tool_registry,
            tools=self.tools,
            tool_choice=self.tool_choice,
            temperature=self.temperature,
            response_format=self.response_format,
            reasoning_effort=getattr(self, "reasoning_effort", None),
            config=dict(self.config),
        )

    def run(self, task: str, max_steps: int = 50) -> Dict[str, Any]:
        self._validate_max_steps(max_steps)
        self._tokens_used = 0
        self._iterations = 0
        plan = self.plan(task)

        results = []
        memories: List[List[Dict[str, str]]] = []

        children = []
        for _ in plan.execute_steps:
            children.append(self._spawn_react_agent())

        # run every step's react loop in its own thread
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = []
            for child, step in zip(children, plan.execute_steps):
                step_ctx = contextvars.copy_context()
                futures.append(
                    pool.submit(
                        step_ctx.run,
                        ReActAgent.run,
                        child,
                        f"Overall plan:\n{plan.plan}\n\nExecute this step:\n{step}",
                        max_steps,
                    )
                )
            for future in futures:
                results.append(future.result())

        for child in children:
            memories.append([dict(msg) for msg in child.memory])

        return {"plan": plan, "results": results, "memories": memories}
