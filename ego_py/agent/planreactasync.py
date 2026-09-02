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
    agent = EgoAgent(agent="plan-react-async", model="deepseek-v4-flash",
                     max_tokens=10000, ...)
    result = agent.run(task, max_steps=10)
"""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from ego_py.agent.planreact import PlanReactAgent
from ego_py.agent.react import ReActAgent


class AgentSwarm(PlanReactAgent):

    PLANNER_SYSTEM_PROMPT = (
        "You are a planning agent. Given the user's task, produce a clear "
        "step-by-step plan:\n\n"
        "- Break the task down into concrete, ordered, self-contained steps.\n"
        "- Each step must be a single, actionable instruction that can be executed on its own.\n"
        "- Number the steps in the order they must be executed.\n"
        "- Output only the plan, nothing else.\n"
        "# KEEP NOTE: \n"
        "- YOU ARE ASYNCRONOUS PLANNING AGENT MEANING THAT ALL YOUR TASKS WILL BE EXECUTED IN PARALLEL,\n"
        "- DONOT PLAN TASKS THAT WOULD CONFLICT TOGETHER, YOUR GOAL IS TO SAVE TIME THROUGH CONCURRENCY, MAKE SURE THE EXECUTOR AGENTS DONT CONFLICT EACH OTHER"

         
  
        

        
    )
    def _spawn_react_agent(self):
        """Fresh react instance of the same provider+agent class.

        New memory, new provider client, fresh token counters — one
        instance per thread, never shared.
        """
        return self.__class__(
            model=self.model,
            max_tokens=self.max_tokens,
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
        with ThreadPoolExecutor(max_workers=len(plan.execute_steps)) as pool:
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
