"""System package: the decorated workflow functions.

Import a workflow and call it — the WorkflowSession decorator records the
run into the directory given by each recorded object's config dict:

    from system import workflow_agents, workflow_plain_react

Every LLM(...) / EgoAgent(...) inside a workflow carries
config={"session_path": <existing sessions dir>}; the first recorded object
decides the directory and every later one must agree (see ego_py/mlops).
"""

from examples.test import (
    workflow_agents,
    workflow_classify,
    workflow_llm_features,
    workflow_one_shot,
    workflow_plain_react,
    workflow_summarize_add_subtract,
    workflow_summarize_multiply_divide,
)


__all__ = [
    "workflow_agents",
    "workflow_classify",
    "workflow_llm_features",
    "workflow_one_shot",
    "workflow_plain_react",
    "workflow_summarize_add_subtract",
    "workflow_summarize_multiply_divide",
    "workflow_skills",
    "concurrency",
    "new_config_workflow",
]
