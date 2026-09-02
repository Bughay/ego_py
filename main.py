# main.py
"""
Entry point — loads the environment, then imports and runs the system
workflows.

Every workflow lives in system/ and is decorated with
@WorkflowSession.capture("<name>"). Each LLM(...) / EgoAgent(...) object
inside a workflow carries its own plain config dict whose
config["session_path"] must be a non-empty string naming an EXISTING
directory (otherwise RuntimeError at record time); the first recorded
object in a workflow decides the directory and every later object must
resolve to the same one. Each run is then recorded as one numbered
session JSON there (LLM calls, tool executions and agent conversations
included):

    workflow_agents()      -> <session_path>/agents_workflow-*.json     (3 steps)
    workflow_one_shot()    -> <session_path>/one_shot_workflow-*.json   (1 call)
    workflow_classify()    -> <session_path>/classify_workflow-*.json   (1 call)
    workflow_plain_react() -> <session_path>/plain_react_workflow-*.json (1 step)
    workflow_summarize_add_subtract()    -> react agent where summarize fires
    workflow_summarize_multiply_divide() -> react agent where summarize fires
    workflow_skills()                    -> skill discovery + prompt injection demo

    python main.py
"""
from dotenv import load_dotenv

from examples import (
    workflow_agents,
    workflow_classify,
    workflow_one_shot,
    workflow_plain_react,
    workflow_skills,
    workflow_summarize_add_subtract,
    workflow_summarize_multiply_divide,
    concurrency,
    new_config_workflow
)


def main():
    load_dotenv()
    new_config_workflow()


if __name__ == "__main__":
    main()
