"""MLOps package: workflow session capture and logging.

WorkflowSession records every LLM call, tool execution and agent
conversation inside a wrapped workflow into one numbered JSON. The save
directory is read per object from the ``config`` dict on each LLM-based
object (see BaseLLM): the first recorded object's
``config["session_path"]`` must be a non-empty string pointing at an
existing directory (otherwise RuntimeError), and every later recorded
object in the same workflow must resolve to the same directory.
"""

from ego_py.mlops.sessions import WorkflowSession

__all__ = ["WorkflowSession"]
