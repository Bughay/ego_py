"""Built-in tools that ship with the agents (file operations, math, etc.).

Nothing is registered automatically. Build the registries by hand and pass
them to the agent as `tool_registry`, e.g.

    from egoai.builtin_tools.file import build_file_tools
    from egoai.builtin_tools.math import build_math_tools

    tool_registry = {**build_math_tools(), **build_file_tools("/abs/workspace")}
"""
