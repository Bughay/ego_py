"""LLM factory — create a provider object just by model name.

    from agent_logic import LLM

    llm = LLM(model="deepseek-v4-flash", system_prompt=..., user_prompt=...)
    llm = LLM(model="grok-4.6", system_prompt=..., user_prompt=...)

`model` decides the provider:
    "deepseek*"  -> DeepseekLLM
    "grok*"      -> GrokLLM

All other parameters pass straight through to the provider class:

    system_prompt, user_prompt, memory   conversation setup (see BaseLLM)
    max_tokens                           response size cap
    temperature                          sampling temperature
                                         (DeepSeek default 0.5, Grok default 1.0)
    response_format                      forced output format, e.g. {"type": "json_object"}
    tool_registry, tools, tool_choice    tool calling config
    reasoning_effort                     thinking level
        DeepSeek: "low" | "medium" | "high"
        Grok:     "low" | "medium" | "high" | "xhigh"
    config                               dict of str -> str settings attached to the
                                         instance (default {}); WorkflowSession reads
                                         config["session_path"] to decide where session
                                         JSONs are saved (see ego_py/mlops). May also
                                         contain config["context_manager"] =
                                         {"summarize": int|None, "max_iteration": int|None}
                                         for automatic context summarization/trimming,
                                         config["skills"] (directory of .md skill files),
                                         config["file"] (bool; True auto-loads the built-in
                                         file tools for the workspace directory) and
                                         config["agents.md"] (directory scanned recursively;
                                         every AGENTS.md file's contents are injected into
                                         the system prompt). Validated and normalized by
                                         ConfigModel (ego_py/llm/config.py).
"""
from ego_py.llm.deepseek import DeepseekLLM
from ego_py.llm.grok import GrokLLM


class LLM:
    """Factory: checks `model` and returns the right provider instance."""

    def __new__(cls, model, *args, **kwargs):
        if not isinstance(model, str):
            raise ValueError(f"model must be a string, got {type(model).__name__}")

        # --- check `model` -> pick the provider class ----------------------
        if model.startswith("deepseek"):
            provider_cls = DeepseekLLM
        elif model.startswith("grok"):
            provider_cls = GrokLLM
        else:
            raise ValueError(f"No provider registered for model {model!r}")

        return provider_cls(model=model, *args, **kwargs)
