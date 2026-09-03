"""Shared config schema for every LLM-based object.

The plain ``config`` dict carried by BaseLLM (and inherited by every
provider, agent and combined factory class) is validated and normalized
through :class:`ConfigModel` — the single source of truth for the known
keys:

    session_path     str | None     session directory (read by WorkflowSession)
    skills           str | None     directory of .md skill files
    file             bool           True -> auto-load build_file_tools() for
                                    the workspace directory
    agents.md        str | None     directory scanned recursively; the
                                    contents of every AGENTS.md file found
                                    are injected into the system prompt
    context_manager  dict | None    {"summarize": int|None,
                                     "max_iteration": int|None}

Any other key must map to a str value (arbitrary user metadata) and is
preserved verbatim. Absent keys are not stored at all — defaults are
applied lazily through ``config.get(...)`` — so ``ConfigModel(None)``
normalizes to the exact same ``{}`` the library has always used.

:meth:`validate` performs the full normalization in a single pass over the
caller's dict and returns a fresh plain dict: keys are validated inline and
written into the result as they are encountered, so the caller's original
key order is reproduced for free and the input dict is never mutated. The
``"agents.md"`` key needs no special field spelling anymore because the
result is written straight through as ``result["agents.md"]``.

    from ego_py.llm.config import ConfigModel

    config = ConfigModel({"file": True, "agents.md": "/repo"})
    config.validate()   # -> {"file": True, "agents.md": "/repo"}

:meth:`from_dict` / :meth:`to_dict` are kept as backward-compatible
aliases for the old two-step chain:

    ConfigModel.from_dict(x).to_dict() == ConfigModel(x).validate()
"""
from __future__ import annotations

from typing import Any, Dict, Optional

#: Dict keys this schema validates (everything else is str -> str metadata).
KNOWN_KEYS = ("session_path", "skills", "file", "agents.md", "context_manager")


class ConfigModel:
    """Schema for the per-object ``config`` dict.

    Thin wrapper around the raw input: :meth:`validate` returns the
    normalized plain dict that BaseLLM stores as ``self.config``. Only
    keys present in the original input are emitted (original order
    preserved), so an empty config normalizes to exactly ``{}`` and
    unknown str -> str keys survive untouched.

    ``from_dict`` / ``to_dict`` are compatibility aliases for the old
    dataclass-style two-step chain.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]]) -> "ConfigModel":
        """Compatibility alias: build an instance around ``config``.

        Validation still runs eagerly here (as it did before the
        class-based rewrite), so ``from_dict`` raises on invalid input
        even when ``to_dict`` is never called; :meth:`validate` is
        idempotent, so the later ``to_dict`` call re-derives the same
        normalized dict.
        """
        instance = cls(config)
        instance.validate()
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Compatibility alias for :meth:`validate`."""
        return self.validate()

    def validate(self) -> Dict[str, Any]:
        """Validate ``self._config`` and return the normalized plain dict.

        None / {} -> {}. The input must be a dict with str keys; each
        known key is checked (file bool, session_path str, skills /
        agents.md string paths, context_manager dict with
        summarize/max_iteration positive-int-or-None) and unknown str ->
        str metadata is passed through. Keys are written into the result
        as they are visited, so the caller's key order is preserved; a
        fresh dict is returned each call and the input is never mutated.
        """
        config = self._config
        if config is None:
            return {}
        if not isinstance(config, dict):
            raise TypeError(
                "config must be a dict of str -> str settings (or None), "
                f"got {type(config).__name__}"
            )

        result: Dict[str, Any] = {}
        for key, value in config.items():
            if not isinstance(key, str):
                raise TypeError(
                    "config must map str keys to str values, got key "
                    f"{key!r}"
                )

            if key == "context_manager":
                if not isinstance(value, dict):
                    raise TypeError(
                        "config['context_manager'] must be a dict, "
                        f"got {type(value).__name__}"
                    )
                normalized: Dict[str, Optional[int]] = {}
                for cm_key, cm_value in value.items():
                    if cm_key not in ("summarize", "max_iteration"):
                        raise ValueError(
                            "config['context_manager'] keys must be "
                            "'summarize' and/or 'max_iteration', got "
                            f"{cm_key!r}"
                        )
                    if cm_value is not None and (
                        not isinstance(cm_value, int)
                        or isinstance(cm_value, bool)
                        or cm_value <= 0
                    ):
                        raise ValueError(
                            "config['context_manager'] values must be "
                            "positive ints or None, got "
                            f"{cm_key!r} -> {cm_value!r}"
                        )
                    normalized[cm_key] = cm_value
                normalized.setdefault("summarize", None)
                normalized.setdefault("max_iteration", None)
                result[key] = normalized
            elif key == "skills":
                self._check_path(value, "skills")
                result[key] = value
            elif key == "agents.md":
                self._check_path(value, "agents.md")
                result[key] = value
            elif key == "file":
                if not isinstance(value, bool):
                    raise TypeError(
                        f"config['file'] must be a bool (True/False), "
                        f"got {type(value).__name__}"
                    )
                result[key] = value
            elif key == "session_path":
                if not isinstance(value, str):
                    raise TypeError(
                        "config must map str keys to str values "
                        f"(config['session_path'] must be a string), "
                        f"got key {key!r} -> value {value!r}"
                    )
                result[key] = value
            else:
                if not isinstance(value, str):
                    raise TypeError(
                        "config must map str keys to str values (except "
                        "'context_manager', 'skills', 'file' and "
                        f"'agents.md'), got key {key!r} -> value {value!r}"
                    )
                result[key] = value

        return result

    @staticmethod
    def _check_path(value: Any, key: str) -> None:
        """Shared check for the string-path keys (skills / agents.md)."""
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"config[{key!r}] must be None or a string path, "
                f"got {type(value).__name__}"
            )
        if isinstance(value, str) and not value.strip():
            raise ValueError(
                f"config[{key!r}] must be a non-empty path when provided"
            )


__all__ = ["ConfigModel", "KNOWN_KEYS"]
