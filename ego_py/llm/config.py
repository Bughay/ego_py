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
applied lazily through ``config.get(...)`` — so ``ConfigModel.from_dict(None)``
normalizes to the exact same ``{}`` the library has always used.

The dataclass field for the ``"agents.md"`` key is ``agents_md`` because
Python identifiers cannot contain a dot; :meth:`from_dict` and
:meth:`to_dict` translate between the two spellings.

    from ego_py.llm.models import ConfigModel

    config = ConfigModel.from_dict({"file": True, "agents.md": "/repo"})
    config.to_dict()   # -> {"file": True, "agents.md": "/repo"}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Dict keys this schema validates (everything else is str -> str metadata).
KNOWN_KEYS = ("session_path", "skills", "file", "agents.md", "context_manager")


@dataclass
class ConfigModel:
    """Dataclass schema for the per-object ``config`` dict.

    Fields mirror the known config keys one-to-one (``agents_md`` maps to
    the ``"agents.md"`` key). ``extra`` collects any unknown str -> str
    keys so they survive validation unchanged. Build instances with
    :meth:`from_dict`; :meth:`to_dict` returns the normalized plain dict
    that BaseLLM stores as ``self.config``.
    """

    session_path: Optional[str] = None
    skills: Optional[str] = None
    file: bool = False
    agents_md: Optional[str] = None
    context_manager: Optional[Dict[str, Optional[int]]] = None
    extra: Dict[str, str] = field(default_factory=dict)

    # Input key order (used by to_dict to reproduce the caller's dict).
    _order: List[str] = field(default_factory=list, repr=False, compare=False)

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]]) -> "ConfigModel":
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(
                "config must be a dict of str -> str settings (or None), "
                f"got {type(config).__name__}"
            )

        session_path: Optional[str] = None
        skills: Optional[str] = None
        file: bool = False
        agents_md: Optional[str] = None
        context_manager: Optional[Dict[str, Optional[int]]] = None
        extra: Dict[str, str] = {}
        order: List[str] = []

        for key, value in config.items():
            if not isinstance(key, str):
                raise TypeError(
                    "config must map str keys to str values, got key "
                    f"{key!r}"
                )
            order.append(key)

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
                context_manager = normalized
            elif key == "skills":
                cls._check_path(value, "skills")
                skills = value
            elif key == "agents.md":
                cls._check_path(value, "agents.md")
                agents_md = value
            elif key == "file":
                if not isinstance(value, bool):
                    raise TypeError(
                        f"config['file'] must be a bool (True/False), "
                        f"got {type(value).__name__}"
                    )
                file = value
            elif key == "session_path":
                if not isinstance(value, str):
                    raise TypeError(
                        "config must map str keys to str values "
                        f"(config['session_path'] must be a string), "
                        f"got key {key!r} -> value {value!r}"
                    )
                session_path = value
            else:
                if not isinstance(value, str):
                    raise TypeError(
                        "config must map str keys to str values (except "
                        "'context_manager', 'skills', 'file' and "
                        f"'agents.md'), got key {key!r} -> value {value!r}"
                    )
                extra[key] = value

        return cls(
            session_path=session_path,
            skills=skills,
            file=file,
            agents_md=agents_md,
            context_manager=context_manager,
            extra=extra,
            _order=order,
        )

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

    def to_dict(self) -> Dict[str, Any]:
        """Return the normalized plain config dict.

        Only keys present in the original input are emitted (original order
        preserved), so an empty config normalizes to exactly ``{}`` and
        unknown str -> str keys survive untouched.
        """
        known = {
            "session_path": self.session_path,
            "skills": self.skills,
            "file": self.file,
            "agents.md": self.agents_md,
            "context_manager": (
                dict(self.context_manager)
                if self.context_manager is not None
                else None
            ),
        }
        result: Dict[str, Any] = {}
        for key in self._order:
            result[key] = known[key] if key in known else self.extra[key]
        return result


__all__ = ["ConfigModel", "KNOWN_KEYS"]
