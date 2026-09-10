"""
WorkflowSession — production MLOps session capture for agent workflows.

A WorkflowSession wraps one complete workflow (a function that may run 1..N
agents and raw LLM calls) and records EVERY interaction that happens inside
it — every LLM call, every tool execution, every agent conversation — into a
single numbered JSON file:

    {
      "1": { "type": "agent_run",  "agent": "...", "task": "...",
             "metadata": {...}, "result": {...}, "memory": [...],
             "llm_calls": [...], "tool_events": [...] },
      "2": {...},
      "3": {...}
    }

Each numbered key is one conversation (one agent.run() call, or one
standalone LLM call).  The reasoning is traceability: for a workflow that
runs several agents you can replay each exact step, its metadata and its
full memory, in order.

Where sessions are saved — the per-object config contract
--------------------------------------------------------
WorkflowSession itself takes NO Config: instead, every LLM-based object
recorded inside the session (any LLM(...), EgoAgent(...), provider or agent
instance) carries its own plain `config` dict (str -> str, defined on
BaseLLM and inherited everywhere). The session directory is read from
`config["session_path"]` of the FIRST recorded object at the moment its
run()/one_shot()/tool call fires:

- the object must have a `config` attribute holding a dict;
- `config["session_path"]` must be a non-empty string;
- the path must exist and be a directory (nothing is auto-created);
- every later recorded object in the same workflow must resolve to the
  SAME directory, otherwise RuntimeError (one session = one directory).

If the first recorded object's config is missing/invalid, the error is
raised immediately with a message that says exactly what is wrong —
sessions never silently skip saving. A workflow that records no objects
at all produces no file and no error.

Usage (decorator — recommended):

    from egoai import EgoAgent, WorkflowSession

    SESSION_DIR = "/existing/sessions/dir"

    @WorkflowSession.capture("agents_workflow")
    def workflow_agents():
        agent1 = EgoAgent(agent="react", model="deepseek-v4-flash",
                          config={"session_path": SESSION_DIR}, ...)
        agent1.run(task)
        agent2 = EgoAgent(agent="plan-execute",
                          config={"session_path": SESSION_DIR}, ...)
        agent2.run(task)
        # ... more agents / raw LLM calls ...

    workflow_agents()   # -> saves <SESSION_DIR>/agents_workflow-<ts>.json

Usage (context manager):

    llm = LLM(model="deepseek-v4-flash", config={"session_path": SESSION_DIR}, ...)
    with WorkflowSession("my_workflow") as session:
        agent.run(task)
        llm.one_shot()
    print(session.steps, session.path)

How it works underneath (no changes to BaseLLM or the agents themselves):

1. On __enter__ the class temporarily patches, at runtime:
     - one_shot()           on every concrete provider class (discovered as
                            BaseLLM subclasses that define their own one_shot)
                            -> records every raw LLM call.
     - _execute_tool_calls() on BaseLLM
                            -> records every tool execution + observation.
     - run()                on ReActAgent / PlanExecuteAgent / PlanReactAgent
                            -> marks a conversation step boundary.
2. The wrappers route through a ContextVar holding the active session and
   the active step, so they are thread-safe and leave the rest of the
   program untouched when no session is active.  Note: contextvars are NOT
   inherited by newly spawned threading.Thread objects, so propagate the
   context explicitly (contextvars.copy_context().run) for worker threads
   that must be recorded.
3. Each decorated-function call gets a FRESH WorkflowSession (the
   decorating instance only supplies the name), so concurrent calls
   of the same workflow never share session state.
4. Nested run() calls (e.g. PlanReactAgent.run calling ReActAgent.run per
   plan step) reuse the outer step instead of creating extra steps, so one
   agent conversation == one numbered record.
5. On __exit__ the original methods are restored (via a refcounted patch
   registry, so concurrent sessions cannot double-wrap), and one JSON file
   is written — unless the workflow recorded no objects or the first
   object's config was invalid (the error was already raised at record
   time). If the wrapped workflow raised, a partial session is still saved
   and the exception is re-raised.
"""
from __future__ import annotations

import contextvars
import copy
import dataclasses
import datetime as dt
import functools
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("mlops.session")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[session] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# --------------------------------------------------------------------------
# Context state: which session/step is currently being recorded.
# ContextVars isolate state per thread/context: each thread sees only the
# session/step it set (or inherited from a copied context). Newly spawned
# threading.Thread objects start with an EMPTY context, so a workflow that
# wants its worker threads recorded must propagate the context explicitly
# (threading.Thread(target=contextvars.copy_context().run, args=(fn,))).
# --------------------------------------------------------------------------
_ACTIVE_SESSION: "contextvars.ContextVar[Optional[WorkflowSession]]" = (
    contextvars.ContextVar("mlops_active_session", default=None)
)
_ACTIVE_STEP: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = (
    contextvars.ContextVar("mlops_active_step", default=None)
)

# --------------------------------------------------------------------------
# Patch registry: maps (class, attr) -> {"original": fn, "refcount": n}.
# The refcount makes concurrent sessions safe: the first session patches,
# every session increments, the last session to exit restores the original.
# --------------------------------------------------------------------------
_PATCH_LOCK = threading.RLock()
_PATCH_REGISTRY: Dict[Tuple[type, str], Dict[str, Any]] = {}


def _now_iso() -> str:
    """UTC timestamp with millisecond precision."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _error_info(exc: BaseException) -> Dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _sanitize_name(name: str) -> str:
    """Make a workflow name safe for use in a filename."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(name)).strip("-")
    return cleaned or "workflow"


# --------------------------------------------------------------------------
# Generic wrapper factories. The wrappers are generic: they route to whatever
# session is active in the current context at call time (see _ACTIVE_SESSION),
# which lets a single patch serve multiple concurrent sessions.
# --------------------------------------------------------------------------

def _make_one_shot_wrapper(original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        session = _ACTIVE_SESSION.get()
        if session is None:
            return original(self, *args, **kwargs)
        return session._record_one_shot(original, self, args, kwargs)
    return wrapped


def _make_tool_execution_wrapper(original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        session = _ACTIVE_SESSION.get()
        if session is None:
            return original(self, *args, **kwargs)
        return session._record_tool_execution(original, self, args, kwargs)
    return wrapped


def _make_run_wrapper(original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        session = _ACTIVE_SESSION.get()
        if session is None:
            return original(self, *args, **kwargs)
        return session._record_run(original, self, args, kwargs)
    return wrapped


def _discover_provider_classes() -> List[type]:
    """Find every BaseLLM subclass that defines its own one_shot().

    Those are the concrete providers (DeepseekLLM, GrokLLM, ... and any
    future provider) — the classes whose one_shot() actually hits an API.
    Patching those base classes covers every instance, including combined
    classes built later by the EgoAgent factory.
    """
    from egoai.llm.base import BaseLLM

    found: List[type] = []
    seen = set()

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            if "one_shot" in sub.__dict__:
                found.append(sub)
            walk(sub)

    walk(BaseLLM)
    return found


# ==========================================================================
# WorkflowSession
# ==========================================================================

class WorkflowSession:
    """
    Records one complete workflow into a single numbered session JSON.

    Parameters:
        name      optional workflow name; used for the file name
                  (sanitized: only A-Za-z0-9_- survive). Default "workflow".

    WorkflowSession takes NO Config and resolves nothing up front. The
    session directory comes from the `config` dict on each recorded object
    (every LLM-based object has one; see BaseLLM): the first recorded
    object's config["session_path"] must be a non-empty string pointing at
    an EXISTING directory — otherwise RuntimeError with a message saying
    exactly what is wrong. Every later recorded object must resolve to the
    same directory.

    Use as a decorator:

        @WorkflowSession.capture("agents_workflow")
        def workflow():
            agent = EgoAgent(..., config={"session_path": "/sessions/dir"})
            agent.run(task)

    or as a context manager:

        with WorkflowSession("agents_workflow") as session:
            ...

    After the workflow finishes, session.steps holds the numbered records
    ({"1": ..., "2": ...}) and session.path points at the written JSON file
    (None when nothing was recorded or the directory never resolved).
    """

    def __init__(self, name: Optional[str] = None):
        self.name = _sanitize_name(name) if name else "workflow"
        # The raw name is kept so a decorating instance acts as a template:
        # every call of the decorated function gets a fresh session.
        self._template_name = name
        # Resolved lazily from the first recorded object's
        # config["session_path"] (see _ensure_enabled); stays None until
        # then — an empty workflow resolves nothing and writes no file.
        self.directory: Optional[Path] = None
        self.session_id = uuid.uuid4().hex
        self.steps: Dict[str, Dict[str, Any]] = {}
        self.path: Optional[Path] = None
        self.error: Optional[Dict[str, str]] = None
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None

        self._enabled = False
        self._patched: List[Tuple[type, str]] = []
        self._lock = threading.Lock()
        self._session_token = None
        self._last_standalone: Optional[Dict[str, Any]] = None
        self._agent_type_map: List[Tuple[type, str]] = []

# ------------------------- decorator API ----------------------------------

    def __call__(self, func: Callable) -> Callable:
        """Use the session instance directly as a decorator.

        The decorated instance acts as a TEMPLATE (name only): every call
        of the wrapped function gets a fresh WorkflowSession, so two
        threads calling the same workflow concurrently — or one thread
        calling it twice — can never share session state (steps, path,
        ContextVar token, patch refcounts).
        """
        if not callable(func):
            raise TypeError("WorkflowSession decorates a callable")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            session = WorkflowSession(name=self._template_name)
            wrapper._last_session = session  # type: ignore[attr-defined]
            with session:
                return func(*args, **kwargs)

        wrapper._workflow_session = self  # type: ignore[attr-defined]
        return wrapper

    @classmethod
    def capture(cls, name: Optional[str] = None) -> "WorkflowSession":
        """Decorator factory: @WorkflowSession.capture("name")

        Returns a TEMPLATE session (see __call__): every call of the
        decorated function creates a fresh session. The session directory
        is read per object at record time from each object's
        config["session_path"] — the first recorded object decides where
        the JSON is saved, and every later object must agree.
        """
        return cls(name=name)

# ------------------------- context manager ---------------------------------

    def _session_path_from(self, instance: Any) -> Path:
        """Resolve one recorded object's config["session_path"] to a directory.

        Raises RuntimeError with a message naming exactly what is wrong when:
          - the object has no config attribute, or it is not a dict;
          - config has no "session_path" key, or it is not a non-empty str;
          - the path does not exist / is not a directory (nothing is ever
            auto-created).
        """
        class_name = instance.__class__.__name__
        config = getattr(instance, "config", None)
        if config is None or not isinstance(config, dict):
            raise RuntimeError(
                f"{class_name} has no config attribute or its config is not a "
                f"dict; expected config={{'session_path': ...}} on every "
                "recorded object"
            )
        session_path = config.get("session_path")
        if not isinstance(session_path, str) or not session_path.strip():
            raise RuntimeError(
                f"{class_name}.config has no 'session_path' key or its value is "
                f"not a non-empty string; expected "
                f"config={{'session_path': '/path/to/sessions'}}, "
                f"got config={config!r}"
            )
        path = Path(session_path).expanduser().resolve()
        if not path.is_dir():
            raise RuntimeError(
                f"{class_name}.config['session_path'] = {session_path!r} does not "
                f"exist or is not a directory; create it before running the "
                "workflow (nothing is auto-created)"
            )
        return path

    def _ensure_enabled(self, instance: Any) -> None:
        """Resolve the session directory from the first recorded object and
        keep every later object honest about the path it resolves to.

        First object: validate its config, adopt the resolved directory,
        enable the session and stamp started_at. Later objects: re-validate
        their config and require the SAME resolved directory — one session
        is one directory, so conflicting session_paths raise RuntimeError.
        """
        path = self._session_path_from(instance)
        with self._lock:
            if not self._enabled:
                self.directory = path
                self._enabled = True
                self.started_at = _now_iso()
            elif path != self.directory:
                raise RuntimeError(
                    f"conflicting session_paths in one workflow: the first "
                    f"recorded object resolved to {str(self.directory)!r}, "
                    f"but {instance.__class__.__name__} resolves to "
                    f"{str(path)!r}; one WorkflowSession records into exactly "
                    "one directory"
                )

    def __enter__(self) -> "WorkflowSession":
        if _ACTIVE_SESSION.get() is not None:
            raise RuntimeError(
                "Nested WorkflowSession is not supported in the same context; "
                "close the outer session first."
            )

        # Always patch and go active: the session directory is resolved
        # lazily from the first recorded object's config (see
        # _ensure_enabled). _enabled stays False until that happens, so an
        # empty workflow records nothing and writes no file.
        # Lazy imports avoid circular imports with egoai.__init__.
        from egoai.agent.planexecute import PlanExecuteAgent
        from egoai.agent.planreact import PlanReactAgent
        from egoai.agent.react import ReActAgent
        from egoai.llm.base import BaseLLM

        # Order matters: plan-react is also a plan-execute/ReAct instance.
        self._agent_type_map = [
            (PlanReactAgent, "plan-react"),
            (PlanExecuteAgent, "plan-execute"),
            (ReActAgent, "react"),
        ]

        # 1. LLM calls — patch every concrete provider's one_shot().
        for cls in _discover_provider_classes():
            self._patch(cls, "one_shot", _make_one_shot_wrapper)
        # 2. Tool executions — patch the shared base helper.
        self._patch(BaseLLM, "_execute_tool_calls", _make_tool_execution_wrapper)
        # 3. Conversation boundaries — patch the three agent run() methods.
        for cls in (ReActAgent, PlanExecuteAgent, PlanReactAgent):
            self._patch(cls, "run", _make_run_wrapper)

        self._session_token = _ACTIVE_SESSION.set(self)
        logger.info("started session '%s' (id=%s)", self.name, self.session_id[:8])
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and exc is not None:
            self.error = _error_info(exc)
        try:
            self.save()
        finally:
            self._restore_patches()
            self._enabled = False
            self.finished_at = _now_iso()
            token = self._session_token
            if token is not None:
                _ACTIVE_SESSION.reset(token)
                self._session_token = None
        if self.error is not None:
            logger.error(
                "session '%s' failed (%s): %s",
                self.name, self.error["type"], self.error["message"],
            )
        return False  # re-raise workflow exceptions

# ------------------------- persistence --------------------------------------

    def save(self) -> Optional[str]:
        """Write the recorded steps to <session_path>/<name>-<ts>-<id8>.json.

        Safe to call manually mid-workflow; the automatic save on exit
        re-writes the same file. Returns the file path, or None when
        nothing was recorded or no session_path was resolved (an empty
        workflow).
        """
        with self._lock:
            if self.directory is None:
                logger.info(
                    "session '%s': no session_path resolved (no recorded "
                    "objects), no file written", self.name
                )
                return None
            if not self.steps:
                logger.info("session '%s': nothing recorded, no file written", self.name)
                return None
            if self.path is None:
                timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
                filename = f"{self.name}-{timestamp}-{self.session_id[:8]}.json"
                self.path = self.directory / filename
            payload = self._sanitize(self.steps)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
            path = str(self.path)
        logger.info(
            "session '%s': saved %d step(s) to %s",
            self.name, len(self.steps), path,
        )
        return path

# ------------------------- patch management --------------------------------

    def _patch(self, cls: type, attr: str, wrapper_factory: Callable) -> None:
        """Temporarily replace cls.attr with a recording wrapper."""
        key = (cls, attr)
        with _PATCH_LOCK:
            entry = _PATCH_REGISTRY.get(key)
            if entry is None:
                original = getattr(cls, attr)
                wrapper = wrapper_factory(original)
                setattr(cls, attr, wrapper)
                _PATCH_REGISTRY[key] = {"original": original, "refcount": 1}
            else:
                entry["refcount"] += 1
            self._patched.append(key)

    def _restore_patches(self) -> None:
        """Restore every method this session patched (last one out restores)."""
        with _PATCH_LOCK:
            for key in self._patched:
                entry = _PATCH_REGISTRY.get(key)
                if entry is None:
                    continue
                entry["refcount"] -= 1
                if entry["refcount"] <= 0:
                    cls, attr = key
                    setattr(cls, attr, entry["original"])
                    del _PATCH_REGISTRY[key]
            self._patched = []

# ------------------------- recorders (called by the patched methods) --------

    def _record_run(self, original: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        """Wrap one agent.run() call as a numbered conversation step.

        If another run() is already active (an internal call such as
        PlanReactAgent.run -> ReActAgent.run), just execute it and let its
        LLM/tool events flow into the outer step.
        """
        active = _ACTIVE_STEP.get()
        if active is not None:
            return original(instance, *args, **kwargs)

        # Resolve/verify the session directory from this object's config.
        self._ensure_enabled(instance)

        task = args[0] if args else kwargs.get("task")
        record: Dict[str, Any] = {
            "type": "agent_run",
            "agent": instance.__class__.__name__,
            "task": task,
            "metadata": self._instance_meta(instance),
            "llm_calls": [],
            "tool_events": [],
        }
        if len(args) > 1:
            record["metadata"]["max_steps"] = args[1]
        elif "max_steps" in kwargs:
            record["metadata"]["max_steps"] = kwargs["max_steps"]
        record["metadata"]["started_at"] = _now_iso()
        start = time.monotonic()

        token = _ACTIVE_STEP.set(record)
        result = None
        try:
            result = original(instance, *args, **kwargs)
        except BaseException as exc:
            record["metadata"]["error"] = _error_info(exc)
            raise
        finally:
            _ACTIVE_STEP.reset(token)
            record["metadata"]["finished_at"] = _now_iso()
            record["metadata"]["duration_s"] = round(time.monotonic() - start, 4)
            if result is not None:
                record["result"] = self._sanitize(result)
            memory = getattr(instance, "memory", None)
            if memory is not None:
                record["memory"] = copy.deepcopy(memory)
            # PlanExecuteAgent snapshots each executor conversation here.
            step_memories = getattr(instance, "_step_memories", None)
            if step_memories:
                record["step_memories"] = copy.deepcopy(step_memories)
            with self._lock:
                self._last_standalone = None
            self._add_step(record)
        return result

    def _record_one_shot(self, original: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        """Wrap one LLM API call and record input/output + appended messages."""
        # Resolve/verify the session directory from this object's config.
        self._ensure_enabled(instance)

        memory_before = copy.deepcopy(getattr(instance, "memory", []))
        call: Dict[str, Any] = {
            "timestamp": _now_iso(),
            "provider": instance.__class__.__name__,
            "model": getattr(instance, "model", None),
            "settings": self._llm_settings(instance),
            "input_message_count": len(memory_before),
        }
        start = time.monotonic()
        result = None
        try:
            result = original(instance, *args, **kwargs)
        except BaseException as exc:
            call["error"] = _error_info(exc)
            raise
        finally:
            call["duration_s"] = round(time.monotonic() - start, 4)
            if result is not None:
                call["output"] = self._sanitize(result)
            memory_after = getattr(instance, "memory", [])
            appended = (
                memory_after[len(memory_before):]
                if len(memory_after) >= len(memory_before) else memory_after
            )
            call["appended_messages"] = copy.deepcopy(appended)

            step = _ACTIVE_STEP.get()
            if step is not None:
                call["call"] = len(step["llm_calls"]) + 1
                step["llm_calls"].append(call)
            else:
                self._add_step(self._standalone_llm_record(instance, call, memory_after))
        return result

    def _record_tool_execution(self, original: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        """Wrap _execute_tool_calls() and record one event per tool call."""
        # Resolve/verify the session directory from this object's config.
        self._ensure_enabled(instance)

        tool_calls = args[0] if args else kwargs.get("tool_calls", [])
        start = time.monotonic()
        try:
            observations = original(instance, *args, **kwargs)
        except BaseException as exc:
            event = {
                "tool": None,
                "arguments": None,
                "error": _error_info(exc),
                "timestamp": _now_iso(),
                "duration_s": round(time.monotonic() - start, 4),
            }
            self._attach_tool_event(event)
            raise

        duration = round(time.monotonic() - start, 4)  # batch total
        for tool_call, observation in zip(tool_calls, observations or []):
            event = {
                "tool": tool_call.get("name"),
                "arguments": copy.deepcopy(tool_call.get("arguments")),
                "observation": observation,
                "timestamp": _now_iso(),
                "duration_s": duration,
            }
            self._attach_tool_event(event)
        return observations

    def _attach_tool_event(self, event: Dict[str, Any]) -> None:
        """Send a tool event into the active step or the last standalone record."""
        step = _ACTIVE_STEP.get()
        if step is not None:
            step["tool_events"].append(event)
        else:
            with self._lock:
                if self._last_standalone is not None:
                    self._last_standalone["tool_events"].append(event)

# ------------------------- record builders -----------------------------------

    def _standalone_llm_record(self, instance: Any, call: Dict[str, Any],
                               memory_after: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a numbered step for an LLM call made outside any agent.run()."""
        call["call"] = 1
        return {
            "type": "llm_call",
            "agent": instance.__class__.__name__,
            "task": None,
            "metadata": {
                **self._instance_meta(instance),
                "started_at": call["timestamp"],
                "finished_at": _now_iso(),
                "duration_s": call.get("duration_s"),
            },
            "memory": copy.deepcopy(memory_after),
            "llm_calls": [call],
            "tool_events": [],
        }

    def _instance_meta(self, instance: Any) -> Dict[str, Any]:
        """Snapshot the static configuration of an agent/LLM instance."""
        tool_registry = getattr(instance, "tool_registry", None) or {}
        return {
            "session_id": self.session_id,
            "session_name": self.name,
            "agent_class": instance.__class__.__name__,
            "agent_type": self._agent_type_of(instance),
            "model": getattr(instance, "model", None),
            "max_tokens": getattr(instance, "max_tokens", None),
            "temperature": getattr(instance, "temperature", None),
            "response_format": copy.deepcopy(getattr(instance, "response_format", None)),
            "tool_choice": copy.deepcopy(getattr(instance, "tool_choice", None)),
            "reasoning_effort": getattr(instance, "reasoning_effort", None),
            "tools": list(tool_registry.keys()),
            "instruction": getattr(instance, "_instruction", None),
            "directory": getattr(instance, "_directory", None),
        }

    def _llm_settings(self, instance: Any) -> Dict[str, Any]:
        """The per-call generation settings actually in effect."""
        tool_registry = getattr(instance, "tool_registry", None) or {}
        return {
            "max_tokens": getattr(instance, "max_tokens", None),
            "temperature": getattr(instance, "temperature", None),
            "response_format": copy.deepcopy(getattr(instance, "response_format", None)),
            "tool_choice": copy.deepcopy(getattr(instance, "tool_choice", None)),
            "reasoning_effort": getattr(instance, "reasoning_effort", None),
            "tools": list(tool_registry.keys()),
        }

    def _agent_type_of(self, instance: Any) -> Optional[str]:
        for cls, label in self._agent_type_map:
            if isinstance(instance, cls):
                return label
        return None

# ------------------------- helpers -------------------------------------------

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        """Make any value JSON-safe: dataclasses -> dicts, datetimes -> ISO."""
        if dataclasses.is_dataclass(value):
            return {k: cls._sanitize(v) for k, v in dataclasses.asdict(value).items()}
        if isinstance(value, (dt.datetime, dt.date, dt.time)):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(k): cls._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._sanitize(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        try:
            return str(value)
        except Exception:
            return repr(value)

    def _add_step(self, record: Dict[str, Any]) -> None:
        """Number the record ("1", "2", ...) in completion order and store it."""
        with self._lock:
            number = len(self.steps) + 1
            record["step"] = number
            self.steps[str(number)] = record
            if record.get("type") == "llm_call":
                self._last_standalone = record

    def __repr__(self) -> str:
        return (
            f"WorkflowSession(name={self.name!r}, steps={len(self.steps)}, "
            f"directory={str(self.directory)!r})"
        )
