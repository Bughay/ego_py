import os
import copy
import json
import inspect
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Callable

from ego_py.builtin_tools.file import build_file_tools, validate_directory
from ego_py.llm.config import ConfigModel

class BaseLLM(ABC):
    """
    Abstract base class for all LLM wrappers.

    Provides common infrastructure:
    - memory management (list of messages)
    - tool registry & schema generation
    - tool execution
    - interactive chat loop
    - Pythonic dunder methods
    - property validation for shared parameters
    - shared structured workflows (extract, classify, summarize, trim_memory)

    Subclasses must implement:
    - _get_api_key()          # provider-specific environment variable
    - _create_client()        # instantiate the underlying API client
    - _build_payload()        # build the request dict for the provider
    - one_shot()              # send the current memory and return response

    Every LLM-based object (providers, agents, combined factory classes)
    also carries a `config` dict validated and normalized by ConfigModel
    (ego_py/llm/models.py) — the single source of truth for its schema. It
    defaults to {} and is stored as a private copy. Recognized keys:
    config["session_path"] (session JSON directory, read by WorkflowSession),
    config["skills"] (directory of .md skill files whose frontmatter is
    discovered at init time, stored on skills_metadata, advertised in the
    system prompt and loadable via the `use_skill` tool), config["file"]
    (bool; True auto-loads build_file_tools() for the workspace directory
    into tool_registry + the tools schema), config["agents.md"] (directory
    scanned recursively; the contents of every AGENTS.md file found are
    injected into the system prompt) and config["context_manager"]
    ({"summarize": int|None, "max_iteration": int|None} for automatic
    context summarization/trimming). Unknown keys must remain str -> str.

    To create an object by model name, use the LLM factory in llm.py:
        from agent_logic import LLM
        llm = LLM(model="deepseek-v4-flash", ...)
    """

#-------------------------- magic methods --------------------------------------

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        memory: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 10000,
        temperature: float = 0.5,
        response_format: Optional[Any] = None,
        tool_registry: Optional[Dict[str, Callable]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        # Skill container. Stays None unless the user enables skills via
        # config["skills"] (a directory of .md skill files), in which case
        # discovery below fills it with a list of {"name", "description",
        # "file_path"} dicts.
        self.skills_metadata = None

        #------------------------------- VALIDATION LOGIC --------------------------------
        if memory is None and (system_prompt is not None and user_prompt is None):
            raise ValueError("You have entered the system prompt but not the user_prompt.")
        if memory is None and (system_prompt is None or user_prompt is None):
            raise ValueError("Either provide 'memory' or both 'system_prompt' and 'user_prompt'.")
        if memory is not None and (system_prompt is not None or user_prompt is not None):
            raise ValueError("Provide only one of: 'memory' OR both 'system_prompt' and 'user_prompt', not both.")

        self._memory = memory if memory is not None else [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        self.tool_registry = tool_registry
        if tool_registry is not None and tools is None:
            tools = self._generate_tools_schema()

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.response_format = response_format
        self.tools = tools
        self.tool_choice = tool_choice

        # Validate + normalize the raw config through the shared schema
        # (  in ego_py/llm/models.py). The result is a private
        # plain-dict copy, so later mutations of the caller's dict cannot
        # leak in; absent keys stay absent and defaults are applied lazily
        # via config.get() below.
        self.config = ConfigModel.from_dict(config).to_dict()

        # Skill discovery: skills_metadata stays None unless skills are
        # enabled. When config["skills"] names a directory, scan it for .md
        # files and render the skill listing into self._skills_prompt. Agent
        # classes append that block to their own prompt attributes right
        # after super().__init__() so run()/plan() memory rebuilds keep it.
        self._skills_prompt = ""
        skills_path = self.config.get("skills")
        if skills_path is not None:
            self.skills_metadata = self._discover_skills(skills_path)
            self._skills_prompt = self._format_skills_prompt(self.skills_metadata)
            # Only when skills are enabled: load the `use_skill` tool into
            # tool_registry + the tools schema so every one_shot() call
            # carries it.
            self._create_skills_tool()

        # AGENTS.md discovery: when config["agents.md"] names a directory,
        # recursively collect every AGENTS.md file under it into
        # self._agents_prompt. Agents append it to their prompt attributes
        # right after super().__init__(), exactly like the skills listing.
        self._agents_prompt = ""
        agents_path = self.config.get("agents.md")
        if agents_path is not None:
            self._agents_prompt = self._discover_agents_md(agents_path)

        # File tools: when config["file"] is True, load build_file_tools()
        # for the workspace directory into tool_registry + the tools schema
        # (same way the use_skill tool is auto-loaded), so every one_shot()
        # call carries file access without a hand-built registry.
        if self.config.get("file"):
            self._load_file_tools()

        # Inject both blocks into the initial memory's system message (raw
        # LLM objects get them here; agents additionally patch their prompt
        # attributes in their own __init__).
        injected = self._skills_prompt + self._agents_prompt
        if injected:
            system_msg = next(
                (msg for msg in self._memory if msg.get("role") == "system"),
                None,
            )
            if system_msg is None:
                self._memory.insert(0, {"role": "system", "content": injected})
            else:
                system_msg["content"] = (
                    system_msg.get("content") or ""
                ) + injected

        # Cumulative tokens used across every one_shot() call made by this
        # object. Maintained internally only; exposed read-only via tokens_used.
        self._tokens_used = 0
        self._iterations = 0
        self._managing_context = False

        self._client = None

    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]

    def __setitem__(self, index, value):
        """Validated assignment, scalar and slice.

        Scalar (agent[i] = {...}): the message must be a dict with 'role'
        and 'content' keys.

        Slice (agent[i:j] = [...]): the value must be a list and every item
        must satisfy the same rule as the memory property setter. Real
        memories are always well-formed, so inter-agent transfers such as
        agent2[1:3] = agent1[1:3] pass through instantly.

        Direct list operations on agent.memory (e.g. agent.memory[0:1] = ...)
        are the intentional raw tier and bypass these checks.
        """
        if isinstance(index, slice):
            if not isinstance(value, list):
                raise TypeError(
                    "Slice assignment requires a list of message dicts, "
                    f"got {type(value).__name__}"
                )
            for msg in value:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    raise ValueError(
                        "Each memory item must be a dict with 'role' and 'content' keys"
                    )
            self.memory[index] = value
            return
        if not isinstance(value, dict) or 'role' not in value or 'content' not in value:
            raise ValueError("Message must be a dict with 'role' and 'content' keys")
        self.memory[index] = value

    def __delitem__(self, index):
        del self.memory[index]

    def __iter__(self):
        return iter(self.memory)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model={self.model!r}, "
            f"memory_len={len(self.memory)}, temp={self.temperature})"
        )

    def __str__(self) -> str:
        """Full readable transcript — this is what print(agent) shows.

        Delegates to format_memory(). Use repr(agent) for the compact
        one-line summary (class, model, memory_len, temp).
        """
        return self.format_memory()

#-------------------------- properties -----------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model must be a non-empty string")
        self._model = value

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int):
        if not isinstance(value, int):
            raise TypeError("max_tokens must be an integer")
        if value <= 0:
            raise ValueError("max_tokens must be positive")
        if value > 1_000_000:
            raise ValueError("max tokens should not be over 1,000,000")
        self._max_tokens = value

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        if value < 0 or value > 2:
            raise ValueError("temperature must be between 0 and 2")
        self._temperature = value

    @property
    def memory(self) -> List[Dict[str, str]]:
        return self._memory

    @memory.setter
    def memory(self, value: List[Dict[str, str]]):
        if not isinstance(value, list):
            raise TypeError("memory must be a list")
        for msg in value:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Each memory item must be a dict with 'role' and 'content' keys")
        self._memory = value

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @tokens_used.setter
    def tokens_used(self, value):
        raise AttributeError(
            "tokens_used is read-only; do not set it directly. It is "
            "maintained internally by one_shot() and manual changes can "
            "corrupt token accounting."
        )

    def _manage_context(self) -> None:
        """Apply config['context_manager'] thresholds before an LLM call.

        Called at the top of every provider one_shot(). Re-entrant safe:
        structured workflows (summarize/extract/classify) set
        _managing_context while issuing their internal one_shot() calls, so
        those temporary conversations are never compressed or trimmed here.
        """
        if self._managing_context:
            return

        cm = self.config.get("context_manager") if isinstance(self.config, dict) else None
        if not cm:
            return

        summarize_threshold = cm.get("summarize")
        max_iteration = cm.get("max_iteration")
        if summarize_threshold is None and max_iteration is None:
            return

        self._managing_context = True
        try:
            self._iterations += 1

            if max_iteration is not None and self._iterations >= max_iteration:
                self._auto_trim_memory()
                self._iterations = 0

            if summarize_threshold is not None and self._tokens_used >= summarize_threshold:
                self._auto_summarize()
                self._tokens_used = 0
        finally:
            self._managing_context = False

    def _auto_summarize(self) -> None:
        """Compress the current conversation into a short summary message."""
        context = self.format_memory()
        summary = self.summarize(context, 2000)
        if not summary:
            return
        system_msgs = [m for m in self.memory if m.get("role") == "system"]
        self.memory = system_msgs + [{"role": "user", "content": summary}]

    def _auto_trim_memory(self) -> None:
        """Drop old non-system messages, keeping the system prompt(s) and the
        most recent KEEP_NON_SYSTEM (4) non-system messages."""
        keep = 4
        system_msgs = [m for m in self.memory if m.get("role") == "system"]
        non_system = [m for m in self.memory if m.get("role") != "system"]
        if len(non_system) <= keep:
            return
        self.memory = system_msgs + non_system[-keep:]

#-------------------------- skill discovery ------------------------------------

    @staticmethod
    def _parse_skill_frontmatter(text: str, file_path: str) -> Dict[str, str]:
        """Parse the YAML frontmatter of a skill .md file.

        Expects a leading '---' line, simple 'key: value' lines, and a
        closing '---' line. Values may be wrapped in single/double quotes;
        the split happens on the first colon so descriptions can contain
        colons. Returns a plain {"name", "description", "file_path"} dict.

        Raises ValueError when the frontmatter is missing or when name /
        description are absent or empty.
        """
        lines = text.splitlines()
        markers = [i for i, line in enumerate(lines) if line.strip() == "---"]
        if len(markers) < 2 or markers[0] != 0:
            raise ValueError(
                f"skill file has no YAML frontmatter (--- blocks): {file_path}"
            )

        meta: Dict[str, str] = {}
        for raw in lines[markers[0] + 1 : markers[1]]:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'").strip()
            if key:
                meta[key] = value

        name = meta.get("name")
        description = meta.get("description")
        if not name or not isinstance(name, str):
            raise ValueError(
                f"skill file is missing a 'name' in its frontmatter: {file_path}"
            )
        if not description or not isinstance(description, str):
            raise ValueError(
                f"skill file is missing a 'description' in its frontmatter: {file_path}"
            )
        return {
            "name": name.strip(),
            "description": description.strip(),
            "file_path": file_path,
        }

    @classmethod
    def _discover_skills(cls, skills_path: str) -> List[Dict[str, str]]:
        """Scan skills_path (one level, non-recursive) for *.md files and
        parse their frontmatter.

        Files without valid frontmatter are skipped so one bad file cannot
        break the whole agent. Returns a list of plain
        {"name", "description", "file_path"} dicts (possibly empty).
        """
        root = validate_directory(skills_path)
        metadata: List[Dict[str, str]] = []
        try:
            entries = sorted(os.listdir(root))
        except OSError as e:
            raise ValueError(f"cannot scan skills directory {root}: {e}")

        for entry in entries:
            if not entry.endswith(".md"):
                continue
            file_path = os.path.join(root, entry)
            if not os.path.isfile(file_path):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except OSError as e:
                raise ValueError(f"cannot read skill file {file_path}: {e}")
            try:
                metadata.append(cls._parse_skill_frontmatter(text, file_path))
            except ValueError:
                continue  # skip files without valid frontmatter
        return metadata

    @staticmethod
    def _format_skills_prompt(metadata: List[Dict[str, str]]) -> str:
        """Render the skill listing injected into the system prompt.

        Returns an empty string when there are no skills, so prompts stay
        byte-identical to a no-skills agent. The block only ADVERTISES the
        skills (name + description); the full content is loaded on demand
        through the `use_skill` tool.
        """
        if not metadata:
            return ""
        lines = [
            "## Available Skills",
            "You have access to the following skills. To use a skill, call "
            "the `use_skill` tool with the skill name.",
            "",
        ]
        for skill in metadata:
            lines.append(f"- **{skill['name']}**: {skill['description']}")
        return "\n\n" + "\n".join(lines)

    @staticmethod
    def _strip_skill_frontmatter(text: str) -> str:
        """Return the skill content after its YAML frontmatter block."""
        lines = text.splitlines()
        markers = [i for i, line in enumerate(lines) if line.strip() == "---"]
        if len(markers) >= 2 and markers[0] == 0:
            return "\n".join(lines[markers[1] + 1 :]).strip()
        return text.strip()

    def _create_skills_tool(self) -> Optional[Callable]:
        """Build the `use_skill` tool from the discovered skills and load it
        into this instance's tool set (tool_registry + tools schema).

        Called automatically by __init__ when config["skills"] is not None,
        so every one_shot() call always has the use_skill tool available.
        Returns the tool callable, or None when there are no discovered
        skills (nothing to call).
        """
        if not self.skills_metadata:
            return None

        metadata_by_name = {s["name"]: s for s in self.skills_metadata}

        def use_skill(name: str) -> str:
            """Load a skill by name and return its full instructions."""
            if not isinstance(name, str) or not name.strip():
                return "Error: use_skill requires the skill name as a string."
            skill = metadata_by_name.get(name.strip())
            if skill is None:
                available = ", ".join(sorted(metadata_by_name)) or "(none)"
                return (
                    f"Error: unknown skill {name!r}. "
                    f"Available skills: {available}"
                )
            try:
                with open(skill["file_path"], "r", encoding="utf-8") as fh:
                    text = fh.read()
            except OSError as e:
                return f"Error: cannot read skill {skill['name']}: {e}"
            return self._strip_skill_frontmatter(text)

        # Register the callable so _execute_tool_calls() can run it, then
        # append its schema to self.tools so the model sees the tool in
        # every one_shot() payload (deduping any pre-existing use_skill).
        if self.tool_registry is None:
            self.tool_registry = {}
        self.tool_registry["use_skill"] = use_skill

        schema = self._generate_tool_schema(use_skill)
        current = list(self.tools) if self.tools else []
        current = [
            t for t in current
            if not (
                isinstance(t, dict)
                and t.get("function", {}).get("name") == "use_skill"
            )
        ]
        self.tools = current + [schema]
        return use_skill

#-------------------------- AGENTS.md discovery --------------------------------

    @classmethod
    def _discover_agents_md(cls, agents_path: str) -> str:
        """Recursively scan agents_path for files named exactly 'AGENTS.md'
        and return their combined contents as one prompt block.

        Every match is rendered under a heading carrying its path relative
        to the scanned root (sorted walk order), so the model can tell which
        context came from where. Returns an empty string when no AGENTS.md
        files exist, keeping prompts byte-identical to a no-config object.
        Raises ValueError when the directory is missing/not a folder or a
        file cannot be read.
        """
        root = validate_directory(agents_path)
        blocks: List[str] = []
        try:
            for dirpath, _, filenames in os.walk(root):
                for name in sorted(filenames):
                    if name != "AGENTS.md":
                        continue
                    file_path = os.path.join(dirpath, name)
                    rel_path = os.path.relpath(file_path, root)
                    try:
                        with open(file_path, "r", encoding="utf-8") as fh:
                            text = fh.read()
                    except OSError as e:
                        raise ValueError(
                            f"cannot read AGENTS.md file {file_path}: {e}"
                        )
                    blocks.append(f"## AGENTS.md — {rel_path}\n{text.strip()}")
        except OSError as e:
            raise ValueError(f"cannot scan agents.md directory {root}: {e}")
        if not blocks:
            return ""
        return "\n\n" + "\n\n".join(blocks)

    def _load_file_tools(self) -> None:
        """Load build_file_tools() for the workspace directory into this
        instance's tool set (tool_registry + tools schema).

        Called automatically by __init__ when config["file"] is True. The
        workspace comes from the `directory` constructor argument, which
        agent classes store as self._directory before super().__init__();
        file=True without a directory is a configuration error. Existing
        tools are kept: same-named registry entries are overridden by the
        file tools and their schemas are deduped by function name (the same
        merge pattern _create_skills_tool uses).
        """
        directory = getattr(self, "_directory", None)
        if not directory:
            raise ValueError(
                "config['file'] is True but no workspace directory is "
                "available; pass directory=<workspace> to the agent. Raw "
                "LLM objects have no directory and cannot auto-load file "
                "tools."
            )
        file_tools = build_file_tools(directory)

        if self.tool_registry is None:
            self.tool_registry = {}
        self.tool_registry = {**self.tool_registry, **file_tools}

        names = set(file_tools)
        current = list(self.tools) if self.tools is not None else []
        current = [
            tool for tool in current
            if not (
                isinstance(tool, dict)
                and tool.get("function", {}).get("name") in names
            )
        ]
        new_schemas = [
            self._generate_tool_schema(func) for func in file_tools.values()
        ]
        self.tools = current + new_schemas

#-------------------------- private helpers -------------------------------------

    @staticmethod
    def _generate_tool_schema(func: Callable) -> Dict:
        """
        Generate the OpenAPI-compatible JSON schema for a given function.
        Supports int, float, str, bool, and list types.
        """
        sig = inspect.signature(func)
        params = sig.parameters
        properties = {}
        required = []

        for name, param in params.items():
            if param.annotation == inspect.Parameter.empty:
                param_type = "string"
            elif param.annotation in (int, float):
                param_type = "number"
            elif param.annotation == str:
                param_type = "string"
            elif param.annotation == bool:
                param_type = "boolean"
            elif param.annotation == list:
                param_type = "array"
            else:
                param_type = "string"  # fallback

            properties[name] = {"type": param_type}
            if param.default == inspect.Parameter.empty:
                required.append(name)

        doc = func.__doc__.strip() if func.__doc__ else f"Executes the {func.__name__} operation."

        result = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return result

    def _generate_tools_schema(self) -> List[Dict]:
        """Generate the full tools JSON schema from self.tool_registry."""
        if not self.tool_registry:
            return []
        return [self._generate_tool_schema(func) for func in self.tool_registry.values()]

    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[str]:
        """
        Execute a list of tool calls using the injected tool_registry.
        Returns a list of result strings (observations) in the same order.
        """
        if not self.tool_registry:
            raise RuntimeError("No tool_registry provided. Cannot execute tools.")

        observations = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("arguments", {})  # already a dict
            func = self.tool_registry.get(tool_name)
            if func is None:
                result = f"Error: Unknown tool '{tool_name}'"
            else:
                try:
                    raw_result = func(**tool_args)
                    if isinstance(raw_result, (dict, list)):
                        result = json.dumps(raw_result)
                    else:
                        result = str(raw_result)
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
            observations.append(result)
        return observations

    @staticmethod
    def _classify_errors(parsed: Any, schema: Dict[str, Dict[str, Any]]) -> List[str]:
        """Return a list of human-readable violations of a classify schema."""
        errors = []
        if not isinstance(parsed, dict):
            return ["output must be a JSON object"]
        for key in schema:
            if key not in parsed:
                errors.append(f"missing key '{key}'")
        for key in parsed:
            if key not in schema:
                errors.append(f"unexpected key '{key}'")
                continue
            choices = schema[key]["choices"]
            value = parsed[key]
            if isinstance(choices, dict):
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(f"'{key}' = {value!r} is not an integer")
                elif value < choices["min"] or value > choices["max"]:
                    errors.append(
                        f"'{key}' = {value} is outside "
                        f"[{choices['min']}, {choices['max']}]"
                    )
            elif value not in choices:
                errors.append(f"'{key}' = {value!r} is not one of {choices}")
        return errors

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        """Strip a leading ``` / trailing ``` markdown fence (with optional
        language tag) from a response string, then trim whitespace."""
        if content.startswith("```"):
            content = content[3:]
            if content.startswith("json"):
                content = content[4:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        return content

    @staticmethod
    def _format_tool_arguments(arguments: Any) -> str:
        """Render tool-call arguments (a JSON string or a dict) as 'k=v, k=v'."""
        if arguments is None:
            return ""
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                return arguments  # not JSON: print the raw string verbatim
        else:
            args = arguments
        if not isinstance(args, dict):
            return str(args)
        return ", ".join(
            f"{k}={json.dumps(v, ensure_ascii=False)}"
            for k, v in args.items()
        )

    # ---------- abstract helpers (must be implemented by subclasses) ----------
    @abstractmethod
    def _get_api_key(self) -> str:
        """Retrieve the API key for this provider (from environment or user input)."""
        pass

    @abstractmethod
    def _create_client(self) -> Any:
        """Create and return the provider-specific client instance."""
        pass

    @abstractmethod
    def _build_payload(self) -> Dict[str, Any]:
        """
        Build the provider-specific request payload (dictionary)
        that will be passed to the underlying client.
        """
        pass

#-------------------------- public methods --------------------------------------

    @abstractmethod
    def one_shot(self) -> Dict[str, Any]:
        """
        Send the current memory (using the built payload) to the LLM,
        append the assistant's response to memory, and return a dict containing:
            - "content":  the assistant's text response (str)
            - "tool_calls": list of parsed tool calls (list of dicts)
            - "reasoning": any reasoning output (str or None)
        """
        pass

    def extract(
        self,
        schema: Dict[str, str],
        example: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> Any:
        """
        Extract structured data from the current conversation against the
        given schema.

        schema: dict where keys are the required output fields and values
                describe what each key requires.
        example: optional str showing examples of the schema; injected into
                 the system prompt when provided, ignored when None.
        instruction: optional str with additional overall context/explanation;
                     injected into the system prompt when provided, ignored
                     when None.

        Returns the parsed JSON dict, or the raw content string if JSON
        parsing fails.
        """
        # ---------- validate inputs ----------
        if not isinstance(schema, dict) or not schema:
            raise ValueError("schema must be a non-empty dict")
        for key, desc in schema.items():
            if not isinstance(key, str) or not isinstance(desc, str):
                raise TypeError("schema keys and values must be strings")
        if example is not None and not isinstance(example, str):
            raise TypeError("example must be a str or None")
        if instruction is not None and not isinstance(instruction, str):
            raise TypeError("instruction must be a str or None")

        # ---------- build the extraction system prompt ----------
        schema_json = json.dumps(schema, indent=2)
        system_instruction = (
            "You are a structured extraction engine. Return ONLY valid JSON with exactly "
            "the following keys. Each value describes what that key requires:\n"
            f"{schema_json}"
        )
        if instruction:
            system_instruction += f"\n\nAdditional instructions:\n{instruction}"
        if example:
            system_instruction += f"\n\nExamples:\n{example}"

        # ---------- prepare temporary memory ----------
        saved_memory = self.memory
        non_system = [msg for msg in saved_memory if msg.get("role") != "system"]
        temp_memory = [{"role": "system", "content": system_instruction}] + non_system

        # ---------- temporarily force JSON output and call one_shot ----------
        # Tools are switched off for the extraction call: the conversation may
        # contain tool chatter from earlier agent turns and the extraction
        # engine must answer with pure JSON, never with new tool calls.
        saved_response_format = self.response_format
        saved_tools = self.tools
        saved_tool_choice = self.tool_choice
        saved_managing_context = self._managing_context
        self.response_format = {"type": "json_object"}
        self.tools = None
        self.tool_choice = None
        self._managing_context = True
        self.memory = temp_memory
        try:
            result = self.one_shot()
        finally:
            self.memory = saved_memory
            self.response_format = saved_response_format
            self.tools = saved_tools
            self.tool_choice = saved_tool_choice
            self._managing_context = saved_managing_context

        # ---------- parse and return ----------
        content = (result.get("content") or "").strip()
        if content.startswith("```"):
            content = content[3:]
            if content.startswith("json"):
                content = content[4:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content

    def classify(
        self,
        schema: Dict[str, Dict[str, Any]],
        example: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> Any:
        """
        Classify the current conversation by picking exactly one allowed
        choice per schema key.

        schema: dict where keys are the required output fields and each value
                is a dict of the form:
                    {
                        "description": str describing what the key requires,
                        "choices": either a list of the ONLY allowed values
                                   (e.g. [1, 2, 3] or ["a", "b"]), or a range
                                   given as "1-5" or {"min": 1, "max": 5},
                                   meaning any integer in that inclusive range
                    }
        example: optional str showing examples of the schema; injected into
                 the system prompt when provided, ignored when None.
        instruction: optional str with additional overall context/explanation;
                     injected into the system prompt when provided, ignored
                     when None.

        Returns the parsed JSON dict with each value guaranteed to be one of
        its choices. Raises ValueError if the model persistently violates
        the schema.
        """
        # ---------- validate and normalize inputs ----------
        if not isinstance(schema, dict) or not schema:
            raise ValueError("schema must be a non-empty dict")
        norm_schema = {}
        for key, field in schema.items():
            if not isinstance(key, str):
                raise TypeError("schema keys must be strings")
            if not isinstance(field, dict):
                raise TypeError(f"schema['{key}'] must be a dict")
            if "description" not in field or "choices" not in field:
                raise ValueError(
                    f"schema['{key}'] must contain 'description' and 'choices'"
                )
            if not isinstance(field["description"], str):
                raise TypeError(f"schema['{key}']['description'] must be a str")
            choices = field["choices"]
            if isinstance(choices, dict):
                if "min" not in choices or "max" not in choices:
                    raise ValueError(
                        f"schema['{key}']['choices'] range dict must have 'min' and 'max'"
                    )
                lo, hi = choices["min"], choices["max"]
                if not isinstance(lo, int) or isinstance(lo, bool) or \
                        not isinstance(hi, int) or isinstance(hi, bool):
                    raise TypeError(f"schema['{key}']['choices'] min/max must be ints")
                if lo > hi:
                    raise ValueError(f"schema['{key}']['choices']: min must be <= max")
                norm_schema[key] = {
                    "description": field["description"],
                    "choices": {"min": lo, "max": hi},
                }
            elif isinstance(choices, str):
                parts = choices.split("-")
                if len(parts) != 2:
                    raise ValueError(
                        f"schema['{key}']['choices'] range must look like '1-5'"
                    )
                try:
                    lo = int(parts[0].strip())
                    hi = int(parts[1].strip())
                except ValueError:
                    raise ValueError(
                        f"schema['{key}']['choices'] range must look like '1-5'"
                    )
                if lo > hi:
                    raise ValueError(f"schema['{key}']['choices']: min must be <= max")
                norm_schema[key] = {
                    "description": field["description"],
                    "choices": {"min": lo, "max": hi},
                }
            elif isinstance(choices, list) and choices:
                norm_schema[key] = {
                    "description": field["description"],
                    "choices": choices,
                }
            else:
                raise ValueError(
                    f"schema['{key}']['choices'] must be a non-empty list, a "
                    f"'min-max' range string like '1-5', or a {{'min': m, 'max': n}} dict"
                )
        if example is not None and not isinstance(example, str):
            raise TypeError("example must be a str or None")
        if instruction is not None and not isinstance(instruction, str):
            raise TypeError("instruction must be a str or None")

        # ---------- build the classification system prompt ----------
        lines = []
        for key, field in norm_schema.items():
            choices = field["choices"]
            if isinstance(choices, dict):
                lines.append(
                    f'  "{key}": return an integer between {choices["min"]} and '
                    f'{choices["max"]} inclusive - {field["description"]}'
                )
            else:
                lines.append(
                    f'  "{key}": pick exactly ONE of {json.dumps(choices)} '
                    f'- {field["description"]}'
                )
        system_instruction = (
            "You are a classification engine. Return ONLY valid JSON with exactly "
            "the keys listed below. Each key's choices tell you what is allowed: "
            "a list means you MUST pick exactly one of the listed values; a range "
            "such as \"1-5\" means you MUST return a single integer within that "
            "inclusive range. You may not invent values, use synonyms, add keys, "
            "or omit keys.\n\n"
            "Fields:\n" + "\n".join(lines)
        )
        if instruction:
            system_instruction += f"\n\nAdditional instructions:\n{instruction}"
        if example:
            system_instruction += f"\n\nExamples:\n{example}"

        # ---------- prepare temporary memory ----------
        saved_memory = self.memory
        non_system = [msg for msg in saved_memory if msg.get("role") != "system"]
        temp_memory = [{"role": "system", "content": system_instruction}] + non_system

        # ---------- temporarily force JSON output and call one_shot ----------
        saved_response_format = self.response_format
        saved_managing_context = self._managing_context
        self.response_format = {"type": "json_object"}
        self._managing_context = True
        self.memory = temp_memory
        try:
            result = self.one_shot()
        finally:
            self.memory = saved_memory
            self.response_format = saved_response_format
            self._managing_context = saved_managing_context

        # ---------- parse ----------
        content = self._strip_code_fences((result.get("content") or "").strip())
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            parsed = content

        # ---------- validate against the schema, retry via one_shot on failure ----------
        errors = self._classify_errors(parsed, norm_schema)
        retries = 2
        while errors and retries > 0:
            retries -= 1
            feedback = (
                "Your previous output was invalid. Correct these issues and return "
                "ONLY valid JSON:\n" + "\n".join(errors)
            )
            saved_memory = self.memory
            saved_response_format = self.response_format
            saved_managing_context = self._managing_context
            self.response_format = {"type": "json_object"}
            self._managing_context = True
            self.memory = [
                {"role": "system", "content": system_instruction},
                *non_system,
                {"role": "user", "content": feedback},
            ]
            try:
                result = self.one_shot()
            finally:
                self.memory = saved_memory
                self.response_format = saved_response_format
                self._managing_context = saved_managing_context
            content = self._strip_code_fences((result.get("content") or "").strip())
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                parsed = content
            errors = self._classify_errors(parsed, norm_schema)

        if errors:
            raise ValueError(
                "Model output violates the classify schema: " + "; ".join(errors)
            )
        return parsed

    def summarize(self, context: str, max_tokens: int) -> str:
        """
        Summarize a given context down to fit within max_tokens.

        context: str containing the full text to summarize.
        max_tokens: int upper bound for the summary length.

        Returns the summary text produced by the model. The existing
        memory and max_tokens settings are left untouched.
        """
        # ---------- validate inputs ----------
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context must be a non-empty string")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise TypeError("max_tokens must be an integer")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        # ---------- build temporary memory ----------
        saved_memory = self.memory
        temp_memory = [
            {
                "role": "system",
                "content": (
                    "You are a summarization engine. Summarize the context "
                    "provided by the user into a concise summary that preserves "
                    "all key information, facts, and conclusions. Output only "
                    "the summary text, nothing else."
                ),
            },
            {"role": "user", "content": context},
        ]

        # ---------- temporarily set max_tokens and call one_shot ----------
        saved_max_tokens = self.max_tokens
        saved_tools = self.tools
        saved_tool_choice = self.tool_choice
        saved_managing_context = self._managing_context
        self.max_tokens = max_tokens
        self.tools = None
        self.tool_choice = None
        self._managing_context = True
        self.memory = temp_memory
        try:
            result = self.one_shot()
        finally:
            self.memory = saved_memory
            self.max_tokens = saved_max_tokens
            self.tools = saved_tools
            self.tool_choice = saved_tool_choice
            self._managing_context = saved_managing_context

        return (result.get("content") or "").strip()

    def trim_memory(self, message_range: List[int]) -> None:
        """
        Remove a range of messages from memory without deleting the system
        prompt.

        message_range: two-element list [from, to] with 1-based inclusive
                       positions matching len(self) / len(self.memory).
                       e.g. [1, 10] with 20 messages leaves the system
                       prompt plus messages 11-20.

        Mutates self.memory in place and returns None.
        """
        # ---------- validate inputs ----------
        if not isinstance(message_range, list) or len(message_range) != 2:
            raise TypeError("message_range must be a list of two ints: [from, to]")
        start, end = message_range
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
        ):
            raise TypeError("message_range must be a list of two ints: [from, to]")
        if start < 1:
            raise ValueError("message_range 'from' must be >= 1")
        if end > len(self.memory):
            raise ValueError(
                f"message_range 'to' must be <= len(memory) ({len(self.memory)})"
            )
        if start > end:
            raise ValueError("message_range 'from' must be <= 'to'")

        # ---------- cut the range, preserving system messages ----------
        system_msgs = [m for m in self.memory if m.get("role") == "system"]
        remaining = []
        for i, msg in enumerate(self.memory):
            if msg.get("role") == "system":
                continue  # system prompt is never deleted
            if start - 1 <= i <= end - 1:
                continue  # inside the cut range
            remaining.append(msg)
        self.memory = system_msgs + remaining

    def save_memory(
        self,
        message_range: List[int],
        skip_system: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Getter counterpart of trim_memory().

        Returns a deep copy of the messages inside the given 1-based
        inclusive range [from, to] — the same indexing convention
        trim_memory() uses. Non-destructive: self.memory is left untouched.

        message_range: [from, to] positions in self.memory (1-based,
                       inclusive), matching trim_memory's indexing.
        skip_system:   when True (default), system messages inside the
                       range are excluded from the result, so the returned
                       chunk is the portable conversation that can be
                       appended to another object's memory.

        The returned list is an independent snapshot: later mutations of
        self.memory never affect it.

        Example — hand a conversation over to another agent:

            carry = agent1.save_memory([1, len(agent1.memory)])
            agent2.memory = [
                {"role": "system", "content": agent2._react_prompt},
                *carry,
            ]
        """
        # ---------- validate inputs (same rules as trim_memory) ----------
        if not isinstance(message_range, list) or len(message_range) != 2:
            raise TypeError("message_range must be a list of two ints: [from, to]")
        start, end = message_range
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
        ):
            raise TypeError("message_range must be a list of two ints: [from, to]")
        if not isinstance(skip_system, bool):
            raise TypeError("skip_system must be a bool")
        if start < 1:
            raise ValueError("message_range 'from' must be >= 1")
        if end > len(self.memory):
            raise ValueError(
                f"message_range 'to' must be <= len(memory) ({len(self.memory)})"
            )
        if start > end:
            raise ValueError("message_range 'from' must be <= 'to'")

        # ---------- collect the range, optionally skipping system ----------
        selected = []
        for i, msg in enumerate(self.memory):
            if start - 1 <= i <= end - 1:
                if skip_system and msg.get("role") == "system":
                    continue
                selected.append(msg)
        return copy.deepcopy(selected)

    def format_memory(self) -> str:
        """
        Render the entire memory as a clean, human-readable transcript for
        terminal inspection. This is the string print(agent) shows (via
        __str__) and can also be called directly:

            print(agent)            # prints the transcript
            text = agent.format_memory()  # returns it as a string

        Message layout:

            === FakeLLM conversation (5 messages) ===

            [1] System prompt:
            <system content>

            [2] User:
            <user content>

            [3] Assistant:
            <assistant content>
                -> Tool call: subtract(a=20, b=4)

            [4] Tool call: subtract -> Result:
            16

            [5] Assistant:
            <final answer>

        Tool names on the Result lines are resolved by matching each tool
        message's tool_call_id against the assistant tool_calls stored in
        memory; when no match exists the raw id is shown instead. Malformed
        tool arguments are printed verbatim rather than raising.

        Never mutates memory. Returns the transcript as a single string.
        """
        plural = "" if len(self.memory) == 1 else "s"
        lines = [
            f"=== {self.__class__.__name__} conversation "
            f"({len(self.memory)} message{plural}) ==="
        ]

        # tool_call_id -> function name, so tool observations can be labelled.
        tool_names = {}
        for msg in self.memory:
            for tc in msg.get("tool_calls") or []:
                function = tc.get("function") or {}
                if tc.get("id") is not None and function.get("name"):
                    tool_names[tc["id"]] = function["name"]

        for i, msg in enumerate(self.memory, start=1):
            role = msg.get("role", "unknown")
            content = "" if msg.get("content") is None else str(msg.get("content"))

            if role == "system":
                label = "System prompt"
            elif role == "user":
                label = "User"
            elif role == "assistant":
                label = "Assistant"
            elif role == "tool":
                call_id = msg.get("tool_call_id")
                lines.append(
                    f"[{i}] Tool call: {tool_names.get(call_id, call_id)} -> Result:"
                )
                if content:
                    lines.append(content)
                lines.append("")
                continue
            else:
                label = str(role).capitalize()

            lines.append(f"[{i}] {label}:")
            if content:
                lines.append(content)

            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    function = tc.get("function") or {}
                    name = function.get("name") or tc.get("id") or "?"
                    args = self._format_tool_arguments(function.get("arguments"))
                    lines.append(f"    -> Tool call: {name}({args})")

            lines.append("")

        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)
