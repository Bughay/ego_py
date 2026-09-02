"""
Built-in file tools for the agents.

These tools let an agent operate on files inside a single workspace
directory. They are meant to be placed inside the LLM tool_registry:

    from agent_logic.agent.builtin_tools.file import build_file_tools

    tool_registry = build_file_tools("/absolute/path/to/workspace")

Every tool is a closure bound to that directory: relative paths are resolved
against it and any path that tries to escape the workspace is rejected.

Tools:
    ls          - list files in a directory
    read_file   - read file contents (line pagination + image note)
    write_file  - create or overwrite a file
    edit_file   - exact string replacements
    delete      - delete a file or a directory tree
    glob        - find files matching a glob pattern
    grep        - search file contents
"""
import shutil
from pathlib import Path
from typing import Callable, Dict

_MAX_READ_CHARS = 20_000
_MAX_GREP_RESULTS = 100
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def validate_directory(directory: str) -> str:
    """Validate a workspace directory and return its resolved absolute path."""
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("directory must be a non-empty string")
    root = Path(directory).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"directory is not a folder: {root}")
    return str(root)


def build_file_tools(directory: str) -> Dict[str, Callable]:
    """
    Build the file tool registry for the given workspace directory.

    Returns a dict mapping tool names to functions so it can be passed
    straight into an LLM/agent constructor as tool_registry.
    """
    root = Path(validate_directory(directory))

    # -------------------------- private helpers --------------------------

    def _resolve(path: str) -> Path:
        """Resolve a tool path argument against the workspace root."""
        p = Path(path)
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        if p != root and not p.is_relative_to(root):
            raise ValueError(f"path escapes the workspace directory: {path}")
        return p

    # -------------------------- the tools --------------------------

    def ls(path: str = ".") -> str:
        """List the files and folders inside a directory (relative to the workspace root)."""
        target = _resolve(path)
        if not target.exists():
            return f"Error: path does not exist: {path}"
        if not target.is_dir():
            return f"Error: path is not a directory: {path}"
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not entries:
            return "(empty directory)"
        lines = [f"[DIR]  {e.name}" if e.is_dir() else f"       {e.name}" for e in entries]
        return "\n".join(lines)

    def read_file(path: str, offset: int = 1, limit: int = 500) -> str:
        """Read the text contents of a file (relative to the workspace root).
        offset: 1-based line number to start reading from.
        limit: maximum number of lines to return (pagination)."""
        target = _resolve(path)
        if not target.exists():
            return f"Error: path does not exist: {path}"
        if target.is_dir():
            return f"Error: path is a directory: {path}"
        if target.suffix.lower() in _IMAGE_SUFFIXES:
            return f"[image] {path} (multimodal content, not readable as text)"
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"Error reading {path}: {exc}"
        lines = text.splitlines()
        if offset < 1:
            return "Error: offset must be >= 1"
        if limit < 1:
            return "Error: limit must be >= 1"
        page = lines[offset - 1: offset - 1 + limit]
        result = "\n".join(page)
        header = f"Lines {offset}-{min(offset + limit - 1, len(lines))} of {len(lines)}:\n"
        return header + result

    def write_file(path: str, content: str) -> str:
        """Create a new file or overwrite an existing one with the given text
        content (relative to the workspace root)."""
        if not isinstance(content, str) or not content:
            return "Error: content must be a non-empty string"
        target = _resolve(path)
        if target.is_dir():
            return f"Error: path is a directory: {path}"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except Exception as exc:
            return f"Error writing {path}: {exc}"
        return f"Wrote {len(content)} characters to {target.relative_to(root)}"

    def edit_file(path: str, old_string: str, new_string: str,
                  replace_all: bool = False) -> str:
        """Replace exact strings inside a file (relative to the workspace root).
        old_string must match exactly once unless replace_all is true."""
        target = _resolve(path)
        if not target.exists():
            return f"Error: path does not exist: {path}"
        if target.is_dir():
            return f"Error: path is a directory: {path}"
        try:
            text = target.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading {path}: {exc}"
        count = text.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}"
        if count > 1 and not replace_all:
            return (f"Error: old_string appears {count} times in {path}; "
                    f"set replace_all=true or use a longer old_string")
        text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        try:
            target.write_text(text, encoding="utf-8")
        except Exception as exc:
            return f"Error writing {path}: {exc}"
        replaced = count if replace_all else 1
        return f"Replaced {replaced} occurrence(s) in {target.relative_to(root)}"

    def delete(path: str) -> str:
        """Delete a file, or a directory and everything inside it (relative
        to the workspace root). The workspace root itself cannot be deleted."""
        target = _resolve(path)
        if target == root:
            return "Error: cannot delete the workspace root"
        if not target.exists():
            return f"Error: path does not exist: {path}"
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except Exception as exc:
            return f"Error deleting {path}: {exc}"
        return f"Deleted {target.relative_to(root)}"

    def glob(pattern: str, path: str = ".") -> str:
        """Find files and folders matching a glob pattern (e.g. '**/*.py')
        under the given directory, relative to the workspace root."""
        base = _resolve(path)
        if not base.exists():
            return f"Error: path does not exist: {path}"
        if not base.is_dir():
            return f"Error: path is not a directory: {path}"
        matches = sorted(
            str(p.relative_to(root))
            for p in base.glob(pattern)
            if p.is_relative_to(root)
        )
        if not matches:
            return f"No matches for '{pattern}' under '{path}'"
        return "\n".join(matches)

    def grep(query: str, path: str = ".", recursive: bool = True) -> str:
        """Search file contents for a literal string and return matching
        lines as 'file:line: text'."""
        if not query:
            return "Error: query must be a non-empty string"
        base = _resolve(path)
        if not base.exists():
            return f"Error: path does not exist: {path}"
        if not base.is_dir():
            return f"Error: path is not a directory: {path}"
        iterator = base.rglob("*") if recursive else base.glob("*")
        results = []
        for p in iterator:
            if not p.is_file():
                continue
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if query in line:
                        results.append(f"{p.relative_to(root)}:{i}: {line}")
            except Exception:
                continue
        if not results:
            return f"No matches for '{query}' under '{path}'"
        if len(results) > _MAX_GREP_RESULTS:
            return (
                f"Found {len(results)} matches (showing first {_MAX_GREP_RESULTS}):\n"
                + "\n".join(results[:_MAX_GREP_RESULTS])
            )
        return "\n".join(results)

    return {
        "ls": ls,
        "read_file": read_file,
        "write_file": write_file,
        "edit_file": edit_file,
        "delete": delete,
        "glob": glob,
        "grep": grep,
    }
