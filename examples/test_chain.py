"""
Poem chain workflow — four chained stages recorded as numbered steps
"1".."4" inside ONE session JSON by a single @WorkflowSession.capture
decorator:

    1. LLM one_shot    — write a short poem
    2. LLM classify    — classify the poem (mood / theme / quality)
    3. LLM extract     — pull structured fields out of the poem
    4. EgoAgent (file) — search the workspace, then save the poem +
                         classification + extraction into a file there

Every stage's config is a plain dict; BaseLLM validates and normalizes
each one through the ConfigModel schema from ego_py/llm/config.py (the
single source of truth for the config schema) at construction time:

    self.config = ConfigModel.from_dict(config).to_dict()

All four objects share the same config["session_path"], so the whole
chain lands in one numbered session file:

    <SESSION_DIR>/poem_chain_workflow-<timestamp>.json

Run directly:

    python examples/test_chain.py
"""
import json
import os
import sys

# Make `ego_py` importable when the script is run directly from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ego_py import EgoAgent, LLM, WorkflowSession  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "example_directory"))
SKILLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "skills"))
SESSION_DIR = os.path.join(PROJECT_ROOT, "sessions")
ARTIFACT_NAME = "poem_analysis.json"

# WorkflowSession never auto-creates directories: they must exist up front.
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)


def _step(n, title: str) -> None:
    print("\n" + "-" * 70)
    print(f"CHAIN STEP {n}: {title}")
    print("-" * 70)


@WorkflowSession.capture("poem_chain_workflow")
def workflow_poem_chain():
    """One decorated workflow = one session JSON with steps "1".."4"."""

    # ---- step 1: one_shot — write the poem ----------------------------- #
    poet = LLM(
        model="deepseek-v4-flash",
        system_prompt="You are a poet. Write only the poem, nothing else.",
        user_prompt="Write a short 8-line poem about the sea at dawn.",
        reasoning_effort="low",
        config={"session_path": SESSION_DIR},
    )
    _step(1, "one_shot — the poem is written")
    poem = (poet.one_shot().get("content") or "").strip()
    print(poem)

    # ---- step 2: classify — label the poem ------------------------------ #
    classifier = LLM(
        model="deepseek-v4-flash",
        system_prompt="Classify poems.",
        user_prompt=poem,
        config={"session_path": SESSION_DIR},
    )
    _step(2, "classify — the poem gets labeled")
    labels = classifier.classify(
        schema={
            "mood": {
                "description": "dominant mood of the poem",
                "choices": ["calm", "melancholic", "joyful", "mysterious"],
            },
            "theme": {
                "description": "the poem's central theme",
                "choices": ["nature", "love", "time", "loss", "hope"],
            },
            "quality": {
                "description": "overall quality of the poem",
                "choices": "1-5",
            },
        },
        instruction="Classify the poem above.",
    )
    print(json.dumps(labels, indent=2))

    # ---- step 3: extract — structured fields from the poem --------------- #
    extractor = LLM(
        model="deepseek-v4-flash",
        system_prompt="Extract data.",
        user_prompt=poem,
        config={"session_path": SESSION_DIR},
    )
    _step(3, "extract — structured fields pulled from the poem")
    data = extractor.extract(
        schema={
            "first_line": "the poem's first line, exactly as written",
            "last_line": "the poem's last line, exactly as written",
            "main_image": "the dominant image or metaphor of the poem",
            "rhyme_scheme": "the rhyme scheme, e.g. 'ABAB' or 'free verse'",
            "line_count": "the number of lines as an integer",
        },
        example=json.dumps({
            "first_line": "The waves roll in",
            "last_line": "and the day begins",
            "main_image": "the sea at dawn",
            "rhyme_scheme": "ABAB",
            "line_count": 8,
        }),
    )
    print(json.dumps(data, indent=2))

    # ---- step 4: file agent — search + save the artifact ----------------- #
    artifact = json.dumps(
        {"poem": poem, "classification": labels, "analysis": data},
        indent=2,
        ensure_ascii=False,
    )
    archivist = EgoAgent(
        agent="react",
        model="deepseek-v4-flash",
        max_tokens=10000,
        instruction="You archive text artifacts into the workspace.",
        directory=WORKSPACE_DIR,
        reasoning_effort="low",
        config={"session_path": SESSION_DIR, "file": True},
    )
    _step(4, "file agent — the artifact is searched for, then saved")
    task = (
        f"First use glob to search the workspace for any existing file "
        f"named {ARTIFACT_NAME}. Then save the following JSON artifact "
        f"named {ARTIFACT_NAME}. Then save the following TEXT artifact "
        f"into a file named {ARTIFACT_NAME} in the workspace directory, "
        f"and read the file back to confirm it was written correctly:\n\n"
        f"{artifact}"
    )
    result = archivist.run(task, max_steps=10)
    print(f"agent reply: {result['content']!r}")
    print(f"agent file tools auto-loaded: {sorted(archivist.tool_registry)}")

    # ---- verification — the chain's end product on disk ------------------ #
    path = os.path.join(WORKSPACE_DIR, ARTIFACT_NAME)
    _step("\u2713", f"verification — {ARTIFACT_NAME} on disk")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        try:
            saved = json.loads(raw)
            print(json.dumps(saved, indent=2)[:2000])
        except json.JSONDecodeError:
            print(raw[:2000])
    else:
        print(f"NOT SAVED — the agent did not write {path}; "
              "tool execution is recorded in the session JSON.")


def main():
    from dotenv import load_dotenv

    load_dotenv()
    workflow_poem_chain()


if __name__ == "__main__":
    main()
