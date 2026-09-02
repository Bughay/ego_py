"""Test suite for the multi-provider LLM wrapper framework.

- test_base_unit.py      : offline unit tests for the shared base class (agent_logic/llm/base.py)
- test_providers_unit.py : offline unit tests for DeepseekLLM / GrokLLM
- fakes.py               : test doubles used by both (no network involved)

The live API-call tests live in main.py (run_all(model)) so they can be
executed manually from the terminal without triggering network calls
during `unittest` runs.
"""
