# Tests

## Layout

| File                     | What it covers                                                          | Network |
|--------------------------|-------------------------------------------------------------------------|---------|
| `fakes.py`               | Test doubles (`FakeLLM`, `StrictLLM`, fake OpenAI-SDK objects)          | ❌      |
| `test_base_unit.py`      | Unit tests for `agent_logic/llm/base.py` — all shared plumbing          | ❌      |
| `test_providers_unit.py` | Unit tests for `DeepseekLLM` / `GrokLLM` — payloads & response parsing  | ❌      |
| `../main.py`             | Live API-call tests (`run_all(model)`) — one_shot / extract / classify / summarize | ✅ |

## Unit tests (offline — no API keys, no network)

From the project root:

```bash
python -m unittest discover -s test -t . -p "test_*.py" -v
```

or, equivalently:

```bash
python -m unittest -v test.test_base_unit test.test_providers_unit
```

### What each unit-test group verifies

- **Constructor & validation** — `memory` vs `system_prompt`+`user_prompt` exclusivity rules.
- **Property validation** — `model` (non-empty str), `temperature` (0–2), `max_tokens` (positive int ≤ 1M), `memory` shape.
- **Dunders** — `len`, indexing, assignment, deletion, iteration, `repr`, `str`.
- **Tool schema generation** — Python signature → OpenAPI-style JSON schema (types, required params, docstring descriptions).
- **Tool execution** — scalar/dict/list results, unknown tools, in-tool exceptions, ordering, missing registry.
- **extract** — JSON parsing, ```json fence stripping, raw fallback, input validation, memory/response_format restoration.
- **classify** — list & range choices, the retry-on-invalid loop, persistent-failure `ValueError`, input validation.
- **summarize** — result text, memory + max_tokens restoration, input validation.
- **trim_memory** — inclusive range cutting, system prompt preservation, invalid inputs.
- **Providers** — model allowlists, `reasoning_effort` validation (incl. Grok `xhigh`), payload shape (DeepSeek thinking on/off, Grok's omitted temperature), `one_shot` parsing (memory append, tool_calls, reasoning), API-key resolution.

## Live LLM tests (real API calls — run manually)

From the project root:

```bash
python main.py grok              # grok-4.6
python main.py deepseek          # deepseek-v4-flash
python main.py grok-4.6-fast     # any exact model name
```

It prints, in order, four clearly-separated blocks:

```
Function:       <workflow name>
system_prompt:  <system prompt sent>
user_prompt:    <user prompt sent>

result:         <model output>
```

1. **one_shot** — reasoning + content for a plain question.
2. **extract** — structured JSON extraction against a schema.
3. **classify** — one allowed value per key (sentiment / urgency / topic).
4. **summarize** — compresses a synthetic **40-message coding-agent transcript**
   (fintech app: backend endpoints → database → frontend) into a two-paragraph
   summary, with a 5000-token output budget.

API keys must be present in the environment (the provider raises a
RuntimeError if the key is missing — no interactive prompt):
- DeepSeek → `DEEPSEEK_API_KEY`
- Grok (xAI) → `XAI_API_KEY`
