import os
import json
from typing import List, Dict, Optional, Any, Callable
from egoai.provider_api import EgoOpenAI
from egoai.llm.base import BaseLLM

class DeepseekLLM(BaseLLM):
    """
    DeepSeek API wrapper. Inherits memory management, tool handling, 
    and chat loop from BaseLLM. Adds DeepSeek-specific:
    - reasoning_effort (thinking mode)
    - model validation (specific model names)
    - API key required: DEEPSEEK_API_KEY env var (raises if missing)
    """

#-------------------------- magic methods --------------------------------------

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        memory: Optional[List[Dict[str, str]]] = None,
        reasoning_effort: Optional[str] = None,
        max_tokens: int = 10000,
        temperature: float = 0.5,
        response_format: Optional[Any] = None,
        tool_registry: Optional[Dict[str, Callable]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Any] = None,
        config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            memory=memory,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            tool_registry=tool_registry,
            tools=tools,
            tool_choice=tool_choice,
            config=config,
        )
        self.reasoning_effort = reasoning_effort
        self._client = self._create_client()

#-------------------------- properties -----------------------------------------
    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model must be a non-empty string")
        allowed_models = [
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "deepseek-flash"
        ]
        if value not in allowed_models:
            raise ValueError(f"Available DeepSeek models: {allowed_models}")
        self._model = value

    @property
    def reasoning_effort(self):
        return self._reasoning_effort

    @reasoning_effort.setter
    def reasoning_effort(self, value):
        allowed = [None, "low", "medium", "high"]
        if value not in allowed:
            raise ValueError(f"reasoning_effort must be one of {allowed}")
        self._reasoning_effort = value

#-------------------------- private helpers -------------------------------------

    def _get_api_key(self) -> str:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set in the environment. "
                "Export it in your shell or load it from .env before running."
            )
        return key

    def _create_client(self):
        api_key = self._get_api_key()
        return EgoOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=600.0,
        )

    def _build_payload(self) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": self.memory,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.response_format:
            payload["response_format"] = self.response_format
        if self.tools:
            payload["tools"] = self.tools
        if self.tool_choice:
            payload["tool_choice"] = self.tool_choice

        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
            payload["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            payload["reasoning_effort"] = None
            payload["extra_body"] = {"thinking": {"type": "disabled"}}

        return payload

#-------------------------- public methods --------------------------------------

    def one_shot(self) -> Dict[str, Any]:
        self._manage_context()
        payload = self._build_payload()

        response = self._client.chat.completions.create(**payload)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._tokens_used += int(getattr(usage, "total_tokens", 0) or 0)
        message = response.choices[0].message

        assistant_msg = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,  # JSON string
                    }
                }
                for tc in message.tool_calls
            ]
        self.memory.append(assistant_msg)

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}  
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        reasoning = getattr(message, 'reasoning_content', None)

        result = {
            "reasoning": reasoning,
            "content": message.content,
            "tool_calls": tool_calls,
        }
        self._print_output(result)
        return result
