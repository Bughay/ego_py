"""
Built-in math tools for the agents.

These tools let an agent do arithmetic without doing the math itself in its
head. Like the file tools, they are meant to be placed inside the LLM
tool_registry explicitly:

    from agent_logic.agent.builtin_tools.math import build_math_tools

    tool_registry = build_math_tools()

Tools:
    add       - a + b
    subtract  - a - b
    multiply  - a * b
    divide    - a / b (division by zero returns an error string)
"""
from typing import Callable, Dict


def add(a: float, b: float) -> float:
    """Add two numbers together and return their sum."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first and return the difference."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers together and return their product."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide the first number by the second and return the quotient."""
    if b == 0:
        return "Error: division by zero"
    return a / b


def build_math_tools() -> Dict[str, Callable]:
    """
    Build the math tool registry.

    Returns a dict mapping tool names to functions so it can be passed
    straight into an LLM/agent constructor as tool_registry.
    """
    return {
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
    }
