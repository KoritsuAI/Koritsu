"""
Configuration settings for the application's language models.

This module initializes the language model configuration based on environment variables.
It supports multiple model providers including OpenAI, Anthropic, and Ollama.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Default temperature setting for models (0 = deterministic, higher values = more random)
default_model_temperature = int(os.getenv("DEFAULT_MODEL_TEMPERATURE", "0"))

# Default model provider (OpenAI, Anthropic, or Ollama)
default_model_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "OPENAI").upper()

# Default model name to use (e.g., gpt-4o, claude-3, etc.)
default_model_name = os.getenv("DEFAULT_MODEL_NAME", "gpt-4o")

# Initialize the appropriate language model based on the provider
if default_model_provider == "OPENAI":
    # OpenAI models (like GPT-3.5, GPT-4, etc.)
    default_langchain_model = ChatOpenAI(model_name=default_model_name, temperature=default_model_temperature)
elif default_model_provider == "ANTHROPIC":
    # Anthropic models (like Claude)
    default_langchain_model = ChatAnthropic(model_name=default_model_name, temperature=default_model_temperature)
elif default_model_provider == "OLLAMA":
    # Locally hosted models through Ollama
    default_langchain_model = ChatOpenAI(
        model_name=default_model_name,
        temperature=default_model_temperature,
        openai_api_key="ollama",  # This can be any non-empty string
        openai_api_base="http://IPADDRESS:11434/v1",
    )
else:
    raise ValueError(f"Unsupported model provider: {default_model_provider}")
