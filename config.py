"""
Configuration settings for the application's language models.

This module initializes the language model configuration based on environment variables.
It supports multiple model providers including OpenAI, Anthropic, Ollama, and Condor.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage
import requests
import json

# Default temperature setting for models (0 = deterministic, higher values = more random)
default_model_temperature = int(os.getenv("DEFAULT_MODEL_TEMPERATURE", "0"))

# Default model provider (OpenAI, Anthropic, Ollama, or Condor)
default_model_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "OPENAI").upper()

# Default model name to use (e.g., gpt-4o, claude-3, etc.)
default_model_name = os.getenv("DEFAULT_MODEL_NAME", "gpt-4o")


# Condor chat model implementation
class CondorChatModel(BaseChatModel):
    """Chat model that uses the Condor API."""
    
    api_url: str
    model_name: str
    temperature: float
    
    def __init__(self, api_url: str, model_name: str = "condor-40b", temperature: float = 0.0):
        """Initialize the Condor chat model."""
        super().__init__()
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a response from the Condor model."""
        # Convert LangChain messages to Condor API format
        condor_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                condor_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                condor_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                condor_messages.append({"role": "assistant", "content": message.content})
        
        # Prepare request data
        request_data = {
            "messages": condor_messages,
            "temperature": self.temperature,
            "max_tokens": 2048,
        }
        
        # Make request to Condor API
        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Handle response
        if response.status_code == 200:
            response_data = response.json()
            return {"generations": [{"text": response_data["content"], "message": AIMessage(content=response_data["content"])}]}
        else:
            raise Exception(f"Condor API returned error: {response.status_code} - {response.text}")
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "condor"


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
elif default_model_provider == "CONDOR":
    # Local Condor model
    condor_api_url = os.getenv("CONDOR_API_URL", "http://localhost:8001")
    condor_model_name = os.getenv("CONDOR_MODEL_NAME", "condor-40b")
    default_langchain_model = CondorChatModel(
        api_url=condor_api_url,
        model_name=condor_model_name,
        temperature=default_model_temperature,
    )
else:
    raise ValueError(f"Unsupported model provider: {default_model_provider}")
