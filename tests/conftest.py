import os
import pytest

# Disable LangSmith tracing globally during tests to prevent background thread hangs and API pings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["HF_TOKEN"] = "dummy-hf-token"
os.environ["GEMINI_API_KEY"] = "test_api_key"
