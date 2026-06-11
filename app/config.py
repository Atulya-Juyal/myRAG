"""
Centralized configuration
Use pydantic to define settings and load them from environment variables or .env file
"""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):

    #llm configuration
    gemini_api_key: str
    primary_llm: str = "gemini-2.5-flash"
    fallback_llm: str = "gemini-2.5-flash"
    jina_api_key: str | None = None
    hf_token: str | None = None
    gemini_api_endpoint: str | None = None
    render_ping_url: str | None = None

    #database configuration
    database_url: str

    #langsmith configuration
    langchain_tracing_v2: bool = True
    langchain_api_key: str
    langchain_project: str = "myRAG"
    
    #application configuration
    app_env: str = "production"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    
@lru_cache()
def get_settings() -> Settings:
    """Cache settings instance - load once - reused everywhere"""
    settings = Settings()
    
    # Inject LangSmith settings into os.environ for tracing
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        if settings.langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
            
    return settings