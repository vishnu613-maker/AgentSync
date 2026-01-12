"""
Configuration management for AgentSync
Loads settings from .env file and provides typed access
✅ UPDATED: Added Zapier MCP Server configuration
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # ============================================================================
    # APPLICATION SETTINGS
    # ============================================================================
    environment: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    
    # ============================================================================
    # DATABASE CONFIGURATION
    # ============================================================================
    database_url: str
    redis_url: str = "redis://localhost:6379"
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "agent_contexts"
    chroma_embedding_model: str = "all-minilm-l6-v2"
    context_search_top_k: int = 5
    context_cache_ttl: int = 300  # 5 minutes
    enable_context_coordination: bool = True

    
    # ============================================================================
    # LLM CONFIGURATION (Ollama, Mock, or Groq)
    # ============================================================================
    llm_mode: str = "mock"  # "mock", "ollama", or "groq"
    ollama_url: str = "http://localhost:11434"  # Ollama server URL
    groq_api_key: str = ""  # Leave empty unless using Groq
    
    # ============================================================================
    # EMAIL CONFIGURATION (Legacy - kept for reference)
    # ============================================================================
    email_address: str = ""
    email_password: str = ""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    
    # ============================================================================
    # GOOGLE CALENDAR (Legacy - kept for reference)
    # ============================================================================
    google_credentials_path: str = "./credentials.json"
    
    # ============================================================================
    # SLACK CONFIGURATION (Legacy - kept for reference)
    # ============================================================================
    slack_bot_token: str = ""
    slack_channel: str = "#agentsync-test"
    
    # ============================================================================
    # SECURITY
    # ============================================================================
    secret_key: str
    jwt_algorithm: str = "HS256"
    
    # ============================================================================
    # AGENT COMMUNICATION SETTINGS
    # ============================================================================
    agent_context_timeout: float = 5.0  # seconds to wait for context response
    enable_agent_to_agent_context: bool = True
    context_cache_ttl: int = 3600  # seconds to cache context
    max_context_retries: int = 3
    
    # ============================================================================
    # ✅ MCP (MODEL CONTEXT PROTOCOL) CONFIGURATION - PHASE 5.5
    # ============================================================================
    
    # Legacy MCP Server URL (deprecated, use ZAPIER_MCP_SERVER_URL instead)
    mcp_server_url: str = os.getenv(
        "MCP_SERVER_URL",
        "https://mcp.zapier.com/api/mcp/s/default"
    )
    
    mcp_auth_token: str = os.getenv("MCP_AUTH_TOKEN", "")
    mcp_timeout: float = float(os.getenv("MCP_TIMEOUT", "30.0"))
    
    mcp_email_service: str = os.getenv("MCP_EMAIL_SERVICE", "gmail")
    mcp_calendar_service: str = os.getenv("MCP_CALENDAR_SERVICE", "google-calendar")
    mcp_slack_workspace: str = os.getenv("MCP_SLACK_WORKSPACE", "")
    
    mcp_enable_email_agent: bool = os.getenv("MCP_ENABLE_EMAIL_AGENT", "True").lower() == "true"
    mcp_enable_calendar_agent: bool = os.getenv("MCP_ENABLE_CALENDAR_AGENT", "True").lower() == "true"
    mcp_enable_slack_agent: bool = os.getenv("MCP_ENABLE_SLACK_AGENT", "True").lower() == "true"
    
    # ============================================================================
    # ✅ ZAPIER MCP SERVER CONFIGURATION - NEW (Phase 6)
    # ============================================================================
    # Your Zapier MCP Server URL - Get this from https://mcp.zapier.com
    # Format: https://mcp.zapier.com/api/mcp/s/{SERVER_ID}/mcp
    ZAPIER_MCP_SERVER_URL: str = os.getenv(
        "ZAPIER_MCP_SERVER_URL",
        "https://mcp.zapier.com/api/mcp/s/default/mcp"
    )
    
    # Zapier API Key (optional - if needed for authentication)
    ZAPIER_API_KEY: str = os.getenv("ZAPIER_API_KEY", "")
    
    # Zapier Webhook URLs (for optional webhook-based integration)
    # These are alternatives to MCP - kept for backward compatibility
    ZAPIER_EMAIL_WEBHOOK: str = os.getenv("ZAPIER_EMAIL_WEBHOOK", "")
    ZAPIER_CALENDAR_WEBHOOK: str = os.getenv("ZAPIER_CALENDAR_WEBHOOK", "")
    ZAPIER_SLACK_WEBHOOK: str = os.getenv("ZAPIER_SLACK_WEBHOOK", "")
    
    # Zapier MCP Settings
    zapier_mcp_timeout: float = float(os.getenv("ZAPIER_MCP_TIMEOUT", "30.0"))
    zapier_mcp_retry_attempts: int = int(os.getenv("ZAPIER_MCP_RETRY_ATTEMPTS", "3"))
    zapier_mcp_retry_delay: float = float(os.getenv("ZAPIER_MCP_RETRY_DELAY", "1.0"))
    
    # Enable/Disable Zapier MCP for each agent
    zapier_email_enabled: bool = os.getenv("ZAPIER_EMAIL_ENABLED", "True").lower() == "true"
    zapier_calendar_enabled: bool = os.getenv("ZAPIER_CALENDAR_ENABLED", "True").lower() == "true"
    zapier_slack_enabled: bool = os.getenv("ZAPIER_SLACK_ENABLED", "True").lower() == "true"
    
    # ============================================================================
    # CONTEXT COORDINATION SETTINGS 
    # ============================================================================
    
    # Context Storage
    context_ttl_days: int = 30  # How long to keep contexts in DB
    context_max_storage: int = 10000  # Max contexts to store per agent
    
    # Context Retrieval
    context_retrieval_top_k: int = 5  # Default number of contexts to retrieve
    context_min_relevance: float = 0.5  # Minimum relevance score (0-1)
    context_max_age_days: int = 30  # Only search contexts from last N days
    
    # Context Enrichment
    context_enrichment_enabled: bool = True  # Enable/disable context enrichment
    context_chromadb_enabled: bool = True  # Enable Tier 2 (ChromaDB search)
    context_agent_request_enabled: bool = True  # Enable Tier 3 (cross-agent context)
    context_enrichment_timeout: int = 10  # Timeout per enrichment tier (seconds)
    
    # Context Analysis (LLM)
    context_analysis_model: str = "phi3.5:latest"  # Model for context analysis
    context_analysis_temperature: float = 0.1  # Low temperature for consistency
    context_max_retries: int = 2  # Retries for failed LLM analysis
    
    # Agent Context Request
    agent_context_timeout: int = 5  # Timeout for agent context requests
    agent_context_max_agents: int = 3  # Max agents to query for context
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Uses lru_cache to ensure settings are loaded only once
    """
    return Settings()