"""
Agents module
Multi-agent framework for AgentSync
"""
from app.agents.base_agent import BaseAgent
# from app.agents.registry import agent_registry, AgentRegistry
from app.agents.email_agent import EmailAgent

__all__ = [
    "BaseAgent",
    "AgentRegistry",
    # "agent_registry",
    "EmailAgent",
]
