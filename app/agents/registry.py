
import logging
from typing import Dict, Optional, List, Any
from app.agents.base_agent import BaseAgent as MCPAgent


logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Singleton registry for agent management
    Works with unified MCPAgent class
    ✅ UPDATED: Now includes dynamic tool discovery
    """
    
    _instance = None
    _agents: Dict[int, MCPAgent] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, agent: MCPAgent) -> bool:
        """Register agent instance"""
        if not isinstance(agent, MCPAgent):
            logger.error(f"❌ Cannot register non-MCPAgent: {type(agent)}")
            return False
        
        self._agents[agent.agent_id] = agent
        logger.info(f"✅ Registered agent: {agent.name} (ID: {agent.agent_id})")
        return True
    
    def get_agent(self, agent_id: int) -> Optional[MCPAgent]:
        """Get agent by ID"""
        return self._agents.get(agent_id)
    
    def get_agent_by_type(self, agent_type: str) -> Optional[MCPAgent]:
        """Get first agent of specific type"""
        for agent in self._agents.values():
            if agent.agent_type == agent_type:
                return agent
        return None
    
    def get_all_agents(self) -> List[MCPAgent]:
        """✅ Get all registered agents"""
        all_agents = []
        for agent in self._agents.values():
            all_agents.append(agent)
        return all_agents
    
    def get_agents_for_user(self, user_id: int) -> List[MCPAgent]:
        """✅ Get agents for specific user"""
        agents = []
        for agent in self._agents.values():
            if agent.user_id == user_id:
                agents.append(agent)
        return agents
    
    # ✅ NEW: Get agent tools for dynamic discovery
    def get_agent_tools(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """
        Get available tools for an agent type
        
        Args:
            agent_type: Type of agent (email, calendar, slack)
            
        Returns:
            Dictionary with agent info and tools list
        """
        agent = self.get_agent_by_type(agent_type)
        if not agent or not hasattr(agent, 'get_info'):
            logger.warning(f"[REGISTRY] Agent type '{agent_type}' not found or missing get_info()")
            return None
        
        try:
            info = agent.get_info()
            
            return {
                "agent_type": agent_type,
                "agent_name": agent.name if hasattr(agent, 'name') else agent_type,
                "tools": [
                    {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "required_params": tool.get("required_params", []),
                        "optional_params": tool.get("optional_params", [])
                    }
                    for tool in info.get("tools", [])
                ]
            }
        except Exception as e:
            logger.error(f"[REGISTRY] Error getting tools for {agent_type}: {e}")
            return None
    
    def list_agents(self, user_id: Optional[int] = None) -> list:
        """List all registered agents (optionally filtered by user)"""
        agents = self._agents.values()
        if user_id:
            agents = [a for a in agents if a.user_id == user_id]
        return [a.get_info() for a in agents]
    
    def deregister(self, agent_id: int) -> bool:
        """Deregister agent"""
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            del self._agents[agent_id]
            logger.info(f"✅ Deregistered agent: {agent.name}")
            return True
        return False
    
    def get_agent_count(self) -> int:
        """Get total number of registered agents"""
        return len(self._agents)
    
    def clear_all(self) -> bool:
        """Clear all registered agents"""
        try:
            self._agents.clear()
            logger.info("✅ Cleared all registered agents")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear agents: {str(e)}")
            return False


# Global registry instance
agent_registry = AgentRegistry()
