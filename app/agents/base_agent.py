
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import httpx
import json


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents
    Each agent (Email, Calendar, Slack) inherits from this
    ✅ UPDATED: Dynamic tool name mapping + instructions parameter for Zapier MCP
    """
    
    def __init__(self, name: str, agent_type: str, agent_id: int = 1):
        self.name = name
        self.agent_type = agent_type  # "email", "calendar", "slack"
        self.agent_id = agent_id
        self.execution_id = str(uuid.uuid4())
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task for this agent
        
        Args:
            task: Task dictionary with action and parameters
        
        Returns:
            Result dictionary with status, data, and errors
        """
        pass
    
    @abstractmethod
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent-specific errors"""
        pass
    
    async def call_zapier_webhook(
        self,
        webhook_url: str,
        action: str,
        parameters: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call Zapier MCP webhook with CORRECT protocol and tool name mapping
        ✅ FIXED: Dynamic tool mapping + proper JSON-RPC format + instructions parameter
        """
        try:
            logger.info(f"[BASE_AGENT: {self.agent_type.upper()}] Calling Zapier for: {action}")
        
        
            
            logger.debug(f"[BASE_AGENT: {self.agent_type.upper()}] Instructions: {json.dumps(parameters)}")
            
            # ✅ CORRECT: Use proper Zapier MCP JSON-RPC format
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": action,
                    "arguments": parameters
                },
                "id": 1
            }
            
            logger.debug(f"[BASE_AGENT: {self.agent_type.upper()}] Zapier payload: {json.dumps(payload)[:300]}")
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            async with httpx.AsyncClient(timeout=100.0) as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    headers=headers
                )
                
                logger.info(f"[BASE_AGENT: {self.agent_type.upper()}] Zapier response: {response.status_code}")
                
                # ✅ Handle both JSON and SSE responses
                if response.status_code == 200:
                    try:
                        # Try JSON response
                        result = response.json()
                        
                        logger.debug(f"[BASE_AGENT: {self.agent_type.upper()}] Zapier result: {json.dumps(result)[:200]}")
                        
                        # Check for error in MCP response
                        if "error" in result:
                            error_data = result.get("error", {})
                            error_msg = error_data.get("message", str(error_data)) if isinstance(error_data, dict) else str(error_data)
                            
                            logger.error(f"[BASE_AGENT: {self.agent_type.upper()}] Zapier error: {error_msg}")
                            return {
                                "status": "failure",
                                "error": f"Zapier error: {error_msg}"
                            }
                        
                        # Success
                        return {
                            "status": "success",
                            "data": result.get("result", result),
                            "message": f"{self.agent_type.capitalize()} action completed"
                        }
                    
                    except json.JSONDecodeError:
                        # Handle SSE format (event stream)
                        logger.debug(f"[BASE_AGENT: {self.agent_type.upper()}] Handling SSE response")
                        
                        lines = response.text.split('\n')
                        for line in lines:
                            if line.startswith('data: '):
                                data = line[6:].strip()
                                try:
                                    response_json = json.loads(data)
                                    
                                    if 'error' in response_json:
                                        logger.error(f"[BASE_AGENT: {self.agent_type.upper()}] Zapier error: {response_json['error']}")
                                        return {
                                            "status": "failure",
                                            "error": response_json.get("error", "Unknown error")
                                        }
                                    
                                    # Success
                                    return {
                                        "status": "success",
                                        "data": response_json,
                                        "message": f"{self.agent_type.capitalize()} action completed"
                                    }
                                except json.JSONDecodeError:
                                    continue
                        
                        # If we get here, response was success but couldn't parse
                        return {
                            "status": "success",
                            "data": {"raw_response": response.text[:500]},
                            "message": f"{self.agent_type.capitalize()} action completed"
                        }
                
                elif response.status_code == 400:
                    logger.error(f"[BASE_AGENT: {self.agent_type.upper()}] Bad request error!")
                    logger.error(f"[BASE_AGENT: {self.agent_type.upper()}] Response: {response.text[:500]}")
                    
                    return {
                        "status": "failure",
                        "error": f"Bad request to Zapier. Check parameters.",
                        "details": response.text[:500]
                    }
                
                else:
                    logger.error(f"[BASE_AGENT: {self.agent_type.upper()}] HTTP error: {response.status_code}")
                    logger.error(f"[BASE_AGENT: {self.agent_type.upper()}] Response: {response.text[:500]}")
                    
                    return {
                        "status": "failure",
                        "error": f"Zapier HTTP error: {response.status_code}",
                        "details": response.text[:200]
                    }
        
        except Exception as e:
            logger.error(f"[BASE_AGENT: {self.agent_type.upper()}] Exception calling Zapier: {e}", exc_info=True)
            return {
                "status": "failure",
                "error": f"Failed to call Zapier: {str(e)}"
            }
    
    def _create_result(
        self,
        status: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create standardized result dictionary
        
        Args:
            status: "success", "failure", "partial"
            data: Result data
            error: Error message if failed
            metadata: Additional metadata
        
        Returns:
            Standardized result
        """
        return {
            "status": status,
            "agent": self.agent_type,
            "execution_id": self.execution_id,
            "timestamp": datetime.now().isoformat(),
            "data": data or {},
            "error": error,
            "metadata": metadata or {},
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information - must be implemented by subclasses"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.agent_type,
            "execution_id": self.execution_id,
        }
