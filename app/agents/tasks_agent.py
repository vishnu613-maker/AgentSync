
import logging
import re
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .base_agent import BaseAgent
from app.services.message_queue import MessageQueueService
from app.services.llm_service import LLMService
from app.config import get_settings

logger = logging.getLogger(__name__)


class TasksAgent(BaseAgent):
    """Tasks agent with instructions-based Zapier API"""

    def __init__(
        self,
        agent_id: int = 3,
        name: str = "TasksAgent",
        mq_service: Optional[MessageQueueService] = None,
        zapier_mcp_url: Optional[str] = None,
        zapier_api_key: Optional[str] = None,
        llm_service: Optional[LLMService] = None
    ):
        super().__init__(name=name, agent_type="tasks", agent_id=agent_id)
        self.status = "idle"
        self.mq_service = mq_service
        self.zapier_mcp_url = zapier_mcp_url or get_settings().ZAPIER_MCP_SERVER_URL
        self.zapier_api_key = zapier_api_key or get_settings().ZAPIER_API_KEY
        self.llm_service = llm_service

        # ✅ Tool definitions - ALL use "instructions" parameter
        self.tools = [
            {
                "name": "find_task",
                "zapier_name": "google_tasks_find_task",
                "description": "Searches for an incomplete task.",
                "required_params": ["instructions"],
                "optional_params": ["list", "title"],
                "intent_keywords": ["find", "search", "look for", "locate", "check task"]
            },
            {
                "name": "get_tasks_by_list",
                "zapier_name": "google_tasks_get_tasks_by_list",
                "description": "List all tasks by task list",
                "required_params": ["instructions"],
                "optional_params": ["task_list", "show_completed"],
                "intent_keywords": ["list", "show", "get all", "view tasks", "list tasks"]
            },
            {
                "name": "create_task_list",
                "zapier_name": "google_tasks_create_task_list",
                "description": "Creates a new task list.",
                "required_params": ["instructions"],
                "optional_params": ["title"],
                "intent_keywords": ["create list", "new list", "add list"]
            },
            {
                "name": "create_task",
                "zapier_name": "google_tasks_create_task",
                "description": "Creates a new task.",
                "required_params": ["instructions"],
                "optional_params": ["due", "notes", "title", "task_list"],
                "intent_keywords": ["create task", "add task", "new task", "create"]
            },
            {
                "name": "update_task",
                "zapier_name": "google_tasks_update_task",
                "description": "Update an existing task.",
                "required_params": ["instructions"],
                "optional_params": ["due", "notes", "title", "status", "task_id", "task_list"],
                "intent_keywords": ["update", "edit", "modify", "mark", "complete"]
            }
        ]

        logger.info(f"✅ Initialized TasksAgent (ID: {agent_id}, Tools: {len(self.tools)})")

    def _detect_intent(self, user_prompt: str) -> Optional[str]:
        """Detect tasks tool intent from user prompt"""
        user_prompt_lower = user_prompt.lower()

        for tool in self.tools:
            for keyword in tool["intent_keywords"]:
                if keyword in user_prompt_lower:
                    logger.info(f"[TASKS_AGENT] Detected intent: {tool['name']}")
                    return tool['name']

        if any(word in user_prompt_lower for word in ["task", "todo", "create", "add"]):
            return "create_task"

        return None

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tasks action via Zapier webhook"""
        logger.info(f"[TASKS_AGENT] Executing: {task.get('action')}")

        action = task.get("action", "create_task")
        parameters = task.get("parameters", {})
        user_input = task.get("user_input", "")

        self.status = "executing"

        try:
            # ✅ Parameters already include "instructions" from orchestrator
            # No parameter extraction needed!

            # Find tool
            tool = next((t for t in self.tools if t["name"] == action), None)

            if tool:
                result = await self._execute_tool(tool, parameters, user_input)
            else:
                result = self._create_result(
                    status="failure",
                    error=f"Unknown tasks action: {action}"
                )

            self.status = "idle"
            return result

        except Exception as e:
            logger.error(f"[TASKS_AGENT] Error: {e}", exc_info=True)
            self.status = "error"
            return await self.handle_error(e, task)
        

    async def request_context(self, query):
        
        logger.info(f"Context Request Sent to Tasks Agent for query: {query}")
        
        try:
            # ✅ Find tool by action name
            tool = {
                "name": "find_task",
                "zapier_name": "google_tasks_find_task",
                "description": "Searches for an incomplete task.",
                "required_params": ["instructions"],
                "optional_params": ["list", "title"],
                "intent_keywords": ["find", "search", "look for", "locate", "check task"]
            }
            
            if tool:
                parameters = {"instructions": query}
                result = await self._execute_tool(tool, parameters)
            else:
                result = self._create_result(
                    status="failure"
                )
            
            return result
        
        except Exception as e:
            logger.error(f"[EMAIL_AGENT] Error: {e}", exc_info=True)
            return {
                "status": "failure",
                "error": str(e),
                "data": {}
            }


    async def _execute_tool(self, tool: Dict[str, Any], parameters: Dict[str, Any], user_input: Optional[str] = None) -> Dict[str, Any]:
        """Execute any tasks tool dynamically"""
        tool_name = tool["name"]

        logger.info(f"[TASKS_AGENT] ========== EXECUTE_TOOL START ==========")
        logger.info(f"[TASKS_AGENT] Tool: {tool_name}")
        logger.info(f"[TASKS_AGENT] Input params: {json.dumps(parameters)}")

        # ✅ Validate that "instructions" parameter exists
        if "instructions" not in parameters:
            logger.error(f"[TASKS_AGENT] Missing required 'instructions' parameter")
            return self._create_result(
                status="failure",
                error="Missing required parameter: instructions"
            )

        logger.info(f"[TASKS_AGENT] Instructions: {parameters['instructions']}")
        logger.info(f"[TASKS_AGENT] Calling Zapier...")

        # Call Zapier with instructions
        result = await self.call_zapier_webhook(
            webhook_url=self.zapier_mcp_url,
            action=tool["zapier_name"],
            parameters=parameters,
            api_key=self.zapier_api_key
        )

        logger.info(f"[TASKS_AGENT] Zapier result: {result}")
        logger.info(f"[TASKS_AGENT] ========== EXECUTE_TOOL END ==========")

        # Return result
        if result.get("status") == "success":
            return self._create_result(
                status="success",
                data={
                    "tool": tool_name,
                    "description": tool["description"],
                    "result": result.get("data", {}),
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:
            return self._create_result(
                status="failure",
                error=f"Tool '{tool_name}' failed: {result.get('error', 'Unknown error')}"
            )

    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors"""
        return self._create_result(
            status="failure",
            error=f"Tasks agent error: {str(error)}"
        )

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.agent_type,
            "status": self.status,
            "tools": [
                {
                    "name": t["name"],
                    "zapier_name": t["zapier_name"],
                    "description": t["description"],
                    "required_params": t["required_params"],
                    "optional_params": t["optional_params"],
                    "keywords": t["intent_keywords"]
                }
                for t in self.tools
            ],
            "tools_count": len(self.tools)
        }
