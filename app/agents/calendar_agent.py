

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from .base_agent import BaseAgent
from app.services.message_queue import MessageQueueService
from app.config import get_settings

logger = logging.getLogger(__name__)


class CalendarAgent(BaseAgent):
    """Calendar agent with instructions-based Zapier API"""

    def __init__(
        self,
        agent_id: int,
        name: str = "CalendarAgent",
        user_id: int = 1,
        mq_service: Optional[MessageQueueService] = None,
        zapier_mcp_url: Optional[str] = None,
        zapier_api_key: Optional[str] = None
    ):
        super().__init__(name=name, agent_type="calendar", agent_id=agent_id, user_id=user_id)

        self.agent_id = agent_id
        self.user_id = user_id
        self.status = "idle"

        self.mq_service = mq_service
        self.zapier_mcp_url = zapier_mcp_url or get_settings().ZAPIER_MCP_SERVER_URL
        self.zapier_api_key = zapier_api_key or get_settings().ZAPIER_API_KEY

        # ✅ UPDATED: All tools now only require "instructions" parameter
        self.tools = [
            {
                "name": "create_event",
                "zapier_name": "google_calendar_create_detailed_event",
                "description": "Create a new calendar event",
                "required_params": ["instructions"],  # ✅ ONLY instructions
                "optional_params": []
            },
            {
                "name": "quick_add_event",
                "zapier_name": "google_calendar_quick_add_event",
                "description": "Create an event from a piece of text. Google parses the text for date, time, and description info.", 
                "required_params": ["instructions"],
                "optional_params": ["text", "attendees", "calendarid"],
                "intent_keywords": []
            },
            {
                "name": "update_event",
                "zapier_name": "google_calendar_update_event",
                "description": "Update an existing calendar event",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "delete_event",
                "zapier_name": "google_calendar_delete_event",
                "description": "Delete a calendar event",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "find_events",
                "zapier_name": "google_calendar_find_events",
                "description": "Find a specific calendar event",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "find_busy_periods_in_calendar",
                "zapier_name": "google_calendar_find_busy_periods_in_calendar",
                "description": "Finds busy time periods in your calendar for a specific timeframe.",
                "required_params": ["instructions"],
                "optional_params": ["end_time", "calendarid", "start_time"],
                "intent_keywords": []
            },
            {
                "name": "find_calendars",
                "zapier_name": "google_calendar_find_calendars",
                "description": "Get comprehensive list of calendars accessible to the user with their properties and access levels. Returns up to 250 matching calendars.",
                "required_params": ["instructions"],
                "optional_params": ["showHidden", "showDeleted", "minAccessRole"],
                "intent_keywords": []
            },
            {
                "name": "add_attendee_s_to_event",
                "zapier_name": "google_calendar_add_attendee_s_to_event",
                "description": "Invites one or more person to an existing event.",
                "required_params": ["instructions"],
                "optional_params": ["eventid", "attendees", "calendarid"],
                "intent_keywords": []
            },
        ]

        logger.info(f"✅ Initialized CalendarAgent (ID: {agent_id}, Tools: {len(self.tools)})")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute calendar task via instructions from orchestrator

        Task format: {
            "action": "create_event",
            "parameters": {
                "instructions": "Create event tomorrow at 3pm called standup meeting"
            },
            "user_input": "original user input"
        }
        """
        logger.info(f"[CALENDAR_AGENT] Executing: {task.get('action')}")

        action = task.get("action", "list_events")
        parameters = task.get("parameters", {})
        user_input = task.get("user_input", "")

        self.status = "executing"

        try:
            # ✅ Find tool by action name
            tool = next((t for t in self.tools if t["name"] == action), None)

            if tool:
                result = await self._execute_tool(tool, parameters, user_input)
            else:
                result = self._create_result(
                    status="failure",
                    error=f"Unknown calendar action: {action}"
                )

            self.status = "idle"
            return result

        except Exception as e:
            logger.error(f"[CALENDAR_AGENT] Error: {e}", exc_info=True)
            self.status = "error"
            return await self.handle_error(e, task)

    async def _execute_tool(self, tool: Dict[str, Any], parameters: Dict[str, Any], user_input: str = "") -> Dict[str, Any]:
        """
        Execute any calendar tool dynamically using instructions

        This is the unified method that works for ALL calendar tools.
        No more individual methods for each action!
        """
        tool_name = tool["name"]

        logger.info(f"[CALENDAR_AGENT] ========== EXECUTE_TOOL START ==========")
        logger.info(f"[CALENDAR_AGENT] Tool: {tool_name}")
        logger.info(f"[CALENDAR_AGENT] Input params: {json.dumps(parameters)}")

        # ✅ Validate that "instructions" parameter exists
        if "instructions" not in parameters:
            logger.error(f"[CALENDAR_AGENT] Missing required 'instructions' parameter")
            return self._create_result(
                status="failure",
                error="Missing required parameter: instructions"
            )

        instructions = parameters.get("instructions", "")
        logger.info(f"[CALENDAR_AGENT] Instructions: {instructions}")

        # ✅ Prepare cleaned parameters for Zapier
        cleaned_params = parameters.copy()

        logger.info(f"[CALENDAR_AGENT] Calling Zapier with tool: {tool['zapier_name']}")

        # ✅ Call Zapier webhook with instructions
        result = await self.call_zapier_webhook(
            webhook_url=self.zapier_mcp_url,
            action=tool["zapier_name"],
            parameters=cleaned_params,
            api_key=self.zapier_api_key
        )

        logger.info(f"[CALENDAR_AGENT] Zapier result status: {result.get('status')}")
        logger.info(f"[CALENDAR_AGENT] ========== EXECUTE_TOOL END ==========")

        # ✅ Return result based on Zapier response
        if result.get("status") == "success":
            return self._create_result(
                status="success",
                data={
                    "tool": tool_name,
                    "description": tool["description"],
                    "instructions": instructions,
                    "result": result.get("data", {}),
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"[CALENDAR_AGENT] Tool '{tool_name}' failed: {error_msg}")
            return self._create_result(
                status="failure",
                error=f"Tool '{tool_name}' failed: {error_msg}"
            )

    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors gracefully"""
        logger.error(f"[CALENDAR_AGENT] Error handler: {error}", exc_info=True)
        return self._create_result(
            status="failure",
            error=f"Calendar agent error: {str(error)}"
        )

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.agent_type,
            "status": self.status,
            "user_id": self.user_id,
            "tools": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "required_params": t["required_params"],
                    "zapier_name": t["zapier_name"]
                }
                for t in self.tools
            ],
            "tools_count": len(self.tools)
        }
