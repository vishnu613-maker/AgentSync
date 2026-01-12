

import logging
import re
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_agent import BaseAgent
from app.services.message_queue import MessageQueueService
from app.services.llm_service import LLMService
from app.config import get_settings

logger = logging.getLogger(__name__)


class EmailAgent(BaseAgent):
    """Email agent with instructions-based Zapier API"""

    def __init__(
        self,
        agent_id: int,
        name: str = "EmailAgent",
        user_id: int = 1,
        mq_service: Optional[MessageQueueService] = None,
        zapier_mcp_url: Optional[str] = None,
        zapier_api_key: Optional[str] = None,
        llm_service: Optional[LLMService] = None
    ):
        super().__init__(name=name, agent_type="email", agent_id=agent_id, user_id=user_id)
        
        self.status = "idle"
        self.mq_service = mq_service
        self.zapier_mcp_url = zapier_mcp_url or get_settings().ZAPIER_MCP_SERVER_URL
        self.zapier_api_key = zapier_api_key or get_settings().ZAPIER_API_KEY
        self.llm_service = llm_service

        # ✅ UPDATED: All tools now only require "instructions" parameter
        self.tools = [
            {
                "name": "send_email",
                "zapier_name": "gmail_send_email",
                "description": "Send a new email message",
                "required_params": ["instructions"],  # ✅ ONLY instructions
                "optional_params": []
            },
            {
                "name": "find_email",
                "zapier_name": "gmail_find_email",
                "description": "Find an email message",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "get_attachment",
                "zapier_name": "gmail_get_attachment_by_filename",
                "description": "Retrieve email attachment by filename",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "delete_email",
                "zapier_name": "gmail_delete_email",
                "description": "Delete email from mailbox",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "archive_email",
                "zapier_name": "gmail_archive_email",
                "description": "Archive email message",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "add_label",
                "zapier_name": "gmail_add_label_to_email",
                "description": "Add label to email",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "reply_email",
                "zapier_name": "gmail_reply_to_email",
                "description": "Send reply to email",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "remove_label_conversation",
                "zapier_name": "gmail_remove_label_from_conversation",
                "description": "Remove label from all emails in conversation",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "remove_label_email",
                "zapier_name": "gmail_remove_label_from_email",
                "description": "Remove label from specific email",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "create_label",
                "zapier_name": "gmail_create_label",
                "description": "Create a new label",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "create_draft_reply",
                "zapier_name": "gmail_create_draft_reply",
                "description": "Create draft reply to email",
                "required_params": ["instructions"],
                "optional_params": []
            },
            {
                "name": "create_draft",
                "zapier_name": "gmail_create_draft",
                "description": "Create draft email",
                "required_params": ["instructions"],
                "optional_params": []
            }
        ]

        logger.info(f"✅ Initialized EmailAgent (ID: {agent_id}, Tools: {len(self.tools)})")

    def _extract_email(self, text: str) -> str:
        """Extract clean email from text"""
        text = text.strip()
        
        # Try to extract from mailto link
        mailto_match = re.search(r'mailto:([\w\.-]+@[\w\.-]+\.\w+)', text)
        if mailto_match:
            email = mailto_match.group(1)
            logger.debug(f"[EMAIL_AGENT] Extracted email from mailto: {email}")
            return email
        
        # Try to extract plain email address
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            email = email_match.group(0)
            logger.debug(f"[EMAIL_AGENT] Extracted email: {email}")
            return email
        
        logger.warning(f"[EMAIL_AGENT] No valid email found in: {text}")
        return text

    def _extract_all_emails(self, text: str) -> List[str]:
        """Extract ALL email addresses from text"""
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, text)
        
        if emails:
            cleaned_emails = [self._extract_email(email) for email in emails]
            unique_emails = []
            for email in cleaned_emails:
                if email not in unique_emails:
                    unique_emails.append(email)
            
            logger.info(f"[EMAIL_AGENT] Extracted emails: {unique_emails}")
            return unique_emails
        
        logger.warning(f"[EMAIL_AGENT] No emails found in: {text}")
        return []

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute email task via instructions from orchestrator
        
        Task format: {
            "action": "send_email",
            "parameters": {
                "instructions": "Send email to john@example.com with subject Hello"
            },
            "user_input": "original user input"
        }
        """
        logger.info(f"[EMAIL_AGENT] Executing: {task.get('action')}")
        
        action = task.get("action", "send_email")
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
                    error=f"Unknown email action: {action}"
                )
            
            self.status = "idle"
            return result
        
        except Exception as e:
            logger.error(f"[EMAIL_AGENT] Error: {e}", exc_info=True)
            self.status = "error"
            return await self.handle_error(e, task)

    async def _execute_tool(self, tool: Dict[str, Any], parameters: Dict[str, Any], user_input: str = "") -> Dict[str, Any]:
        """
        Execute any email tool dynamically using instructions
        
        This is the unified method that works for ALL email tools.
        """
        tool_name = tool["name"]
        
        logger.info(f"[EMAIL_AGENT] ========== EXECUTE_TOOL START ==========")
        logger.info(f"[EMAIL_AGENT] Tool: {tool_name}")
        logger.info(f"[EMAIL_AGENT] Input params: {json.dumps(parameters)}")
        
        # ✅ Validate that "instructions" parameter exists
        if "instructions" not in parameters:
            logger.error(f"[EMAIL_AGENT] Missing required 'instructions' parameter")
            return self._create_result(
                status="failure",
                error="Missing required parameter: instructions"
            )
        
        instructions = parameters.get("instructions", "")
        logger.info(f"[EMAIL_AGENT] Instructions: {instructions}")
        
        # # ✅ Prepare cleaned parameters for Zapier
        # cleaned_params = parameters.copy()
        
        # # ✅ Special handling for send_email: extract recipient emails from instructions if needed
        # if tool_name == "send_email":
        #     emails = self._extract_all_emails(instructions)
        #     if emails and "to" not in parameters:
        #         cleaned_params["to"] = emails
        #         logger.info(f"[EMAIL_AGENT] Extracted recipients from instructions: {cleaned_params['to']}")
        
        logger.info(f"[EMAIL_AGENT] Calling Zapier with tool: {tool['zapier_name']}")
        
        # ✅ Call Zapier webhook with instructions
        result = await self.call_zapier_webhook(
            webhook_url=self.zapier_mcp_url,
            action=tool["zapier_name"],
            parameters=parameters,
            api_key=self.zapier_api_key
        )
        
        logger.info(f"[EMAIL_AGENT] Zapier result status: {result.get('status')}")
        logger.info(f"[EMAIL_AGENT] ========== EXECUTE_TOOL END ==========")
        
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
            logger.error(f"[EMAIL_AGENT] Tool '{tool_name}' failed: {error_msg}")
            return self._create_result(
                status="failure",
                error=f"Tool '{tool_name}' failed: {error_msg}"
            )

    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors gracefully"""
        logger.error(f"[EMAIL_AGENT] Error handler: {error}", exc_info=True)
        return self._create_result(
            status="failure",
            error=f"Email agent error: {str(error)}"
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
