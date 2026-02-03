# ================================================================
# 🔄 CONTEXT ENRICHMENT - UPDATED WITH TASK_RESPONSE
# Includes task_response in LLM prompt for better context
# ================================================================
import re
import time
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from app.database.connection import SessionLocal
from app.services.context_db_service import get_context_db_service
from app.config import get_settings
logger = logging.getLogger(__name__)

@dataclass
class TaskWithContext:
    """Task with context needs"""
    need_context: bool
    action: str
    search_words: Optional[str] = None
    query: Optional[str] = None
    missing_params: List[str] = None
    instruction: Optional[str] = None
    target_agent: Optional[str] = None


class ContextEnrichment:
    """
    2-Tier Context Enrichment:
    
    TIER 2: ChromaDB Search
    - Use search_words to search ChromaDB
    - If found sufficient context → Generate instructions (need_context: false)
    - If NOT found → Keep as need_context: true with target_agent
    
    TIER 2: Agent Context Request (for future implementation)
    - Request context from target agent
    """
    
    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.model = "gpt-oss:20b"
        self.persist_directory = "./data/chroma"  
        try:
            # Use same client configuration as your app
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            print(f"✅ Connected to ChromaDB at: {self.persist_directory}")
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            raise
    
    async def enrich_tasks_with_context(
        self,
        user_input: str,
        task_response: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        
        
        enriched_tasks = []
        
        for task in task_response:
            logger.info(f"[CONTEXT_ENRICHMENT] Processing task: {task.get('action')}")
            
            # Skip if doesn't need context
            # Check if task was enriched (has instruction)
            if enriched_tasks and len(enriched_tasks) > 0:
                enriched_task = enriched_tasks
                
                # Handle nested list structure
                # enriched_tasks might be [[{...}]] instead of [{...}]
                if isinstance(enriched_task, list) and len(enriched_task) > 0:
                    enriched_task = enriched_task
                
                # Now check if it's a dict
                if isinstance(enriched_task, dict):
                    if not enriched_task.get("need_context", False):
                        # ✅ Context found - instruction generated
                        instruction = enriched_task.get("instruction")
                        logger.info(f"[CONTEXT_ENRICHMENT] ✅ Generated instruction: {instruction}")
                        return (True, instruction, None)
            
            # TIER 2: Search ChromaDB using search_words
            search_words = task.get("search_words")
            
            if search_words:
                logger.info(f"[CONTEXT_ENRICHMENT] TIER 2 : Searching ChromaDB for: {search_words}")
                
                collection = self.client.get_collection(name="agent_contexts")
    
                # Perform search
                chromadb_results = collection.query(
                    query_texts=search_words,
                    # agent_id=agent_id,
                    n_results=5,
                    include=["documents", "metadatas", "distances"]
                )
                
                if chromadb_results:
                    logger.info(f"[CONTEXT_ENRICHMENT] TIER 2 Found {len(chromadb_results)} results in ChromaDB")
                    logger.info(f"[CONTEXT_ENRICHMENT] TIER 2 Found:  {chromadb_results}")
                    
                    # Try to generate instructions from ChromaDB results
                    enriched_task = await self._generate_instruction_from_chromadb(
                        user_input=user_input,
                        task=task,
                        chromadb_results=chromadb_results
                    )
                    
                    if enriched_task and not enriched_task.get("need_context"):
                        enriched_tasks.append(enriched_task)
                        continue
                    
                    # ===== NEW TIER 3 LOGIC =====
                    target_agent = enriched_task.get("target_agent")
                    requester_agent = enriched_task.get("agent")
                    
                    if target_agent:
                        logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Initiating agent-to-agent request")
                        
                        # Request context from agent
                        agent_context_response = await self._request_agent_context(
                            target_agent=target_agent,
                            requester_agent = requester_agent,
                            query=task.get("query"),
                            max_hops=2,
                            current_hop=1
                        )
                        
                        # Process response
                        final_result = await self._generate_instruction_from_response_context(
                            original_task_spec=enriched_task,
                            agent_response=agent_context_response,
                            user_input=user_input
                        )
                        
                        enriched_tasks.append(final_result)
            
            return enriched_tasks
        

# ================================================================
#                     TIER 2 IMPLEMENTATION 
# ================================================================
            
    
    async def _generate_instruction_from_chromadb(
        self,
        user_input: str,
        task: Dict[str, Any],
        chromadb_results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to generate instruction from ChromaDB context
        
        NEW: Includes task_response in the prompt for better context understanding
        
        Returns task with instruction if successful, None if insufficient context
        """
        
        try:
            # Format ChromaDB results for LLM
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 2 Retrived ChromaDB results: {chromadb_results}")
            context_text = self._format_chromadb_results(chromadb_results)
            
            action = task.get("action")
            missing_params = task.get("missing_params", [])
            
            # NEW: Format task_response if available
            task_response_text = ""
            if task:
                task_response_text = f"""
AGENT RESPONSE:
{json.dumps(task, indent=2)}

Task Response Details:
- search_words: {task.get('search_words', 'N/A')}
- query: {task.get('query', 'N/A')}
- missing_params: {task.get('missing_params', [])}
- action: {task.get('action', 'N/A')}
"""
            
            # Create prompt for LLM to generate instruction
            prompt = f"""You are a context fulfilling assistant at TIER 2 in a multi-agent system. There are 3 main agents in the system Email Agent, Calendar Agent and Tasks Agent from who you get the tasks to be performed that had insufficient Context. Your job is to generate a clear instruction for the action required using the context retrieved from ChromaDB.

USER INPUT: {user_input}

ACTION REQUIRED: {action}

PREVIOUS AGENT RESPONSE: {task_response_text}

CHROMADB CONTEXT (retrieved information):
{context_text}

MISSING PARAMETERS: {', '.join(missing_params) if missing_params else 'None'}

YOUR JOB:
1. Analyze the user input, previous agent response, and ChromaDB context
2. By analyzing the previous agent response you can understand what context is missing in user input that caused the action to be incomplete. 
3. Check if ChromaDB context contains sufficient information to fulfill the action's missing parameters/context.
4. If YES → Generate a clear instruction that includes ALL necessary parameters from the context

IMPORTANT NOTE: Few create task actions of calendar agent will need context that is from email agent so consider contexts of email agent while analyzing the context for calendar agent.
    Example:
        The chromadb will have the context related to the Board Meeting which was udated into chromadb by the email agent since the user received a mail saying there is a Board Meeting which will have all the information of that event.
        now the asks calendar agent to create Board Meeting Event but the calendar agent doesnt know the timings of the Meeting, so due to insufficient context that task ill come to you with query to get the meeting timings, hence you have to use the Board Meeting context from email agent among the retrived results and generate the instructions with meeting timings.

    The above exapmple will apply to all other scenarios also act accordingly.

For action '{action}':
- MUST include all required parameters in the instruction
- Extract specific values from ChromaDB context (task list names, task titles, etc.)
- Make instruction specific and actionable
- Use exact names from context, not generic placeholders

RESPOND WITH ONLY JSON (no explanation):
If context is sufficient:
[
{{"need_context": false, "action": "{action}", "instruction": "Generated instruction here with all parameters from context"}}
]

IMPORTANT RULES:
- Use exact names from ChromaDB results
- Include all parameters needed for the action
- Be specific (e.g., "Mark 'Interview Preparation' in 'My Tasks' as completed")
- Do not include placeholder parameters
- Remember: This instruction will be sent to an external API (Zapier), so it must be complete and unambiguous

Example 1: 
    User_Input: Mark the Interview Preparation task as copleted or Find the Interview Preparation task and tell its status
    -For a Previous Agent Response like:
    [
    {{"need_context": true, "agent": "tasks", "action": "update_task", "search_words": "Interview Preparation task", "query": "task list name of task named Interview Preparation", "missing_params": ["task_list_name"]}}
    ]
    Here the missing context is 'task_list_name'. If ChromaDB retrieved results contains context of the task list name, generate instruction like:
    [
    {{"need_context": false, "agent": "tasks", "action": "update_task", "instruction": "Mark 'Interview Preparation' in 'My Tasks' as completed"}}
    ]
    If context is NOT sufficient in chromadb also:
    Then return a response like below to directly request context from the related agent by specifying the target agent(agent which may have required context)
    [
    {{"need_context": true, "agent": "tasks", "action": "update_task", "target_agent": "tasks", "query": "task list name of task named Interview Preparation", "missing_params": ["task_list_name"]}}
    ]


Example 2:
    User_Input: Send send a remainder mail to the guests or attendees of the Team Meeting event to attend the meeting without fail
    -For a Previous Agent Response like:
    [
    {{"need_context": true, "agent": "email", "action": "send_email", "search_words": "Team Meeting", "query": "find the mail id of attendees or guests for the Team Meeting Event", "missing_params": ["attendees_mail_id"]}}
    ]
    Here the missing context is 'attendees or event guests mail id'. If Agent retrieved results contains context of the event timings, generate instruction like:
    [  {{"need_context": false, "agent": "email" "action": "send_email", "instruction": "Send send a remainder mail to abc@gmail.com and john@gmail.com with subject and body remainding about to attend the Team Meeting Event"}}
    If context is NOT sufficient in chromadb also:
    Then return a response like below to directly request context from the related agent by specifying the target agent(agent which may have required context)
    [
    {{"need_context": true, "agent": "email" "action": "send_email", "target_agent": "calendar", "query": "find the mail id of attendees or guests for the Team Meeting Event", "missing_params": ["attendees_mail_id"]}}
    ]
    
Examplle 3:
    User_Input: Send a friendly mail to Vishnu K or Send a Interview remainder mail to Vishnu K
    -For a Previous Agent Response like:
    [
    {{"need_context": true, "agent": "email", "action": "send_email", "search_words": "Vishnu K", "query": "find the email_id of Vishnu K", "missing_params": ["recipient_email"]}}
    ]
    Here the missing context is 'recipient_email'. If ChromaDB retrieved results contains context of the recipient email, generate instruction like:
    [
    {{"need_context": false, "agent": "email", "action": "send_email", "instruction": "Send an email to Vishnu K(vishnu3sgk08@gmail.com) with subject Greetings and body Hope you are doing well"}}
    ]
    If context is NOT sufficient in chromadb also:
    Then return a response like below to directly request context from the related agent by specifying the target agent(agent which may have required context)
    [
    {{"need_context": true, "agent": "email", "action": "send_email", "target_agent": "email", "query": "find the email_id of Vishnu K", "missing_params": ["recipient_email"]}}
    ]

    
    Important: For all the send_email action where mail id of quests or attendees of an event is missing then set calendar as target_agent
               Only if email id of a person or any other is missing where is no event involved then set target_agent as email 
        
    The above examples are just for your understanding, perform a deep analysis on user request, query, missing_parameters, previous agent response and context retrieved then generate a appropriate output
"""

# If context is NOT sufficient:
# Respond by including the agent name(email, calendar and tasks agents) from which the relavant context can be obtained, like:
# [
# {{"need_context": true, "action": "update_task", "target_agent": "tasks", "query": "task list name of task named Interview Preparation", "missing_params": ["task_list_name"]}}
# ]
            
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 2 Calling LLM to generate instruction...")
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 2 Including task_response in prompt")
            
            response = await self.llm_service.call_ollama(
                model_name=self.model,
                prompt=prompt
            )
            
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 2 LLM Response: {response[:200]}...")
            
            # Parse response
            llm_response = response.strip()
            
            # Check if response is null
            if llm_response.lower() == "null":
                logger.info(f"[CONTEXT_ENRICHMENT] TIER 2 LLM determined insufficient context")
                return None
            
            # Try to parse as JSON
            try:
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
                if json_match:
                    llm_response = json_match.group(0)

                # Parse JSON
                parsed_response = json.loads(llm_response)

                return parsed_response[0]  # Return the first (and only) task dict
            
            except json.JSONDecodeError:
                logger.warning(f"[CONTEXT_ENRICHMENT] TIER 2 Failed to parse LLM response as JSON")
                return None
            
        except Exception as e:
            logger.error(f"[CONTEXT_ENRICHMENT] TIER 2 Error generating instruction: {e}", exc_info=True)
            return None
        
        return None
    
    def _format_chromadb_results(self, chromadb_results: Dict[str, Any]) -> str:
        
        if not chromadb_results:
            return "No relevant context found in ChromaDB."
        
        # Extract arrays from ChromaDB result
        documents_list = chromadb_results.get('documents', [])
        metadatas_list = chromadb_results.get('metadatas', [])
        distances_list = chromadb_results.get('distances', [])
        
        if not documents_list or len(documents_list) == 0:
            return "No relevant context found in ChromaDB."
        
        # FIX: CORRECTLY flatten by checking first element type
        # If first element is a list → we have nested structure
        if isinstance(documents_list, list):
            documents = documents_list      # Flatten: [['doc']] → ['doc']
        else:
            documents = documents_list         # Already flat
        
        # Same for metadata
        if metadatas_list and isinstance(metadatas_list, list):
            metadatas = metadatas_list      # Flatten: [[{...}]] → [{...}]
        else:
            metadatas = metadatas_list
        
        # Same for distances
        if distances_list and isinstance(distances_list, list):
            distances = distances_list      # Flatten: [[0.15]] → [0.15]
        else:
            distances = distances_list
        
        if not documents or len(documents) == 0:
            return "No relevant context found in ChromaDB."
        
        context_parts = []
        
        # Iterate through results (NOW everything is flat!)
        for i, doc in enumerate(documents):
            # doc is now a STRING
            if isinstance(doc, str):
                context_parts.append(f"- {doc}")
            else:
                context_parts.append(f"- {str(doc)}")
            
            # Add metadata (NOW is a DICT)
            if i < len(metadatas):
                metadata = metadatas[i]        # Now: {...} (dict)
                if isinstance(metadata, dict):
                    agent_type = metadata.get('agent_type', 'unknown')
                    timestamp = metadata.get('timestamp', '')
                    if timestamp:
                        context_parts.append(f"  (Agent: {agent_type}, Time: {timestamp})")
                    else:
                        context_parts.append(f"  (Agent: {agent_type})")
            
            # Add distance (NOW is a FLOAT)
            if i < len(distances):
                distance = distances[i]        # Now: 0.15 (float)
                if isinstance(distance, (int, float)):
                    relevance = 1 - distance
                    context_parts.append(f"  Relevance: {relevance:.2%}")
        
        return "\n".join(context_parts)




# ================================================================
#                   TIER 3 IMPLEMENTATION 
# ================================================================


        
    # async def _request_agent_context(
    #     self,
    #     target_agent: str,
    #     query: str,
    #     max_hops: int = 2,
    #     current_hop: int = 1,
    #     zapier_webhook_url: str = get_settings().ZAPIER_MCP_SERVER_URL
    # ) -> Dict[str, Any]:
        
        
    #     logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Requesting context from {target_agent}")
    #     logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Query: {query}")
    #     logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Hop: {current_hop}/{max_hops}")
        
    #     # ✅ SAFETY CHECK 1: Prevent infinite loops (circular dependencies)
    #     if current_hop >= max_hops:
    #         logger.warning(f"[CONTEXT_ENRICHMENT] TIER 3 ⚠️  Max hops exceeded (circular dependency detected)")
    #         return {
    #             "status": "error",
    #             "agent": target_agent,
    #             "error": "Max hops exceeded - circular dependency detected",
    #             "hop_count": current_hop
    #         }
        
    #     from app.agents.email_agent import EmailAgent
    #     from app.agents.calendar_agent import CalendarAgent
    #     from app.agents.tasks_agent import TasksAgent
    #     # ✅ STEP 1: Map agent 
    #     target_agent_classes = {
    #         "email": EmailAgent(),
    #         "calendar": CalendarAgent(),
    #         "tasks": TasksAgent()
    #     }
        
    #     target_agent_class = target_agent_classes.get(target_agent)
        
    #     if not target_agent_class:
    #         logger.error(f"[CONTEXT_ENRICHMENT] TIER 3 ❌ Unknown agent: {target_agent}")
    #         return {
    #             "status": "error",
    #             "agent": target_agent,
    #             "error": f"Unknown agent type: {target_agent}",
    #             "hop_count": current_hop
    #         }
        
    #     # ✅ STEP 2: Call the tool
    #     result = await target_agent_class.request_context(query)
        
    #     # ✅ STEP 3: Return formatted response
    #     if result.get("status") == "success":
    #         logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 ✅ Successfully retrieved from {target_agent}")
            
    #         return {
    #             "status": "success",
    #             "agent": target_agent,
    #             "data": result.get("data", {}),
    #             "hop_count": current_hop,
    #             "query": query
    #         }
        
    #     else:
    #         logger.error(f"[CONTEXT_ENRICHMENT] TIER 3 ❌ Failed to retrieve from {target_agent}")
            
    #         return {
    #             "status": "error",
    #             "agent": target_agent,
    #             "error": result.get("error", "Unknown error"),
    #             "hop_count": current_hop
    #         }
        


    
    async def _request_agent_context(
        self,
        target_agent: str,
        requester_agent: str,
        query: str,
        max_hops: int = 2,
        current_hop: int = 1,
        zapier_webhook_url: str = get_settings().ZAPIER_MCP_SERVER_URL,
    ) -> Dict[str, Any]:
        """
        Request context from a target agent (email/calendar/tasks),
        and log the request/response in context_requests table (Tier 3).
        """
        if not zapier_webhook_url:
            zapier_webhook_url = get_settings().ZAPIER_MCP_SERVER_URL

        logger.info(
            f"[CONTEXT_ENRICHMENT] TIER 3 Requesting context from {target_agent}"
        )
        logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Query: {query}")
        logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Hop: {current_hop}/{max_hops}")

        # safety: prevent infinite recursion
        if current_hop >= max_hops:
            logger.warning(
                "[CONTEXT_ENRICHMENT] TIER 3 Max hops exceeded (circular dependency)"
            )
            return {
                "status": "error",
                "agent": target_agent,
                "error": "Max hops exceeded - circular dependency detected",
                "hop_count": current_hop,
            }

        # # 1) log request start
        # start_time = time.time()
        # request_id = await self._log_context_request_start(
        #     target_agent=target_agent,
        #     query=query,
        #     current_hop=current_hop,
        # )


        db = SessionLocal()
        
        context_db_service = get_context_db_service(db_session=db)

        
        # Store in SQLite with link to ChromaDB
        start_time = time.time()
        request_id = context_db_service._log_context_request_start(
        target_agent=target_agent,
        requester_agent = requester_agent,
        query=query,
        )
            

        # 2) resolve target agent class
        from app.agents.email_agent import EmailAgent
        from app.agents.calendar_agent import CalendarAgent
        from app.agents.tasks_agent import TasksAgent

        target_agent_classes = {
            "email": EmailAgent(),
            "calendar": CalendarAgent(),
            "tasks": TasksAgent(),
        }

        target_agent_class = target_agent_classes.get(target_agent)

        if not target_agent_class:
            logger.error(
                f"[CONTEXT_ENRICHMENT] TIER 3 Unknown agent: {target_agent}"
            )

            # log failure
            context_db_service._log_context_request_completion(
                request_id=request_id,
                agent_response={"status": "failure"},
                agent_name=target_agent,
                start_time=start_time,
            )

            return {
                "status": "error",
                "agent": target_agent,
                "error": f"Unknown agent type: {target_agent}",
                "hop_count": current_hop,
                "request_id": request_id,
            }

        logger.info(
            f"[CONTEXT_ENRICHMENT] TIER 3 Calling target agent {target_agent}"
        )

        # 3) call the target agent’s request_context
        result = await target_agent_class.request_context(query)

        # 4) log completion (success or failure)
        context_db_service._log_context_request_completion(
            request_id=request_id,
            agent_response=result,
            agent_name=target_agent,
            start_time=start_time,
        )

        db.close()

        # 5) normalize return shape for the rest of your pipeline
        if result.get("status") == "success":
            logger.info(
                f"[CONTEXT_ENRICHMENT] TIER 3 ✅ Successfully retrieved from {target_agent}"
            )
            return {
                "status": "success",
                "agent": target_agent,
                "data": result.get("data", {}),
                "hop_count": current_hop,
                "query": query,
                "request_id": request_id,
            }

        logger.error(
            f"[CONTEXT_ENRICHMENT] TIER 3 ❌ Failed to retrieve from {target_agent}"
        )
        return {
            "status": "error",
            "agent": target_agent,
            "error": result.get("error", "Unknown error"),
            "hop_count": current_hop,
            "request_id": request_id,
        }





    async def _generate_instruction_from_response_context(
        self,
        original_task_spec: Dict[str, Any],
        agent_response: Dict[str, Any],
        user_input: str
    ) -> Dict[str, Any]:
        
        try:
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Processing agent response for instruction generation")
            
            action = original_task_spec.get("action")
            target_agent = agent_response.get("agent", "unknown")
            
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Action: {action}")
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Agent response status: {agent_response.get('status')}")
            
            # ✅ CHECK: Agent response was successful?
            if agent_response.get("status") != "success":
                logger.warning(f"[CONTEXT_ENRICHMENT] TIER 3 Agent response failed: {agent_response.get('error')}")
                
                # Return error message for user clarification
                error_msg = f"""I searched the {target_agent} agent for: "{original_task_spec.get('query', 'N/A')}"
                
    But couldn't find the necessary information. The user request: "{user_input}" is missing: 
    - {', '.join(original_task_spec.get('missing_params', []))}

    Please provide more details about {', '.join(original_task_spec.get('missing_params', []))} for the {target_agent} agent."""
                
                logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Returning error message")
                
                return {
                    "need_context": True,
                    "action": action,
                    "error_msg": error_msg.strip(),
                    "missing_params": original_task_spec.get("missing_params", []),
                    "tier3_failed": True,
                    "failure_reason": agent_response.get("error")
                }
            
            # ✅ STEP 1: Agent response successful - format it for LLM
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Formatting agent response for LLM")
            
            agent_response_text = self._format_agent_response_for_llm(
                agent_response=agent_response,
                agent_name=target_agent
            )
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 agent context response: {agent_response_text}")
            
            # ✅ STEP 3: Create LLM prompt to synthesize both sources
            prompt = f"""You are a context fulfilling assistant at TIER 3 in a multi-agent system. There are 3 main agents in the system Email Agent, Calendar Agent and Tasks Agent from who you get the tasks to be performed that had insufficient Context only if TIER 2 also did not had enough context. Your job is to generate a clear instruction for the action required using the context retrieved from requested agent.

    The TIER 2 is Chromadb context assistant which processes tasks from agents and genrates instructions using the contexts in chromadb, if the chromadb also did not had enough context then the tasks are given to you to generate instructions using the context given the related agent.

    USER REQUEST: {user_input}

    TIER 2 assistant response: {original_task_spec}

    TARGET AGENT RESPONSE FOR CONTEXT REQUEST:
    {agent_response_text}


    YOUR JOB:
    1. Analyze agent response
    2. Extract values that fill the missing parameters
    3. If you have enough info → Generate complete instruction
    4. If still insufficient → Return error message asking for clarification

    IMPORTANT RULES:
    - Use EXACT values from agent response (task names, email addresses, dates, times, etc.)
    - Include ALL required parameters in the instruction
    - Be specific and unambiguous (e.g., "Mark 'Interview Preparation' in 'My Tasks' as completed")
    - This instruction will be sent to Zapier, so it must be complete
    - Do NOT use generic placeholder values

    RESPOND WITH ONLY JSON (no markdown, no explanation):

    Example 1: 
    Actual User_Input: Mark the Interview Preparation task as copleted or Find the Interview Preparation task and tell its status
    -For a Previous Agent Response like:
    [
    {{"need_context": true, "agent": "tasks", "action": "update_task", "target_agent": "tasks", "query": "task list name of task named Interview Preparation", "missing_params": ["task_list_name"]}}
    ]
    Here the missing context is 'task_list_name'. If Agent retrieved results contains context of the task list name, generate instruction like:
    [
    {{"need_context": false, "agent": "tasks", "action": "update_task", "instruction": "Mark 'Interview Preparation' in 'My Tasks' as completed"}}
    ]
    If context is NOT sufficient in Agent response also:
    Then return a error message asking tthe user to provide the missing context like: 
    [
    {{"need_context": true, "agent": "tasks", "action": "update_task", "error_msg": "This user request: {user_input} doest have enough context. Missing <missing parameters> related to <task mentioned in user input>", "missing_params": ["task_list_name"]}}
    ]

Example 2:
    Actual User_Input: Send send a remainder mail to the guests or attendees of the Team Meeting event to attend the meeting without fail
    -For a Previous Agent Response like:
    [
    {{"need_context": true, "agent": "email", "action": "send_email", "target_agent": "calendar", "query": "find the mail id of attendees or guests for the Team Meeting Event", "missing_params": ["attendees_mail_id"]}}
    ]
    Here the missing context is 'attendees or event guests mail id'. If Agent retrieved results contains context of the event timings, generate instruction like:
    [  {{"need_context": false, "agent": "email", "action": "send_email", "instruction": "Send a remainder mail to abc@gmail.com and john@gmail.com with subject and body remainding about to attend the Team Meeting Event"}}
    If context is NOT sufficient in Agent response also:
    Then return a error message asking tthe user to provide the missing context like: 
    [
    {{"need_context": true, "agent": "email", "action": "send_email",  "error_msg": "This user request: {user_input} doest have enough context. Missing <missing parameters> related to <event mentioned in user input>", "missing_params": ["attendees_mail_id"]}}
    ]
    
Example 3:
    Actual User_Input: Send a friendly mail to Vishnu K or Send a Interview remainder mail to Vishnu K
    -For a Previous Agent Response like:
    [
    {{"need_context": true, "agent": "email", "action": "send_email", "target_agent": "email", "query": "find the email_id of Vishnu K", "missing_params": ["recipient_email"]}}
    ]
    Here the missing context is 'recipient_email'. If Agent retrieved results contains context of the recipient email, generate instruction like:
    [
    {{"need_context": false, "agent": "email", "action": "send_email", "instruction": "Send an email to Vishnu K(vishnu3sgk08@gmail.com) with subject Greetings and body Hope you are doing well"}}
    ]
    If context is NOT sufficient in Agent response also:
    Then return a error message asking the user to provide the missing context like: 
    [
    {{"need_context": true, "agent": "email", "action": "send_email",  "error_msg": "This user request: {user_input} doest have enough context. Missing <missing parameters> related to <email recipient mentioned in user input>", "missing_params": ["recipient_email"]}}
    ]

    The above examples are just for your understanding, perform a deep analysis on user request, query, missing_parameters, previous agent response and context retrieved then generate a appropriate output
 
    Remember: Use exact values from agent response, not generic placeholders.
    Important: In the target agent response mainly focus on resolved parmeters and include only the actual missing parameter into the instruction and don't include any id or other related info of the missing parameter
    EX: consider the missing parameter is task_list_name and if the target agent response contains resolved parameters like this:
    "resolvedParams": {{"list":"name":"List","label":"AgentSync","value":"Szh6ZjJyeXBWemgyTVRiYw","reason":"llm-guess","status":"guessed"}}
    then include "AgentSync" into instruction which is the actual missing parameter and dont use "Szh6ZjJyeXBWemgyTVRiYw" which is the id of AgentSync.
    
    """
            
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Calling LLM to synthesize responses")
            
            # ✅ STEP 4: Call LLM
            response = await self.llm_service.call_ollama(
                model_name=self.model,
                prompt=prompt
            )
            
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 LLM response: {response[:200]}...")
            
            # ✅ STEP 5: Parse LLM response
            response = response.strip()
            
            # Extract JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            parsed = json.loads(response)
            result = parsed[0]  # Get first (only) result
            
            logger.info(f"[CONTEXT_ENRICHMENT] TIER 3 Final result: need_context={result.get('need_context')}")
            
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"[CONTEXT_ENRICHMENT] TIER 3 Failed to parse LLM response: {e}")
            
            return {
                "need_context": True,
                "action": original_task_spec.get("action", "unknown"),
                "error_msg": f"System error: Could not process response. Please try again.",
                "missing_params": original_task_spec.get("missing_params", []),
                "processing_error": True
            }
        
        except Exception as e:
            logger.error(f"[CONTEXT_ENRICHMENT] TIER 3 Error generating instruction: {e}", exc_info=True)
            
            return {
                "need_context": True,
                "action": original_task_spec.get("action", "unknown"),
                "error_msg": f"System error: {str(e)}",
                "missing_params": original_task_spec.get("missing_params", []),
                "processing_error": True
            }



    def _format_agent_response_for_llm(
        self,
        agent_response: Dict[str, Any],
        agent_name: str
    ) -> str:
        """
        Format agent response data for LLM context
        
        Takes the tool response and formats it nicely for the LLM prompt
        
        Args:
            agent_response: Response from _request_agent_context
            agent_name: "email", "calendar", or "tasks"
        
        Returns:
            Formatted string for LLM
        """
        
        data = agent_response.get("data", {})
        tool_used = agent_response.get("tool_used", "unknown")
        
        if agent_name == "email":
            formatted = f"""Email Search Results (from {tool_used}):
    - Found emails related to the query
    - Data: {json.dumps(data, indent=2)}
    - Tool: {tool_used}
    - Status: {agent_response.get('status')}"""
        
        elif agent_name == "calendar":
            formatted = f"""Calendar Search Results (from {tool_used}):
    - Found events related to the query
    - Data: {json.dumps(data, indent=2)}
    - Tool: {tool_used}
    - Status: {agent_response.get('status')}"""
        
        elif agent_name == "tasks":
            formatted = f"""Tasks Search Results (from {tool_used}):
    - Found tasks/task lists related to the query
    - Data: {json.dumps(data, indent=2)}
    - Tool: {tool_used}
    - Status: {agent_response.get('status')}"""
        
        else:
            formatted = f"""Agent Response (from {tool_used}):
    {json.dumps(agent_response, indent=2)}"""
        
        return formatted