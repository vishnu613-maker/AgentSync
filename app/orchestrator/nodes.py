"""
FIXED: nodes.py with Dict Access - All state["key"] format

✅ FIXED: All state.field → state["field"]
✅ FIXED: All state.list.append() → state["list"].append()
✅ FIXED: All nested dict access corrected
✅ WORKS: With LangGraph dict state objects
"""

import json
import asyncio
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging
import uuid
import re

from .state import OrchestratorState, AgentTask
from app.services.message_queue import MessageQueueService
from app.services.llm_service import LLMService
from app.config import get_settings
from app.agents.registry import agent_registry


# from app.orchestrator.context_analyzer import get_context_analyzer
# from app.services.context_retrieval_service import get_context_retrieval_service


logger = logging.getLogger(__name__)

def encode_to_toon(data: Any) -> str:
    """
    Pure Python TOON encoder - no external dependencies.
    Converts JSON data to TOON format for efficient LLM processing.
    """
    result = []
    
    if isinstance(data, list):
        # Handle array of objects
        if len(data) > 0 and all(isinstance(item, dict) for item in data):
            # Get field names from first object
            first_obj = data[0]
            fields = list(first_obj.keys())
            field_names = ",".join(fields)
            
            # Build TOON array format
            result.append(f"results[{len(data)}]{{{field_names}}}:")
            
            # Add each row
            for obj in data:
                row_values = []
                for field in fields:
                    val = obj.get(field, "")
                    # Convert to string representation
                    if isinstance(val, bool):
                        row_values.append(str(val).lower())
                    elif isinstance(val, dict) or isinstance(val, list):
                        # Nested structures - keep as minimal JSON
                        row_values.append(json.dumps(val, separators=(',', ':')))
                    elif val is None:
                        row_values.append("")
                    else:
                        # Escape commas in values
                        str_val = str(val).replace(',', '\\,')
                        row_values.append(str_val)
                result.append("  " + ",".join(row_values))
        else:
            # Fallback to JSON for non-uniform arrays
            return json.dumps(data, separators=(',', ':'))
    
    elif isinstance(data, dict):
        # Handle single object or nested structure
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                if all(isinstance(item, dict) for item in value):
                    # Recursive call for nested arrays
                    toon_array = encode_to_toon(value)
                    result.append(f"{key}:")
                    for line in toon_array.split('\n'):
                        result.append("  " + line)
                else:
                    result.append(f"{key}: {json.dumps(value, separators=(',', ':'))}")
            elif isinstance(value, dict):
                result.append(f"{key}:")
                nested = encode_to_toon(value)
                for line in nested.split('\n'):
                    result.append("  " + line)
            else:
                result.append(f"{key}: {value}")
    
    else:
        # Fallback for primitives
        return str(data)
    
    return "\n".join(result)



def store_context_to_db(
        state: OrchestratorState,
        llm_response: str,
        agent_type: str
    ) -> None:
        """
        Store context in ChromaDB and SQLite after task execution
        
        Args:
            state: Current orchestrator state
            llm_response: LLM-generated response with CONTEXT_SUMMARY section
            agent_type: Type of agent that executed task (email/calendar/tasks)
        """
        try:
            from app.services.chroma_service import get_chroma_service
            from app.services.context_db_service import get_context_db_service
            from app.database.connection import SessionLocal
            
            logger.info(f"[STORE_CONTEXT] Storing context for {agent_type} agent")
            
            # Extract CONTEXT_SUMMARY section from LLM response
            # Try multiple patterns
            context_match = re.search(
                r'\*\*CONTEXT_SUMMARY:\*\*\s*\n(.*?)(?=\n\n|\Z)',
                llm_response,
                re.DOTALL
            )
            
            if not context_match:
                # Try alternative pattern without asterisks
                context_match = re.search(
                    r'CONTEXT_SUMMARY:\s*\n(.*?)(?=\n\n|\Z)',
                    llm_response,
                    re.DOTALL
                )
            
            if not context_match:
                logger.warning("[STORE_CONTEXT] No CONTEXT_SUMMARY found in LLM response")
                # Fallback: use entire response as context
                context_summary = llm_response[:500]  # Limit length
            else:
                context_summary = context_match.group(1).strip()
            
            logger.debug(f"[STORE_CONTEXT] Context summary: {context_summary[:100]}...")
            
            # Get agent ID from agent registry
            from app.agents.registry import agent_registry
            agent = agent_registry.get_agent(agent_type)
            agent_id = agent.agent_id if agent else 1
            
            logger.info(f"[STORE_CONTEXT] Using agent_id: {agent_id} for agent_type: {agent_type}")
            
            # Build metadata
            metadata = {
                "agent_type": agent_type,
                "user_input": state.get("user_input", "")[:200],  # Limit length
                "timestamp": datetime.utcnow().isoformat(),
                "task_count": len(state.get("task_results", [])),
                "execution_id": state.get("execution_id", "unknown"),
                "detected_agents": ",".join(state.get("detected_agents", []))
            }
            
            # Initialize services
            chroma_service = get_chroma_service()
            
            # Create SQLite session
            db = SessionLocal()
            try:
                context_db_service = get_context_db_service(db_session=db)
                
                # Store in ChromaDB (generates embeddings automatically)
                vector_id = chroma_service.add_context(
                    agent_id=agent_id,
                    context_summary=context_summary,
                    metadata=metadata
                )
                
                logger.info(f"[STORE_CONTEXT] ✅ Stored in ChromaDB: {vector_id}")
                
                # Store in SQLite with link to ChromaDB
                db_context = context_db_service.insert_agent_context(
                    agent_id=agent_id,
                    context_summary=context_summary,
                    vector_id=vector_id,
                    metadata=metadata,
                    context_key=f"{agent_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                )
                
                if db_context:
                    logger.info(f"[STORE_CONTEXT] ✅ Stored in SQLite: context_id={db_context.id}")
                else:
                    logger.error("[STORE_CONTEXT] ❌ Failed to store in SQLite")
            
            finally:
                db.close()
            
        except Exception as e:
            logger.error(f"[STORE_CONTEXT] ❌ Error storing context: {e}", exc_info=True)
            # Don't fail the whole request if context storage fails


class OrchestratorNodes:
    """All nodes that execute within the LangGraph orchestrator"""

    def __init__(self, mq_service: MessageQueueService, llm_service: LLMService):
        self.mq_service = mq_service
        self.llm_service = llm_service
        self.model = "gpt-oss:20b"
        self.settings = get_settings()
        
        # self.context_analyzer = get_context_analyzer(llm_service)
        
        

    # ==================== ENHANCED ANALYZE_INTENT_NODE ====================
    # Location: nodes.py
    # Replace the entire analyze_intent_node method with this:

    async def analyze_intent_node(self, state: OrchestratorState) -> OrchestratorState:
        """NODE 1: Analyze user input to detect intent and required agents
        
        ✅ ENHANCED: Now detects friendly chat, context storage, or task execution
        """
        logger.info(f"[ORCHESTRATOR] Analyzing intent: {state['user_input']}")
        recent_history = state["conversation_history"][-5:] if state["conversation_history"] else []
        context = json.dumps(recent_history, indent=2)

        try:
            response = await self.llm_service.call_ollama(
                model_name=self.model,
                prompt=f"""
                You are a intelligent task analyzer and response generator wwho is part of a Multi Agents Context Management Platform which also contains other Agents like Email Agent, Tasks Agent and Calendar Agent.
                    Your work is to analyze the user request and determine your response type

    DECISION TREE:

    Step 1: Is this JUST a greeting/friendly chat (NO task execution)?
    - Keywords: "hello", "hi", "how are you", "what can you do", "help me", "who are you", "thanks"
    - Pattern: User asking about YOU, not asking to DO something
    - Response type: "friendly_chat"
    - Example: "Hi AgentSync, what can you do?" → friendly_chat

    Step 2: Is user asking to STORE/REMEMBER information (NOT task execution)?
    - Keywords: "store", "save", "remember", "note down", "i want to store", "save this context", "keep this", "record this"
    - Pattern: User wants to SAVE information for future reference, not execute an action
    - Response type: "store_context"
    - Extract: ALL relevant details from user input as context
    - Example: "Store this email about board meeting..." → store_context

    Step 3: Otherwise it's a TASK REQUEST
    - Pattern: User wants to DO something (send, find, create, etc.)
    - Response type: "run_task"
    - Detect agents needed: email, calendar, tasks

    Conversation context (last 5 messages):
    {context}

    User request: {state["user_input"]}

    Available agents:
    - email: Send/fetch/archive emails
    - calendar: Create/update/view calendar events
    - tasks: create task list/create/update/find tasks

    RESPONSE FORMATS:

    Format 1 - Friendly Chat:
    {{
        "run_task": false,
        "store_context": false,
        "friendly_chat": true,
        "response": "Your generated friendly response here"
    }}

    Format 2 - Context Storage:
    {{
        "run_task": false,
        "store_context": true,
        "friendly_chat": false,
        "context": "Comprehensive context about what to store - include all details",
        "agent_type": "email|calendar|tasks|general"
    }}

    Format 3 - Task Execution:
    {{
        "run_task": true,
        "store_context": false,
        "friendly_chat": false,
        "agents": ["email"]
    }}
    for multiple agents:
    {{
        "run_task": true,
        "store_context": false,
        "friendly_chat": false,
        "agents": ["email", "calendar"]
    }}

    CRITICAL REQUIREMENTS:
    1. Return ONLY valid JSON - no markdown, no comments
    2. Exactly ONE of run_task/store_context/friendly_chat must be true
    3. For friendly_chat: include "response" field
    4. For store_context: include "context" and "agent_type" fields
    5. For run_task: include "agents" array
    6. Pure JSON object only"""
            )

            response = response.strip()
            logger.info(f"[ORCHESTRATOR] LLM Raw Response: {response}")
            
            # Extract JSON with regex
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
            if not json_match:
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            analysis = json.loads(response)
            
            # ✅ ENHANCED: Parse enhanced response format
            state["run_task"] = analysis.get("run_task", True)
            state["store_context"] = analysis.get("store_context", False)
            state["friendly_chat"] = analysis.get("friendly_chat", False)
            state["response"] = analysis.get("response", "")
            state["context"] = analysis.get("context", "")
            state["detected_agents"] = analysis.get("agents", [])

            state["execution_trace"].append({
                "node": "analyze_intent",
                "timestamp": datetime.now().isoformat(),
                "run_task": state["run_task"],
                "store_context": state["store_context"],
                "friendly_chat": state["friendly_chat"],
                "agents": state["detected_agents"]
            })

            logger.info(f"[ORCHESTRATOR] ✅ Intent Analysis Complete:")
            logger.info(f"   run_task={state['run_task']}, store_context={state['store_context']}, friendly_chat={state['friendly_chat']}")

        except json.JSONDecodeError as e:
            logger.error(f"[ORCHESTRATOR] Failed to parse LLM response: {e}")
            state["execution_errors"]["intent_analysis"] = str(e)
            state["run_task"] = True
            state["store_context"] = False
            state["friendly_chat"] = False
            state["detected_agents"] = ["email"]

        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Error in analyze_intent: {e}", exc_info=True)
            state["execution_errors"]["intent_analysis"] = str(e)
            state["run_task"] = True
            state["store_context"] = False
            state["friendly_chat"] = False
            state["detected_agents"] = ["email"]

        return state

    
    
    
    
    async def _analyze_and_enrich_context(
        self,
        user_input: str,
        agent_name: str,
        task_response: Dict[str, Any],
        action: str,
        available_tools: List[Dict],
        state: OrchestratorState
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        
        
        logger.info(f"[CONTEXT_ENRICHMENT] Starting enrichment for {action}")
        
        
        from app.orchestrator.context_enrichment import ContextEnrichment
        
        enricher = ContextEnrichment(
            llm_service=self.llm_service
        )
        
        # # Get task_response from state
        # task_response_data = state.get("raw_task_response")

        # logger.info(f"[CONTEXT_ENRICHMENT] raw_task_response: {task_response_data}")
        
        # Enrich single task (TIER 1: ChromaDB)
        enriched_tasks = await enricher.enrich_tasks_with_context(
            user_input=user_input,
            task_response=[task_response],  
            # task_response=task_response_data
        )
        
        logger.info(f"[CONTEXT_ENRICHMENT] Enrichment complete")
        
        # Check if task was enriched (has instruction)
        if enriched_tasks and len(enriched_tasks) > 0:
            enriched_task = enriched_tasks
            
            
            if not enriched_task[0].get("need_context", True):
                instruction = enriched_task[0].get("instruction")
                logger.info(f"[CONTEXT_ENRICHMENT] ✅ Generated instruction: {instruction}")
                return (True, instruction, None)
            
            else:
                logger.info(f"[CONTEXT_ENRICHMENT] ❌ Context still insufficient after enrichment")
                error_msg = enriched_task[0].get("error_msg")
                return (False, None, error_msg)
        
        # ❌ Context insufficient
        logger.warning(f"[CONTEXT_ENRICHMENT] Insufficient context for {action}")
        error_msg = f"Could not find sufficient context for {action}. Missing: {task_response.get('missing_params', [])}"
        return (False, None, error_msg)


        
            
            

    async def create_tasks_node(self, state: OrchestratorState) -> OrchestratorState:
        """NODE 2: Create tasks with INSTRUCTIONS-BASED approach"""
        logger.info(f"[ORCHESTRATOR] Creating tasks for agents")

        state["task_metadata"]["task_id"] = str(uuid.uuid4())
        tasks: List[AgentTask] = []
        agent_tools_map = {}

        # Fetch tools for each agent
        for agent_name in state["detected_agents"]:
            tools_info = agent_registry.get_agent_tools(agent_name)
            if tools_info:
                agent_tools_map[agent_name] = tools_info

        logger.info(f"[ORCHESTRATOR] ✅ Fetched tools for {', '.join(state['detected_agents'])}")
        
        recent_history = state["conversation_history"][-5:] if state["conversation_history"] else []
        context = json.dumps(recent_history, indent=2)
        logger.info(f"[ORCHESTRATOR] Recent conversation context: {context}")

        # ✅ NEW: For each agent, generate separate tasks with instructions
        for agent_name in state["detected_agents"]:
            logger.info(f"[ORCHESTRATOR] Creating task for {agent_name}")

            available_tools = []
            if agent_name in agent_tools_map:
                tools = agent_tools_map[agent_name]["tools"]
                available_tools = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in tools])

            task_created = False

            # ✅ SPECIAL: For tasks agent, generate separate instructions for each action
            if agent_name == "tasks":
                
                tools = agent_tools_map.get(agent_name, {}).get("tools", [])
                # Generate instructions for tasks
                instructions_response = await self.llm_service.call_ollama(
                    model_name=self.model,
                    prompt=f"""
                    You are a tasks Agent which is part of a Multi Agents Context Management Platform which also contains other Agents like Email Agent and Calendar Agent.
                    Your work is to manage all the request related to Tasks, now analyze the user request and determine:

                As mentioned u are a part of multi agents system so sometimes u may get user inputs that will also include requests related to other agents, in this case just ignore them and just focus on the requests related to tasks.
                Example:
                User Input: Create a task named Exercise in My Tasks task list and send a interview reminder mail to Vishnu3sgk08@gmail.com just to say hello.
                In case if u get such kind of input then just focus on "Create a task named Exercise in My Tasks task list" and ignore the requets related to other agents.
                
                Analyze the user request and determine:
                1. What TASK ACTION the user wants (create, find, update, etc.)
                2. How many DISTINCT tasks the user is asking for
                3. For each task, check if sufficient context exists
                4. Generate instructions OR identify missing context for each task

                CRITICAL RULES:
                - Only generate tasks if user EXPLICITLY asks for multiple operations
                - Single queries like "get me all tasks" = 1 task
                - Multiple operations like "create X and update Y" = 2 tasks
                - Default to 1 task unless user clearly specifies multiple
                
                

                User Request: {state["user_input"]}

                Available Tools:
                {available_tools}

                TASK COUNT DETECTION:
                - Keywords for multiple tasks: "and", "also", "create...update", "plus", "additionally"
                - Keywords for single task: "get", "show", "list", "find", "check", "retrieve"
                
                ACTION IDENTIFICATION KEYWORDS:
                - "find_task" -> "intent_keywords": ["find", "search", "look for", "locate", "check task"]
                - "create_task_list" -> "intent_keywords": ["create list", "new list", "add list"]
                - "get_tasks_by_list" -> "intent_keywords": ["list", "show", "get all", "view tasks", "list tasks"]
                - "create_task" -> "intent_keywords": ["create task", "add task", "new task", "create"]
                - "update_task" -> "intent_keywords": ["update", "edit", "modify", "mark", "complete"]
                
                REQUIRED PARAMETERS by Action:
                - create_task: task_list_name (REQUIRED), task_title (REQUIRED), due_date (REQUIRED)
                - find_task: task_list_name (REQUIRED), task_title (REQUIRED)
                - create_task_list: list_title (REQUIRED)
                - update_task: task_list_name (REQUIRED), task_title (REQUIRED)
                - get_tasks_by_list: task_list_name (REQUIRED)
                
                Analyze the Conversation context also with the user request to determine if sufficient context exists to fulfill each task, where sometimes you can find parameters like task_list_name, task_name and others in the Conversation context that are missing in user request.
                If any required parameter that is missing in user input is found in conversational context them include that parameter in the instruction with out fail. 
                RESPONSE FORMAT Examples:
                The output should be in the below given JSON array format only.
                If ALL required parameters present:
                Examples:
                - Get me all the tasks present in AgentSync list.
                response:
                [
                {{"need_context": false, "action": "get_tasks_by_list", "instructions": "Get all tasks in AgentSync list"}}
                ]

                If missing required parameters but found in conversational context:
                Examples:
                - Mark Interview Preparation task as completed. (or) mark the task named Interview Preparation as completed.
                here in the above input the task list of the task named Interview Preparation is missing, if the conversational context has the information about the task list name of Interview Preparation task then include that in the instruction.

                response:
                [
                {{"need_context": false, "action": "update_task", "instructions": "Mark the Interview Preparation task in [TASK_LIST_NAME_FROM_CONTEXT] as completed"}}
                ]
                
                If missing required parameters(parameters also not found in conversational context):
                Examples:
                - Mark Interview Preparation task as completed. (or) mark the task named Interview Preparation as completed.
                response:
                [
                {{"need_context": true, "action": "update_task", "search_words": "Interview Preparation task", "query": "task list name of task named Interview Preparation", "missing_params": ["task_list_name"]}}
                ]
                
                - Get me all the tasks present in task list.
                response:
                [
                {{"need_context": true, "action": "get_tasks_by_list", "search_words": "task list", "query": "get all the tas list names and list all tasks in it", "missing_params": ["task_list_name"]}}
                ]
                
                IMPORTANT:
                - search_words: Keywords that would find this in ChromaDB (use user's phrasing)
                - query: What information we need to find
                - instruction: Only include if need_context=false
                - Keep all required params in search_words and query for context retrieval
                
                Return ONLY a JSON array with the detected number of tasks:
                For multiple tasks with sufficient context:
                Examples:
                - Create a task Major Project Completion in the AgentSync Task list with due on 12/12/2025 4pm and Find task Budget Review in the Finance Tasks list.
                response:
                [
                {{"need_context": false, "action": "create_task", "instructions": "Create a task Major Project Completion in the AgentSync Task list with due on 12/12/2025 4pm"}},
                {{"need_context": false, "action": "find_task", "instructions": "Find task Budget Review in the Finance Tasks list"}}
                ]
                For multiple tasks needing context:
                Examples:
                - Mark the task in AgentSync task list as completed and Get me all the tasks present in task list.
                response:
                [
                {{"need_context": true, "action": "update_task", "search_words": "AgentSync task list", "query": "get the latest task created in AgentSync task list and mark it as completed", "missing_params": ["task_name"]}}
                {{"need_context": true, "action": "get_tasks_by_list", "search_words": "task list", "query": "get all the tas list names and list all tasks in it", "missing_params": ["task_list_name"]}}
                ]
                
                Clarification:
                - Mark the DSA Assignment task in AgentSync task list as completed -> action: update_task, task_title: DSA Assignment and task_list_name: AgentSync
                response:
                [{{"need_context": false, "action": "update_task", "instructions": "Mark the DSA Assignment task in AgentSync task list as completed"}}]
                
                - Create a task Major Project Completion in the AgentSync Task list with due on 12/12/2025 4pm. -> action: create_task, task_title: Major Project Completion, task_list_name: AgentSync, due_date: 12/12/2025 4pm
                response:
                [{{"need_context": false, "action": "create_task", "instructions": "Create a task named Major Project Completion in AgentSync task list"}}]
                
                - Create a task named DSA Assignment in AgentSync task list and also mark the Interview Reminder task in My Tasks task list as completed
                for first part: action: create_task, task_title: DSA Assignment, task_list_name: AgentSync
                for second part: action: update_task, task_title: Interview Reminder, task_list_name: My Tasks
                response:
                [
                {{"need_context": false, "action": "create_task", "instructions": "Create a task named DSA Assignment in AgentSync task list"}},
                {{"need_context": false, "action": "update_task", "instructions": "Mark the Interview Reminder task in My Tasks task list as completed"}}
                ]
                
                Important:      
                Your Work is to only identify the Action need to be used based on the user request if all the context is avalable in the user input and add the instruction and then return the JSON object as shown in the examples.
                If any specific task list is mentioned in the user request then make sure to include that in the instructions.
                Example: create a task Major Project Completion in the AgentSync Task list with due on 12/12/2025 4pm.
                In the above given example "AgentSync" is the task list name which need to be included in the instructions.
                The response should only contain a JSON object without any extra text like Explaination or Note and others as given in the example and donot miss any commas, brackets and inverted commas.
                """
                )


                try:

                    instructions_response = instructions_response.strip()
                    logger.info(f"[ORCHESTRATOR] LLM Response: {instructions_response}")
                    json_match = re.search(r'\[.*\]', instructions_response, re.DOTALL)
                    if json_match:
                        instructions_response = json_match.group(0)
                    
                    tasks_list = json.loads(instructions_response)
                    
                    if not isinstance(tasks_list, list):
                        tasks_list = [tasks_list]
                    
                    logger.info(f"[ORCHESTRATOR] Generated {len(tasks_list)} task(s) with instructions")
                    
                    for task_spec in tasks_list:
                        action = task_spec.get("action", "create_task")
                        need_context = task_spec.get("need_context", False)
                        
                        if need_context:
                            logger.info(f"[ORCHESTRATOR] Task needs context, starting enrichment")
                            
                            has_context, enriched_instructions, error_msg = await self._analyze_and_enrich_context(
                                user_input=state["user_input"],
                                agent_name=agent_name,
                                task_response=task_spec,
                                action=action,
                                available_tools=tools,
                                state=state
                            )
                            
                            if has_context:
                                # Context enrichment successful
                                instructions = enriched_instructions
                                logger.info(f"[ORCHESTRATOR] ✅ Context enriched: {instructions}...")
                            else:
                                # Context still insufficient - store error and skip task
                                logger.warning(f"[ORCHESTRATOR] ❌ Insufficient context: {error_msg}")
                                state["execution_errors"][f"{agent_name}_{action}"] = error_msg
                                state["insufficient_context"] = True
                                state["has_critical_error"] = True
                                state["error_message"] = error_msg
                                
                                # Return immediately - don't create fallback!
                                return state
                        else:
                            # Sufficient context from initial analysis
                            instructions = task_spec.get("instructions", state["user_input"])
                        
                        # Validate action exists in tools
                        available_tool_names = [t["name"] for t in tools]
                        
                        if action not in available_tool_names:
                            logger.warning(f"[ORCHESTRATOR] Action '{action}' not in available tools")
                            action = available_tool_names[0] if available_tool_names else "create_task"
                        
                        # Create task with enriched instructions
                        parameters = {"instructions": instructions}
                        
                        task = AgentTask(
                            agent_name=agent_name,
                            action=action,
                            parameters=parameters,
                            task_id=f"{state['task_metadata']['task_id']}_{agent_name}_{len(tasks)}"
                        )
                        
                        tasks.append(task)
                        logger.info(f"[ORCHESTRATOR] ✅ Created task: {action}")
                        logger.info(f"[ORCHESTRATOR] Instructions: {instructions[:100]}...")
                        
                        task_created = True

                except json.JSONDecodeError as e:
                    logger.error(f"[ORCHESTRATOR] JSON parse error: {e}")
                    task_created = False
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] Error: {e}", exc_info=True)
                    task_created = False


            
            
            elif agent_name == "email":
                tools = agent_tools_map.get(agent_name, {}).get("tools", [])
                # Generate instructions for email
                instructions_response = await self.llm_service.call_ollama(
                    model_name=self.model,
                    prompt=f"""
                    You are a Email Agent which is part of a Multi Agents Context Management Platform which also contains other Agents like Calendar Agent and Tasks Agent.
                    Your work is to manage all the tasks related to Email, now analyze the user request and determine:

                As mentioned u are a part of multi agents system so sometimes u may get user inputs that will also include requests related to other agents, in this case just ignore them and just focus on the requests related to emails.
                Example:
                User Input: Create a task named Exercise in My Tasks task list and send a interview reminder mail to Vishnu3sgk08@gmail.com just to say hello.
                In case if u get such kind of input then just focus on "send a interview reminder mail to Vishnu3sgk08@gmail.com just to say hello" and ignore the requets related to other agents.
                
                Analyze the user request and determine:
                1. What EMAIL ACTION the user wants (send, find, reply, delete, archive, etc.)
                2. How many DISTINCT tasks the user is asking for
                3. For each task, check if sufficient context exists
                4. Generate instructions OR identify missing context for each task

                CRITICAL RULES:
                - Generate instructions ONLY, don't extract parameters!
                - Only generate tasks if user EXPLICITLY asks for multiple operations
                - Single queries like "send a mail" = 1 task
                - Multiple operations like "find mail X and send reply to same" = 2 tasks
                - Default to 1 task unless user clearly specifies multiple
                
                Conversation context (last 5 messages):
                {context}

                User Request: {state["user_input"]}

                Available Email Tools:
                {available_tools}

                
                TASK COUNT DETECTION:
                - Keywords for multiple tasks: "and", "also", "create...update", "plus", "additionally"
                - Keywords for single task: "get", "show", "list", "find", "check", "retrieve"
                
                Return ONLY a JSON array with email tasks:
                
                ACTION IDENTIFICATION KEYWORDS:
                - "send_email" -> "intent_keywords": ["send", "compose", "email to", "write to", "draft"]
                - "find_email" -> "intent_keywords": ["find", "search", "look for", "locate", "check email"]
                - "reply_email" -> "intent_keywords": ["reply", "respond to", "answer", "get back to"]
                - "delete_email" -> "intent_keywords": ["delete", "remove", "trash", "discard"]
                - "remove_label_email" -> "intent_keywords": ["remove label", "unlabel", "clear label", "unread", "mark unread"]
                - "create_draft_reply" -> "intent_keywords": ["draft reply", "create draft", "prepare reply", "compose reply"]
                - "create_draft" -> "intent_keywords": ["draft", "create draft", "prepare", "compose"]
                - "add_label" -> "intent_keywords": ["add label", "label", "mark", "categorize"]
                
                REQUIRED PARAMETERS by Action:
                - send_email: recipient_email (REQUIRED)
                - find_email: search_query (REQUIRED)
                - reply_email: message about the mail to be replied(REQUIRED)
                - delete_email: Email message to be Deleted(REQUIRED)
                - remove_label_email: message about the email (REQUIRED), label_name (REQUIRED)
                - add_label: message about the email (REQUIRED), label_name (REQUIRED)
                - create_draft_reply: search_query (REQUIRED)
                - create_draft: recipient_email (REQUIRED)
                
                Analyze the Conversation context also with the user request to determine if sufficient context exists to fulfill each task, where sometimes you can find parameters like task_list_name, task_name and others in the Conversation context that are missing in user request.
                If any required parameter that is missing in user input is found in conversational context them include that parameter in the instruction with out fail. 
                RESPONSE FORMAT Examples:
                The output should be in the below given JSON array format only.
                If ALL required parameters present:
                Examples:
                - Get me all the tasks present in AgentSync list.
                response:
                [
                {{"need_context": false, "action": "send_email", "instructions": "Send an email to john@example.com with subject Hello and message Hi John"}}
                ]
                
                If missing required parameters but found in conversational context:
                Examples:
                - send a friendly mail to Vishnu K. (or) send an email to Vishnu K. with subject Greetings and body Hope you are doing well.
                here in the above input the recipient email of the recipient is missing, if the conversational context has the information about the email associated to recipient then include that in the instruction.
                response:
                [
                {{"need_context": false, "action": "send_email", "instructions": "Send an email to <email associated ti Vishnu K> with subject Hello and message Hi John"}}
                ]
                
                If missing required parameters(parameters also not found in conversational context):
                Examples:
                - Send a reminder mail to all the team members involved in the Finance Disscussion event.
                response:
                [
                {{"need_context": true, "action": "send_email", "search_words": "Finance Disscussion event", "query": "find the email of team members involved in Finance Disscussion event", "missing_params": ["recipient_emails"]}}
                ]
                
                IMPORTANT:
                - search_words: Keywords that would find this in ChromaDB (use user's phrasing)
                - query: What information we need to find
                - instruction: Only include if need_context=false
                - Keep all required params in search_words and query for context retrieval
                
                Return ONLY a JSON array with the detected number of tasks:
                For multiple tasks with sufficient context:
                Examples:
                - Send a reply mail to Interview Reminder mail sent by vishnu3sgk08@gmail.com to express thanks and Find the unread mails in my inbox.
                response:
                [
                {{"need_context": false, "action": "reply_email", "instructions": "Reply to the Interview Reminder email sent by vishnu3sgk08@gmail.com to thank for reminding me"}},
                {{"need_context": false, "action": "find_email", "instructions": "Find or check for all the unread mails in the inbox"}}
                ]
                
                For multiple tasks needing context:
                Examples:
                - send an email to Vishnu K. with subject Greetings and body Hope you are doing well and Send a reminder mail to all the team members or attendees involved in the Finance Disscussion event.
                response:
                [
                {{"need_context": true, "action": "send_email", "search_words": "Vishnu K", "query": "find the email_id of Vishnu K", "missing_params": ["recipient_email"]}}
                {{"need_context": true, "action": "send_email", "search_words": "Finance Disscussion event", "query": "find the email of team members involved in Finance Disscussion event", "missing_params": ["recipient_emails"]}}                
                ]
                
                Clarification Examples:
                - Send an remove label unread from the Interview Reminder mail email sent by vishnu3sgk08@gmail.com. -> action: remove_label_email, label_name: unread, recipient_email: vishnu3sgk08@gmail.com.
                response:
                [{{"need_context": false, "action": "remove_label_email", "instructions": "Remove label unread from the Interview Reminder email sent by vishnu3sgk08@gmail.com."}}]

                Important:
                - For send/reply actions → include recipient info in instructions and also include the subject and body in the instructtion itself.                
                - For find_email → include search query in instructions and if the user asks to find a particular mail and explain it just include the seaarch query in the insruction dont include anything about asking to explain tthe mail.
                Example: 
                    Find and explain the DAA Assignment mail sent by Vishnu K completely.
                    Response:
                    [ 
                    {{"need_context": false, "action": "find_email", "instructions": "Find or check for DAA Assignment mail in the inbox"}}
                    ]
                - For Example for input like "check if any one has said hello mails in the inbox" here the instruction should be "Search for emails with subject or body containing 'hello' in the inbox" hence action is find_email.
                - For delete → mention which email to delete like sender or sender email or subject.
                - For send/reply actions → the subject and body must be included in the instruction itself.
                - If there is any send mail request to multiple mails with same subject and body(the body should not be personalized) then create only one send_email task with all the recipient mails in the instruction included in itself.
                - For the requests where user asks to send mails to multiple recipients with different subject and body then create 2 different send_email tasks one for each recipient with respective subject and body in the instruction itself.
                Example: send friendly mail to mail john@gmail.com and an interview reminder mail to nick@gmail.com
                [
                {{"action": "send_email", "instructions": "Send an email to john@example.com with subject Hello and message Hi John"}},
                {{"action": "send_email", "instructions": "Send an email to nick@example.com with subject Interview Reminder and message about reminding the interview"}}
                ]
                - Instructions must be COMPLETE and SELF-CONTAINED

                Return ONLY the JSON array, NO other text.
                Your Work is to only identify the Action need to be used based on the user request and add the instruction and then return the JSON object as shown in the examples.
                For send_email/reply_email actions → generaate the subject and body based on user request and include it in the instruction itself.
                The response should only contain a JSON object without any extra text like Explaination or Note and others as given in the example.
                Make sure u return the response strictly according to the example given above and donot missout any inverted comma.
                """
                )

                try:
                    # Parse instructions response
                    instructions_response = instructions_response.strip()
                    logger.info(f"[ORCHESTRATOR] LLM Response: {instructions_response}")
                    json_match = re.search(r'\[.*\]', instructions_response, re.DOTALL)
                    if json_match:
                        instructions_response = json_match.group(0)
                    
                    tasks_list = json.loads(instructions_response)
                    if not isinstance(tasks_list, list):
                        tasks_list = [tasks_list]
                    
                    logger.info(f"[ORCHESTRATOR] Generated {len(tasks_list)} email task(s) with instructions")
                    
                    # Create a task for each instruction
                    for task_spec in tasks_list:
                        action = task_spec.get("action", "create_task")
                        need_context = task_spec.get("need_context", False)
                        instructions = task_spec.get("instructions", state["user_input"])
                        
                        # ✅ NEW: Special handling for reply_email - auto-find email first
                        if action == "reply_email" or action == "remove_label_email" or action == "create_draft_reply" or action == "add_label" or action == "delete_email":
                            logger.info(f"[ORCHESTRATOR] running find_email first")
                            
                            # Step 1: Generate find_email instruction
                            find_instruction_response = await self.llm_service.call_ollama(
                                model_name=self.model,
                                prompt=f"""Based on this user request, generate a search query to find the email they want.

                    User Request: {state["user_input"]}

                    Extract:
                    - Who is the sender? (name or email)
                    - What is the email about? (subject keywords)

                    Return ONLY the search instruction, no extra text.
                    Example: "Search for email from John about project update" 
                    In the search query just include instructions to search for some mails with a X subjecct or body that is enough donot include other information like to search for emails with friendly tone.
                    Correct search query Example: Search for emails containing 'hello' in subject or body in the inbox.
                    Incorrect search query Example: Search for emails containing 'hello' in subject and sender name or address with a friendly tone to reply back hello within Inbox.
                    Your response should ONLY contain the search instruction.
                    One important thing:
                        The response that u generate or return must be parsable or extractable by the below code which will run after u respond:
                            if find_result.get("status") == "success":
                                try:
                                    email_data = find_result.get("data", {{}})
                                    result_content = email_data.get("result", {{}}).get("result", {{}}).get("content", [])
                                    
                                    thread_id = None
                                    from_email = None
                                    
                                    for item in result_content:
                                        if isinstance(item, dict) and item.get("type") == "text":
                                            content_data = json.loads(item.get("text", "\"{{}}\""))
                                            results_list = content_data.get("results", [])
                                            
                                            if results_list and len(results_list) > 0:
                                                first_email = results_list[0]
                                                thread_id = first_email.get("thread_id")
                                                from_data = first_email.get("from", {{}})
                                                from_email = from_data.get("email")
                                                break
                                                """
                            )
                            
                            find_instruction = find_instruction_response.strip().strip('"')
                            logger.info(f"[ORCHESTRATOR] Generated find instruction: {find_instruction}")
                            
                            # Step 2: Execute find_email
                            from app.agents.email_agent import EmailAgent
                            
                            temp_email_agent = EmailAgent(
                                agent_id=1,
                                # user_id=state.get("user_id", 1),
                                mq_service=self.mq_service,
                                llm_service=self.llm_service
                            )
                            
                            find_result = await temp_email_agent.execute({
                                "action": "find_email",
                                "parameters": {"instructions": find_instruction},
                                "user_input": state["user_input"]
                            })
                            
                            logger.info(f"[ORCHESTRATOR] Find email result: {find_result.get('status')}")
                            
                            # Step 3: Extract thread_id and from_email
                            if find_result.get("status") == "success":
                                try:
                                    email_data = find_result.get("data", {})
                                    result_content = email_data.get("result", {}).get("result", {}).get("content", [])
                                    
                                    thread_id = None
                                    from_email = None
                                    
                                    for item in result_content:
                                        if isinstance(item, dict) and item.get("type") == "text":
                                            content_data = json.loads(item.get("text", "{}"))
                                            results_list = content_data.get("results", [])
                                            
                                            if results_list and len(results_list) > 0:
                                                first_email = results_list[0]
                                                thread_id = first_email.get("thread_id")
                                                from_data = first_email.get("from", {})
                                                from_email = from_data.get("email")
                                                break
                                    
                                    logger.info(f"[ORCHESTRATOR] Extracted - Thread: {thread_id}, From: {from_email}")
                                    
                                    # Step 4: Enhance instructions with thread_id and from_email
                                    if action == "reply_email" and thread_id and from_email:
                                        enhanced_instructions = f"""Reply to email thread: {thread_id}
                                        Send to: {from_email} \nUser's reply request: {state["user_input"]} \nGenerate appropriate reply email."""
                                        
                                    if action == "create_draft_reply" and thread_id and from_email:
                                        enhanced_instructions = f"""Reply to email thread: {thread_id}
                                        Send to: {from_email} \nUser's reply request: {state["user_input"]} \nGenerate appropriate reply email."""
                                        
                                    if action == "remove_label_email" and thread_id and from_email:
                                        enhanced_instructions = f"""Remove label from email with id: {thread_id} \nUser's request: {state["user_input"]} \nPerform the label removal as requested."""
                                        
                                    if action == "add_label" and thread_id and from_email:
                                        enhanced_instructions = f"""Remove label from email with id: {thread_id} \nUser's request: {state["user_input"]} \nPerform the label removal as requested."""
                                        
                                    if action == "delete_email" and thread_id and from_email:
                                        enhanced_instructions = f"""Delete the email with message_id: {thread_id} \nSender mail_id: {from_email}  \nUser's request: {state["user_input"]} \nPerform the deletion as requested."""
                                        
                                        
                                    instructions = enhanced_instructions
                                    logger.info(f"[ORCHESTRATOR] ✅ Enhanced instructions with thread_id and from_email")
                                    # else:
                                    #     logger.warning(f"[ORCHESTRATOR] Could not extract thread_id or from_email")
                                        
                                except Exception as e:
                                    logger.error(f"[ORCHESTRATOR] Error processing find result: {e}", exc_info=True)
                            else:
                                logger.warning(f"[ORCHESTRATOR] Find email failed: {find_result.get('error')}")
                        
                        # Continue with normal task creation
                        if need_context:
                            logger.info(f"[ORCHESTRATOR] Task needs context, starting enrichment")
                            
                            has_context, enriched_instructions, error_msg = await self._analyze_and_enrich_context(
                                user_input=state["user_input"],
                                agent_name=agent_name,
                                task_response=task_spec,
                                action=action,
                                available_tools=tools,
                                state=state
                            )
                            
                            if has_context:
                                # Context enrichment successful
                                instructions = enriched_instructions
                                logger.info(f"[ORCHESTRATOR] ✅ Context enriched: {instructions}...")
                            else:
                                # Context still insufficient - store error and skip task
                                logger.warning(f"[ORCHESTRATOR] ❌ Insufficient context: {error_msg}")
                                state["execution_errors"][f"{agent_name}_{action}"] = error_msg
                                state["insufficient_context"] = True
                                state["has_critical_error"] = True
                                state["error_message"] = error_msg
                                
                                # Return immediately - don't create fallback!
                                return state
                        
                        
                        # Validate action exists in tools
                        available_tool_names = [t["name"] for t in tools]
                        
                        if action not in available_tool_names:
                            logger.warning(f"[ORCHESTRATOR] Action '{action}' not in available tools")
                            action = available_tool_names[0] if available_tool_names else "create_task"
                        
                        # Create task with enriched instructions
                        parameters = {"instructions": instructions}
                        
                        if action == "delete_email":
                            parameters = {"instructions": instructions, "message_id": thread_id}
                            
                        
                        task = AgentTask(
                            agent_name=agent_name,
                            action=action,
                            parameters=parameters,
                            task_id=f"{state['task_metadata']['task_id']}_{agent_name}_{len(tasks)}"
                        )
                        
                        tasks.append(task)
                        logger.info(f"[ORCHESTRATOR] ✅ Created task: {action}")
                        logger.info(f"[ORCHESTRATOR] Instructions: {instructions[:100]}...")
                        
                        task_created = True

                except json.JSONDecodeError as e:
                    logger.error(f"[ORCHESTRATOR] JSON parse error: {e}")
                    task_created = False
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] Error: {e}", exc_info=True)
                    task_created = False


            
            
            elif agent_name == "calendar":
                tools = agent_tools_map.get(agent_name, {}).get("tools", [])
                # Generate instructions for calendar
                instructions_response = await self.llm_service.call_ollama(
                    model_name=self.model,
                    prompt=f"""
                    You are a Calendar Agent which is part of a Multi Agents Context Management Platform which also contains other Agents like Email Agent and Tasks Agent.
                    your work is to manage all the tasks reated to calendar now analyze the user request and determine:

                As mentioned u are a part of multi agents system so sometimes u may get user inputs that will also include requests related to other agents, in this case just ignore them and just focus on the requests related to calendar.
                Example:
                User Input: Create a event Finance Disscussion on 4/12/2025 from 3pm to 4pm IST and send a interview reminder mail to Vishnu3sgk08@gmail.com just to say hello.
                In case if u get such kind of input then just focus on "Create a event Finance Disscussion on 4/12/2025 from 3pm to 4pm IST" and ignore the requets related to other agents.
                
                Analyze the user request and determine:
                1. What CALENDAR ACTION the user wants (create, list, update, delete, find)
                2. How many DISTINCT tasks the user is asking for
                3. For each task, check if sufficient context exists
                4. Generate instructions OR identify missing context for each task

                CRITICAL RULES:
                - Generate instructions ONLY, don't extract parameters!
                - Only generate tasks if user EXPLICITLY asks for multiple operations
                - Single queries like "create an event" = 1 task
                - Multiple operations like "create even X and delete event Y" = 2 tasks
                - Default to 1 task unless user clearly specifies multiple
                
                Conversation context (last 5 messages):
                {context}

                User Request: {state["user_input"]}

                Available Calendar Tools:
                {available_tools}

                TASK COUNT DETECTION:
                - Keywords for multiple tasks: "and", "also", "create...update", "plus", "additionally"
                - Keywords for single task: "get", "show", "list", "find", "check", "retrieve"
                
                
                ACTION IDENTIFICATION KEYWORDS:
                - "quick_add_event" -> "intent_keywords": ["quick add", "add event quickly", "create event from text"]
                - "create_event" -> "intent_keywords": ["create event", "create detailed", "detailed", "schedule", "book", "set up", "add event"]
                - "update_event" -> "intent_keywords": ["update", "reschedule", "change", "modify", "move"]
                - "find_events" -> "intent_keywords": ["find", "list", "show", "get", "view"]
                - "delete_event" -> "intent_keywords": ["delete", "cancel", "remove", "discard", "drop"]
                - "add_attendee_s_to_event" -> "intent_keywords": ["invite", "add attendee", "include person", "invite person"]
                
                REQUIRED PARAMETERS by Action:
                - quick_add_event: event_details in natural language (REQUIRED)
                - create_event: event_title (REQUIRED), date and time (REQUIRED), attendees (OPTIONAL)
                - update_event: event_title (REQUIRED), details to be updated (REQUIRED)
                - find_events: search query (REQUIRED)
                - delete_event: event_title (REQUIRED)
                - add_attendee_s_to_event: event_title (REQUIRED), attendee_emails (REQUIRED)

                Return ONLY a JSON array with calendar tasks:
                
                Analyze the Conversation context also with the user request to determine if sufficient context exists to fulfill each task, where sometimes you can find parameters like task_list_name, task_name and others in the Conversation context that are missing in user request.
                If any required parameter that is missing in user input is found in conversational context them include that parameter in the instruction with out fail. 
                RESPONSE FORMAT Examples:
                The output should be in the below given JSON array format only.
                If ALL required parameters present:
                Examples: create event named Project Kickoff on July 5th at 10 AM to 12PM for IST.
                [
                {{"action": "create_event", "instructions": "Create an event called 'Project Kickoff' on July 10th at 10 AM to 12PM with respect to IST"}}
                ]
                
                If missing required parameters but found in conversational context:
                Examples:
                - Update the recently created event timimg to 12/12/2025 3pm to 5pm.
                here in the above input the event title is missing, if the conversational context has the information about the event title that was most recently created then include that in the instruction.
                response:
                [
                {{"need_context": false, "action": "update_event", "instructions": "Update the event <event title in the conversational context> timing to 12/12/2025 3pm to 5pm"}}
                ]
                
                If missing required parameters(parameters also not found in conversational context):
                Examples:
                - Create an event called Interview in Agent sync Calendar.
                here in the above input details of event start and end time is missing. 
                response:
                [
                {{"need_context": true, "action": "update_event", "search_words": "Interview", "query": "find the timings related to event Interview", "missing_params": ["event_timings"]}}
                ]
                
                IMPORTANT:
                - search_words: Keywords that would find this in ChromaDB (use user's phrasing)
                - query: What information we need to find
                - instruction: Only include if need_context=false
                - Keep all required params in search_words and query for context retrieval
                
                
                
                For multiple tasks with sufficient context:
                Examples:
                - Find all the events that i have today in my Agent sync Calendar and 
                response:
                [
                {{"need_context": false, "action": "find_event", "instructions": "Find all the events that i have today in my Agent sync Calendar."}},
                {{"need_context": false, "action": "add_attendee_s_to_event", "instructions": "Add attendees Vishnu K(vishnu3sgk08@gmail.com and Shyam S(shyamramesh715@gmail.com) to the event Project Discussion scheduled on 15th June 2024 at 3 PM in Agent sync Calendar."}}
                ]
                
                For multiple tasks needing context:
                Examples:
                - Create an event called Interview in Agent sync Calendar and Add Vishnu K. as an attendee to the Team Meeting event.
                response:
                [
                {{"need_context": true, "action": "create_event", "search_words": "Interview", "query": "find the event timings and other details related to Interview event", "missing_params": ["event_timings"]}}
                {{"need_context": true, "action": "add_attendee_s_to_event", "search_words": "Vishnu K", "query": "find the email associated to Vishnu K", "missing_params": ["attendee_email"]}}                
                ]
                Important:
                - For create/update → include event title, date, time in instructions
                - For list → include time range (today, this week, specific date)
                - For find → include search criteria
                - For delete → mention which event to delete
                - Instructions must be COMPLETE and SELF-CONTAINED

                Return ONLY the JSON array, NO other text.
                Your Work is to only identify the Action need to be used based on the user request and add the instruction and then return the JSON object as shown in the examples.

                The response should only contain a JSON object without any extra text like Explaination or Note and others as given in the example.
                """
                )

                try:
                    instructions_response = instructions_response.strip()
                    logger.info(f"[ORCHESTRATOR] LLM Response: {instructions_response}")
                    json_match = re.search(r'\[.*\]', instructions_response, re.DOTALL)
                    if json_match:
                        instructions_response = json_match.group(0)
                    
                    tasks_list = json.loads(instructions_response)
                    
                    if not isinstance(tasks_list, list):
                        tasks_list = [tasks_list]
                    
                    logger.info(f"[ORCHESTRATOR] Generated {len(tasks_list)} task(s) with instructions")
                    
                    for task_spec in tasks_list:
                        action = task_spec.get("action", "create_task")
                        need_context = task_spec.get("need_context", False)
                        
                        if need_context:
                            logger.info(f"[ORCHESTRATOR] Task needs context, starting enrichment")
                            
                            has_context, enriched_instructions, error_msg = await self._analyze_and_enrich_context(
                                user_input=state["user_input"],
                                agent_name=agent_name,
                                task_response=task_spec,
                                action=action,
                                available_tools=tools,
                                state=state
                            )
                            
                            if has_context:
                                # Context enrichment successful
                                instructions = enriched_instructions
                                logger.info(f"[ORCHESTRATOR] ✅ Context enriched: {instructions}...")
                            else:
                                # Context still insufficient - store error and skip task
                                logger.warning(f"[ORCHESTRATOR] ❌ Insufficient context: {error_msg}")
                                state["execution_errors"][f"{agent_name}_{action}"] = error_msg
                                state["insufficient_context"] = True
                                state["has_critical_error"] = True
                                state["error_message"] = error_msg
                                
                                # Return immediately - don't create fallback!
                                return state
                        else:
                            # Sufficient context from initial analysis
                            instructions = task_spec.get("instructions", state["user_input"])
                        
                        # Validate action exists in tools
                        available_tool_names = [t["name"] for t in tools]
                        
                        if action not in available_tool_names:
                            logger.warning(f"[ORCHESTRATOR] Action '{action}' not in available tools")
                            action = available_tool_names[0] if available_tool_names else "create_task"
                        
                        # Create task with enriched instructions
                        parameters = {"instructions": instructions}
                        
                        task = AgentTask(
                            agent_name=agent_name,
                            action=action,
                            parameters=parameters,
                            task_id=f"{state['task_metadata']['task_id']}_{agent_name}_{len(tasks)}"
                        )
                        
                        tasks.append(task)
                        logger.info(f"[ORCHESTRATOR] ✅ Created task: {action}")
                        logger.info(f"[ORCHESTRATOR] Instructions: {instructions[:100]}...")
                        
                        task_created = True

                except json.JSONDecodeError as e:
                    logger.error(f"[ORCHESTRATOR] JSON parse error: {e}")
                    task_created = False
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] Error: {e}", exc_info=True)
                    task_created = False
            
            
            # ✅ For other agents, use standard approach
            else:
                logger.info(f"[ORCHESTRATOR] Using STANDARD approach for {agent_name}")

                task_params = None

                try:
                    # Standard parameter extraction for email/calendar
                    task_params = await self.llm_service.call_ollama(
                        model_name="mistral",
                        prompt=self._get_task_prompt(agent_name, available_tools, state["user_input"])
                    )

                    task_params = task_params.strip()
                    task_params = re.sub(r'```json\n?', '', task_params, flags=re.MULTILINE)
                    task_params = re.sub(r'```\n?', '', task_params, flags=re.MULTILINE)

                    json_match = re.search(r'\{.*\}', task_params, re.DOTALL)
                    if json_match:
                        task_params = json_match.group(0)

                    logger.debug(f"[ORCHESTRATOR] Extracted JSON: {task_params}")
                    params = json.loads(task_params)
                    action = params.pop("action", None)

                    available_tool_names = [
                        t["name"] for t in agent_tools_map.get(agent_name, {}).get("tools", [])
                    ]

                    if not action or not isinstance(action, str):
                        logger.warning(f"[ORCHESTRATOR] Invalid action from LLM, using first available tool")
                        action = available_tool_names[0] if available_tool_names else "send_email"

                    elif action not in available_tool_names:
                        logger.warning(f"[ORCHESTRATOR] Action '{action}' not in available tools")
                        action = available_tool_names[0] if available_tool_names else "send_email"

                    logger.info(f"[ORCHESTRATOR] ✅ Action is STRING: {action} (type: {type(action).__name__})")
                    logger.info(f"[ORCHESTRATOR] ✅ Created task for {agent_name}: {action}")

                    task = AgentTask(
                        agent_name=agent_name,
                        action=action,
                        parameters=params,
                        task_id=f"{state['task_metadata']['task_id']}_{agent_name}_{len(tasks)}"
                    )

                    tasks.append(task)
                    task_created = True

                except json.JSONDecodeError as e:
                    logger.error(f"[ORCHESTRATOR] JSON parse error: {e}")
                    logger.error(f"[ORCHESTRATOR] Raw response: {task_params}")

                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] Error: {e}", exc_info=True)

            if not task_created:
                logger.warning(f"[ORCHESTRATOR] Using fallback task for {agent_name}")
                available_tool_names = [
                    t["name"] for t in agent_tools_map.get(agent_name, {}).get("tools", [])
                ]
                
                fallback_action = available_tool_names if available_tool_names else "send_email"
                
                if agent_name == "tasks":
                    fallback_params = {"instructions": state["user_input"]}
                elif agent_name == "email":
                    fallback_params = {"instructions": state["user_input"]}
                elif agent_name == "calendar":
                    fallback_params = {"instructions": state["user_input"]}
                else:
                    fallback_params = {}
                
                task = AgentTask(
                    agent_name=agent_name,
                    action=fallback_action,
                    parameters=fallback_params,
                    task_id=f"{state['task_metadata']['task_id']}_{agent_name}_{len(tasks)}"
                )
                
                tasks.append(task)


        state["tasks"] = tasks
        state["execution_trace"].append({
            "node": "create_tasks",
            "timestamp": datetime.now().isoformat(),
            "task_count": len(tasks),
            "tasks": [{"agent": t.agent_name, "action": t.action} for t in tasks]
        })

        logger.info(f"[ORCHESTRATOR] Created {len(tasks)} tasks total")

        for task in tasks:
            logger.info(f"[ORCHESTRATOR] Task: {task.agent_name} → {task.action}")

        return state

    def _get_task_prompt(self, agent_name: str, available_tools: str, user_input: str) -> str:
        """Get task creation prompt for non-tasks agents"""
        base_prompt = f"""Convert this action into structured parameters for the {agent_name} agent.

Actions: {user_input}

Available Tools for {agent_name}:
{available_tools}

Return ONLY ONE JSON object, no multiple!
Use ONLY tools from the list above.
NO comments, NO extra text, ONLY valid JSON.

action must be a STRING, exact tool name from list."""

        if agent_name == "email":
            return base_prompt + """

TOOL SELECTION RULES:
- "reply" → reply_email
- "send" (no reply) → send_email
- "find/search/check" → find_email
- "delete" (email mentioned) → delete_email

Example responses:
{"action": "send_email", "to": "user@example.com", "subject": "Hello", "body": "Message"}
{"action": "find_email", "query": "subject:interview"}"""

        elif agent_name == "calendar":
            return base_prompt + """

TOOL SELECTION RULES:
- "create/schedule/book" event → create_detailed_event
- "update/reschedule" event → update_event
- "find/list/show" events → find_events
- "delete/cancel" event → delete_event

Example responses:
{"action": "create_detailed_event", "title": "Meeting", "start_time": "2025-11-20T10:00:00"}"""

        return base_prompt

    def _extract_fallback_params(self, agent_name: str, action: str, user_input: str) -> Dict[str, Any]:
        """Extract fallback parameters when LLM fails"""
        params = {}

        if agent_name == "email":
            if action == "send_email":
                email_pattern = r'[\w\.-]+@[\w\.-]+'
                email_match = re.search(email_pattern, user_input)
                params = {
                    "to": email_match.group(0) if email_match else "",
                    "subject": "Message from AgentSync",
                    "body": "Hello! This is a message from AgentSync."
                }

        elif agent_name == "calendar":
            params = {
                "title": "Event",
                "start_time": datetime.now().isoformat()
            }

        elif agent_name == "tasks":
            params = {
                "instructions": user_input
            }

        return params

    async def queue_tasks_node(self, state: OrchestratorState) -> OrchestratorState:
        """NODE 3: Queue tasks to Redis"""
        logger.info(f"[ORCHESTRATOR] Queuing tasks to Redis")

        for task in state["tasks"]:
            try:
                queue_key = f"{task.agent_name}_tasks"
                task_dict = {
                    "task_id": task.task_id,
                    "agent_name": task.agent_name,
                    "action": task.action,
                    "parameters": task.parameters,
                    "user_input": state["user_input"],
                    # "user_id": state["user_id"],
                    # "session_id": state["session_id"],
                    "queued_at": datetime.now().isoformat()
                }

                await self.mq_service.enqueue_task(queue_key, task_dict)
                state["queue_status"][task.task_id] = "queued"
                logger.info(f"[ORCHESTRATOR] Queued task {task.task_id} to {queue_key}")

            except Exception as e:
                logger.error(f"[ORCHESTRATOR] Failed to queue task: {e}")
                state["execution_errors"][task.task_id] = str(e)
                state["queue_status"][task.task_id] = "failed"

        state["execution_trace"].append({
            "node": "queue_tasks",
            "timestamp": datetime.now().isoformat(),
            "queued_count": sum(1 for s in state["queue_status"].values() if s == "queued")
        })

        return state

    async def collect_responses_node(self, state: OrchestratorState) -> OrchestratorState:
        """NODE 4: Collect responses from agents"""
        logger.info(f"[ORCHESTRATOR] Collecting agent responses")

        timeout_per_task = 150
        poll_interval = 0.5

        for task in state["tasks"]:
            result_key = task.task_id
            max_polls = int(timeout_per_task / poll_interval)
            polls = 0

            logger.info(f"[ORCHESTRATOR] Waiting for result from {task.agent_name}")

            while polls < max_polls:
                try:
                    result = await self.mq_service.get_result(result_key)
                    if result:
                        # ✅ Store by task_id, NOT agent_name
                        state["agent_responses"][task.task_id] = result
                        state["queue_status"][task.task_id] = "completed"
                        logger.info(f"[ORCHESTRATOR] Received result from {task.agent_name}: {task.action}")
                        break

                    polls += 1
                    await asyncio.sleep(poll_interval)

                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] Error polling: {e}")
                    polls += 1
                    await asyncio.sleep(poll_interval)

            if polls >= max_polls:
                logger.warning(f"[ORCHESTRATOR] Timeout waiting for {task.agent_name}")
                state["execution_errors"][task.agent_name] = f"Response timeout after {timeout_per_task}s"
                state["queue_status"][task.task_id] = "timeout"

        return state

            


    async def aggregate_results_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        NODE 5: Aggregate results into final response with TOON optimization
        ✅ ADDED: Context storage after aggregation
        """
        logger.info("[ORCHESTRATOR] Aggregating results")
        
        try:
            all_results = {}
            for task_id, result in state["agent_responses"].items():
                status = result.get("status", "unknown")
                message = result.get("message", "No message")
                action = result.get("action", "unknown")
                
                all_results[task_id] = {
                    "status": status,
                    "message": message,
                    "action": action,
                    "data": result.get("data", {})
                }
            
            logger.info(f"[ORCHESTRATOR] Aggregated {len(all_results)} task result(s)")
            
            # Build essential results for LLM
            essential_results = []
            for task_id, result in all_results.items():
                essential_results.append({
                    "status": result["status"],
                    "action": result["action"],
                    "data": result["data"]
                })
                
            logger.info(f"[ORCHESTRATOR] Zapier Response: {essential_results}")
            
            # Detect format
            format_type = "JSON"
            tasks_summary = json.dumps(essential_results, indent=2)
            
            
            # Try TOON encoding
            try:
                toon_output = encode_to_toon(essential_results)
                if toon_output and len(toon_output) < len(tasks_summary):
                    logger.info("[ORCHESTRATOR] ✅ TOON encoding successful")
                    tasks_summary = toon_output
                    format_type = "TOON"
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] TOON encoding failed: {e}")
            
            logger.info(f"[ORCHESTRATOR] Essential results for summary - {format_type}:")
            logger.info(tasks_summary)
            
            # ✅ ENHANCED PROMPT with CONTEXT_SUMMARY
            prompt = f"""You are a task completion reporter. Analyze the task execution and provide a comprehensive summary.

    USER REQUEST:
    {state["user_input"]}
    
    ACTIONS Performed:
    {action}

    TASKS EXECUTED ({len(essential_results)} total) - Format: {format_type}:
    TASK RESULTS: 
    {tasks_summary}

    INSTRUCTIONS:
    1. ANALYZE the data section which contains ALL response information
    2. If status is "success", summarize what was accomplished using details from data
    3. If you find 'isError': true in data, treat as error even if status shows success
    4. If status is "error", mention the error and action that failed
    5. For tasks with results (lists, created items, updates), mention specific details

    OUTPUT FORMAT (provide BOTH sections):

    **USER_RESPONSE:**
    [2 concise sentences for the user]
    - Use bullet points if multiple tasks were executed
    - Be specific: mention task names, item counts, or operations performed
    - Exclude technical details like IDs, timestamps, or internal metadata
    - Be confident and user-friendly
    - Understand the user request, analyse the task results and generate the user response based on what the user has asked for.

    **CONTEXT_SUMMARY:**
    [Detailed context for future reference - include:]
    - Agent type that executed the task (email/calendar/tasks)
    - Specific action performed (e.g., "sent email", "created event", "updated task")
    - Operation result (success/failure, item counts, IDs if relevant)
    - incase of errors, include error messages and failure reasons
    - in case of operations related to tasks include task list name, for email include subject or recipient info, for calendar include event title or time.
    - for tasks agent contexts when mentioning task list name mention it like: "My Tasks task list" in the output(if My Tasks is the name of the task list) not like "My Tasks list".
    - while generating context related to events, tasks, meetings or others if there is timings given for them include the timimgs also in the context summary.
    - For find_email action include all the contents in the body without missing anything in the context summary and include if any timings or deadline in to context without fail if present in thee body.
    Return BOTH sections, stricty with 4 to 5 sentences max for both sections combined."""
            
            # Call LLM
            start_time = datetime.now()
            response = await self.llm_service.call_ollama(
                model_name=self.model,
                prompt=prompt
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"[ORCHESTRATOR] LLM response time: {elapsed:.2f}s")
            logger.info(f"[ORCHESTRATOR] Got LLM response (length: {len(response)})")
            
            # ✅ Extract USER_RESPONSE section
            user_response_match = re.search(
                r'\*\*USER_RESPONSE:\*\*\s*\n(.*?)(?=\n\n\*\*CONTEXT_SUMMARY|\Z)',
                response,
                re.DOTALL
            )
            
            if user_response_match:
                final_response = user_response_match.group(1).strip()
            else:
                # Fallback: use entire response
                final_response = response
            
            state["final_response"] = final_response
            
            # ✅ STORE CONTEXT - THIS IS THE KEY ADDITION!
            # Determine primary agent type
            agent_types = [state.get("detected_agents", ["unknown"])[0]]
            primary_agent = agent_types[0] if agent_types else "unknown"
            
            if essential_results[0]["status"] == "success":            
                logger.info(f"[ORCHESTRATOR] Calling store_context_to_db for agent: {primary_agent}")
                
                store_context_to_db(
                    state=state,
                    llm_response=response,
                    agent_type=primary_agent
                )
            
                logger.info("[ORCHESTRATOR] ✅ Aggregation and context storage complete")
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Error in aggregation: {e}", exc_info=True)
            state["final_response"] = "Task completed but could not generate summary."
        
        # Add to conversation history
        state["conversation_history"].append({"role": "user", "content": state["user_input"]})
        state["conversation_history"].append({"role": "assistant", "content": state["final_response"]})
        
        state["execution_trace"].append({
            "node": "aggregate_results",
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info("[ORCHESTRATOR] Aggregation complete")
        
        return state



    def _generate_fallback_response_multi(self, all_results: Dict, user_input: str) -> str:
        """Generate response for multiple tasks when LLM fails"""
        response_parts = []
        
        for task_id, result in all_results.items():
            action = result.get("action", "unknown")
            status = result.get("status", "unknown")
            
            if status == "success":
                response_parts.append(f"✓ {action} completed successfully")
            else:
                response_parts.append(f"✗ {action} failed: {result.get('message', 'Unknown error')}")
        
        if response_parts:
            return "Tasks executed:\n" + "\n".join(response_parts)
        
        return "All tasks completed."
