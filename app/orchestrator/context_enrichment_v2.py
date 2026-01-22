# ================================================================
# 🔄 CONTEXT ENRICHMENT V2 - UPDATED WITH TASK_RESPONSE
# Includes task_response in LLM prompt for better context
# ================================================================
import re
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings

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


class ContextEnrichmentV2:
    """
    2-Tier Context Enrichment:
    
    TIER 1: ChromaDB Search
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
        """
        Main enrichment function - processes each task
        
        Input tasks format:
        [{
            "need_context": true,
            "action": "update_task",
            "search_words": "Interview Preparation task",
            "query": "task list name of Interview Preparation",
            "missing_params": ["task_list_name"]
        }]
        
        task_response: Full response from task agent (context information)
        
        Output tasks format:
        [{
            "need_context": false,
            "action": "update_task",
            "instruction": "Mark Interview Preparation in My Tasks as completed"
        }]
        OR
        [{
            "need_context": true,
            "action": "update_task",
            "target_agent": "tasks",
            "query": "task list name of Interview Preparation",
            "missing_params": ["task_list_name"]
        }]
        """
        
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
                logger.info(f"[CONTEXT_ENRICHMENT] TIER 1: Searching ChromaDB for: {search_words}")
                
                collection = self.client.get_collection(name="agent_contexts")
    
                # Perform search
                chromadb_results = collection.query(
                    query_texts=search_words,
                    # agent_id=agent_id,
                    n_results=5,
                    include=["documents", "metadatas", "distances"]
                )
                
                if chromadb_results:
                    logger.info(f"[CONTEXT_ENRICHMENT] Found {len(chromadb_results)} results in ChromaDB")
                    logger.info(f"[CONTEXT_ENRICHMENT] Found:  {chromadb_results}")
                    
                    # Try to generate instructions from ChromaDB results
                    enriched_task = await self._generate_instruction_from_chromadb(
                        user_input=user_input,
                        task=task,
                        chromadb_results=chromadb_results
                    )
                    
                    if enriched_task:
                        enriched_tasks.append(enriched_task)
                        continue
                    
                    if not enriched_task[0].get("need_context", True):
                        return enriched_task
                
                logger.info(f"[CONTEXT_ENRICHMENT] Insufficient context in ChromaDB")
            
        #     # TIER 3: Keep need_context=true and set target_agent for later request
        #     logger.info(f"[CONTEXT_ENRICHMENT] TIER 2: Keeping need_context=true, will request from agent")
            
        #     code for requesting context from agent
        
        return enriched_tasks
            
    
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
            logger.info(f"[CONTEXT_ENRICHMENT] Retrived ChromaDB results: {chromadb_results}")
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
            prompt = f"""You are a context fulfilling assistant. There are 3 main agents Email Agent, Calendar Agent and Tasks Agent from who you get the tasks to be performed that had insufficient Context. Your job is to generate a clear instruction for the action required using the context retrieved from ChromaDB.

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
    -For a Previous Agent Response like:
    [
    {{"need_context": true, "action": "update_task", "search_words": "Interview Preparation task", "query": "task list name of task named Interview Preparation", "missing_params": ["task_list_name"]}}
    ]
    Here the missing context is 'task_list_name'. If ChromaDB retrieved results contains context of the task list name, generate instruction like:
    [
    {{"need_context": false, "action": "update_task", "instruction": "Mark 'Interview Preparation' in 'My Tasks' as completed"}}
    ]
    If context is NOT sufficient in chromadb also:
    Then rerurn a error message asking tthe user to provide the missing context like: 
    [
    {{"need_context": true, "action": "update_task", "error_msg": "This user request: {user_input} doest have enough context. Missing <missing parameters> related to <task mentioned in user input>", "missing_params": ["task_list_name"]}}
    ]

Example 2:
    -For a Previous Agent Response like:
    [
    {{"need_context": true, "action": "update_event", "search_words": "Interview", "query": "find the timings related to event Interview", "missing_params": ["event_timings"]}}
    ]
    Here the missing context is 'event_timings'. If ChromaDB retrieved results contains context of the event timings, generate instruction like:
    [  {{"need_context": false, "action": "update_event", "instruction": "Update the event 'Interview' in Agent Sync Calendar to have timings on 20th June 2024 at 10 AM"}}
    If context is NOT sufficient in chromadb also:
    Then rerurn a error message asking tthe user to provide the missing context like: 
    [
    {{"need_context": true, "action": "update_event",  "error_msg": "This user request: {user_input} doest have enough context. Missing <missing parameters> related to <event mentioned in user input>", "missing_params": ["event_timings"]}}
    ]
    
Examplle 3:
    -For a Previous Agent Response like:
    [
    {{"need_context": true, "action": "send_email", "search_words": "Vishnu K", "query": "find the email_id of Vishnu K", "missing_params": ["recipient_email"]}}
    ]
    Here the missing context is 'recipient_email'. If ChromaDB retrieved results contains context of the recipient email, generate instruction like:
    [
    {{"need_context": false, "action": "send_email", "instruction": "Send an email to Vishnu K(vishnu3sgk08@gmail.com) with subject Greetings and body Hope you are doing well"}}
    ]
    If context is NOT sufficient in chromadb also:
    Then rerurn a error message asking the user to provide the missing context like: 
    [
    {{"need_context": true, "action": "send_email",  "error_msg": "This user request: {user_input} doest have enough context. Missing <missing parameters> related to <email recipient mentioned in user input>", "missing_params": ["recipient_email"]}}
    ]

"""

# If context is NOT sufficient:
# Respond by including the agent name(email, calendar and tasks agents) from which the relavant context can be obtained, like:
# [
# {{"need_context": true, "action": "update_task", "target_agent": "tasks", "query": "task list name of task named Interview Preparation", "missing_params": ["task_list_name"]}}
# ]
            
            logger.info(f"[CONTEXT_ENRICHMENT] Calling LLM to generate instruction...")
            logger.info(f"[CONTEXT_ENRICHMENT] Including task_response in prompt")
            
            response = await self.llm_service.call_ollama(
                model_name=self.model,
                prompt=prompt
            )
            
            logger.info(f"[CONTEXT_ENRICHMENT] LLM Response: {response[:200]}...")
            
            # Parse response
            llm_response = response.strip()
            
            # Check if response is null
            if llm_response.lower() == "null":
                logger.info(f"[CONTEXT_ENRICHMENT] LLM determined insufficient context")
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
                logger.warning(f"[CONTEXT_ENRICHMENT] Failed to parse LLM response as JSON")
                return None
            
        except Exception as e:
            logger.error(f"[CONTEXT_ENRICHMENT] Error generating instruction: {e}", exc_info=True)
            return None
        
        return None
    
    def _format_chromadb_results(self, chromadb_results: Dict[str, Any]) -> str:
        """
        Format ChromaDB query results for LLM context
        
        ChromaDB query() returns NESTED lists (one per query_text):
        {
            'ids': [['id1', 'id2', ...]],              # Nested
            'documents': [['doc1', 'doc2', ...]],      # Nested
            'metadatas': [[{...}, {...}]],              # Nested
            'distances': [[0.15, 0.23, ...]]            # Nested
        }
        
        We query with 1 search term → 1 nested list inside.
        Need to flatten by accessing .
        """
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
