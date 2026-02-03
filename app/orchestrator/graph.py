"""
LangGraph orchestrator
Defines the graph structure and execution flow
"""
import logging
import uuid
from datetime import datetime
from typing import Optional
from langgraph.graph import StateGraph, START, END

from .state import OrchestratorState
from .nodes import OrchestratorNodes
from app.services.message_queue import MessageQueueService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)



class AgentOrchestratorGraph:
    """LangGraph-based orchestrator for AgentSync"""
    
    def __init__(
        self,
        mq_service: MessageQueueService,
        llm_service: LLMService,
    ):
        self.mq_service = mq_service
        self.llm_service = llm_service
        self.nodes = OrchestratorNodes(mq_service, llm_service)
        self.graph = StateGraph(OrchestratorState)
        self._build_graph()
        self.compiled_graph = self.graph.compile()
    
    def _build_graph(self):
        """Build orchestrator graph with conditional routing"""
        
        # Existing nodes
        self.graph.add_node("analyze_intent", self.nodes.analyze_intent_node)
        self.graph.add_node("create_tasks", self.nodes.create_tasks_node)
        self.graph.add_node("queue_tasks", self.nodes.queue_tasks_node)
        self.graph.add_node("collect_responses", self.nodes.collect_responses_node)
        self.graph.add_node("aggregate_results", self.nodes.aggregate_results_node)
        # NEW: Error response node
        self.graph.add_node("send_error_response", self.send_error_response_node)
        self.graph.add_node("friendly_chat_response", self.friendly_chat_node)
        self.graph.add_node("save_context", self.save_context_node)

        self.graph.set_entry_point("analyze_intent")
        
        self.graph.add_conditional_edges(
            "analyze_intent",
            self._route_after_intent_analysis,
            {
                "create_tasks": "create_tasks",
                "friendly_chat_response": "friendly_chat_response",
                "save_context": "save_context"
            }
        )
        
        # ✅ NEW TERMINAL EDGES FOR CHAT AND CONTEXT
        self.graph.add_edge("friendly_chat_response", END)
        self.graph.add_edge("save_context", END)
        
        # CONDITIONAL: Route based on task creation result
        self.graph.add_conditional_edges(
            "create_tasks",
            self._route_after_task_creation,
            {"execute": "queue_tasks", "error": "send_error_response"}
        )
        
        # Rest of flow
        self.graph.add_edge("queue_tasks", "collect_responses")
        self.graph.add_edge("collect_responses", "aggregate_results")
        self.graph.add_edge("aggregate_results", END)
        
        # NEW: Error response ends immediately
        self.graph.add_edge("send_error_response", END)

    def _route_after_task_creation(self, state: OrchestratorState) -> str:
        """
        Decide: execute tasks or return error?
        
        Returns: "execute" or "error"
        """
        if state.get("insufficient_context") or state.get("has_critical_error"):
            logger.info("[ORCHESTRATOR] Critical error - terminating execution")
            return "error"
        
        return "execute" 
    

    def _route_after_intent_analysis(self, state: OrchestratorState) -> str:
        """
        ✅ NEW METHOD: Decide next node based on intent analysis result
        """
        run_task = state.get("run_task", True)
        store_context = state.get("store_context", False)
        friendly_chat = state.get("friendly_chat", False)
        
        logger.info(f"[ORCHESTRATOR] 🔀 Routing: run_task={run_task}, store_context={store_context}, friendly_chat={friendly_chat}")
        
        if friendly_chat:
            logger.info("[ORCHESTRATOR] ➜ Route: friendly_chat_response")
            return "friendly_chat_response"
        
        elif store_context:
            logger.info("[ORCHESTRATOR] ➜ Route: save_context")
            return "save_context"
        
        else:  # run_task
            logger.info("[ORCHESTRATOR] ➜ Route: create_tasks")
            return "create_tasks"


       

    async def send_error_response_node(self, state: OrchestratorState):
        """
        Terminal node: Send error response without executing tasks
        """
        error_msg = state.get("error_message", "An error occurred")
        logger.warning(f"[ORCHESTRATOR] Terminating with error: {error_msg}")
        
        state["final_response"] = error_msg
        state["status"] = "error"
        
        return state
    
    async def _load_conversation_history(self) -> list:
        """Load conversation history from Redis"""
        try:
            import json
            
            history_key = "conversation_history"  # Simple fixed key
            
            history_json = await self.mq_service.redis_client.get(history_key)
            
            if history_json:
                # ✅ FIX: Redis returns bytes, decode to string first
                if isinstance(history_json, bytes):
                    history_json = history_json.decode('utf-8')
                
                history = json.loads(history_json)
                logger.info(f"[ORCHESTRATOR] ✅ Loaded {len(history)} messages from Redis")
                return history
            
            logger.info("[ORCHESTRATOR] No previous history found in Redis")
            return []
            
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] Could not load history from Redis: {e}")
            return []
        

    async def friendly_chat_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Handle friendly chat requests
        ✅ NEW NODE: Returns generated response without executing any tasks
        """
        logger.info("[ORCHESTRATOR] 🤖 Handling friendly chat request")
        
        response = state.get("response", "Hello! How can I help you today?")
        
        state["final_response"] = response
        state["status"] = "friendly_chat_completed"
        
        # Add to conversation history
        state["conversation_history"].append({
            "role": "user",
            "content": state["user_input"]
        })
        state["conversation_history"].append({
            "role": "assistant",
            "content": response
        })
        
        state["execution_trace"].append({
            "node": "friendly_chat_response",
            "timestamp": datetime.now().isoformat(),
            "response_preview": response[:100] + "..." if len(response) > 100 else response
        })
        
        logger.info(f"[ORCHESTRATOR] ✅ Friendly chat completed")
        
        return state


    async def save_context_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Store context to database
        ✅ NEW NODE: Uses existing store_context_to_db function
        """
        logger.info("[ORCHESTRATOR] 💾 Storing context to database")
        
        context_text = state.get("context", "")
        agent_type = state.get("detected_agents", ["email"])[0] if state.get("detected_agents") else "email"
        
        if not context_text or context_text.strip() == "":
            logger.warning("[ORCHESTRATOR] ⚠️  No context text provided")
            state["final_response"] = "❌ No context text provided to store"
            state["status"] = "context_storage_failed"
            return state
        
        try:
            from .nodes import store_context_to_db
            
            # Create synthetic LLM response with CONTEXT_SUMMARY section
            synthetic_response = f"""**CONTEXT_SUMMARY:**
    {context_text}"""
            
            # Call existing function
            store_context_to_db(
                state=state,
                llm_response=synthetic_response,
                agent_type=agent_type
            )
            
            response = f"✅ Context stored successfully for {agent_type} agent"
            state["final_response"] = response
            state["status"] = "context_stored"
            
            logger.info(f"[ORCHESTRATOR] ✅ Context stored successfully")
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] ❌ Failed to store context: {e}", exc_info=True)
            state["final_response"] = f"❌ Failed to store context: {str(e)}"
            state["execution_errors"]["context_storage"] = str(e)
            state["status"] = "context_storage_failed"
        
        # Add to conversation history
        state["conversation_history"].append({
            "role": "user",
            "content": state["user_input"]
        })
        state["conversation_history"].append({
            "role": "assistant",
            "content": state["final_response"]
        })
        
        state["execution_trace"].append({
            "node": "store_context",
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "status": state["status"]
        })
        
        return state


    async def _save_conversation_history(self, conversation_history: list):
        """Save conversation history to Redis"""
        try:
            import json
            
            history_key = "conversation_history"  # Simple fixed key
            
            history_json = json.dumps(conversation_history)
            
            # Store with 24-hour expiry
            await self.mq_service.redis_client.setex(
                history_key,
                86400,  # 24 hours
                history_json
            )
            
            logger.info(f"[ORCHESTRATOR] Saved {len(conversation_history)} messages to Redis")
            
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] Could not save history to Redis: {e}")

  
    async def orchestrate(
        self,
        user_input: str,
        # user_id: str,
        # session_id: str,
        conversation_history: Optional[list] = None
    ) -> OrchestratorState:
        """Execute orchestration flow
        
        Args:
            user_input: User's natural language request
            user_id: User identifier
            session_id: Session identifier
            conversation_history: Previous conversation for context
        
        Returns:
            Final orchestrator state with results
        """
        # logger.info(f"[ORCHESTRATOR] Starting orchestration for user {user_id}")
        
        try:
            if conversation_history is None:
                conversation_history = await self._load_conversation_history()
            # ✅ Create initial state (dict structure)
            # Since OrchestratorState is TypedDict, we work with dicts directly
            initial_state: OrchestratorState = {
                "user_input": user_input,
                # "user_id": user_id,
                # "session_id": session_id,
                "conversation_history": conversation_history,
                "run_task": True,
                "store_context": False,
                "friendly_chat": False,
                "response": "",
                "context": "",
                "user_intent": "",
                "detected_agents": [],
                "required_actions": [],
                "tasks": [],
                "task_metadata": {
                    "task_id": str(uuid.uuid4()),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "timestamp": datetime.now()
                },
                "insufficient_context":False,  
                "has_critical_error":False,    
                "error_message":None,
                "agent_responses": {},
                "queue_status": {},
                "execution_errors": {},
                "execution_trace": [],
                "final_response": ""
            }
            
            logger.info(f"[ORCHESTRATOR] Initial state created: user_input='{user_input}'")
            # logger.info(f"[ORCHESTRATOR] Session: {session_id}")
            
            # ✅ Invoke graph with dict (TypedDict is just a dict at runtime)
            final_state_dict = await self.compiled_graph.ainvoke(initial_state)
            
            await self._save_conversation_history(final_state_dict["conversation_history"])
            
            logger.info(f"[ORCHESTRATOR] Graph execution complete")
            logger.info(f"[ORCHESTRATOR] Final response: {final_state_dict.get('final_response', '')[:100]}...")
            
            # ✅ Return dict as OrchestratorState (TypedDict is just a dict)
            return final_state_dict
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Orchestration failed: {e}", exc_info=True)
            raise   


