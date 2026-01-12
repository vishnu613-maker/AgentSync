"""
LangGraph State management for orchestrator
Defines the state structure that flows through the graph
"""
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TaskMetadata:
    """Metadata for a task"""
    task_id: str
    user_id: str
    timestamp: datetime
    session_id: str


@dataclass
class AgentTask:
    """Task to be executed by an agent"""
    agent_name: str  # "email", "calendar", "slack"
    action: str  # e.g., "send_email", "create_event"
    parameters: Dict[str, Any]
    task_id: str = ""  # Will be generated
    status: str = "pending"  # pending, queued, processing, completed, failed


class OrchestratorState(TypedDict):
    """
    State that flows through LangGraph orchestrator
    Think of this as the context that travels through each node
    """
    # Input
    user_input: str
    # user_id: str
    # session_id: str
    
    # Conversation history (for context)
    conversation_history: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    
    run_task: bool                    # Whether to execute task (default True)
    store_context: bool               # Whether to store context to DB (default False)
    friendly_chat: bool               # Whether this is friendly chat (default False)
    response: str                     # Response text for friendly chat (default "")
    context: str
    
    # Analysis phase
    detected_agents: List[str]  # Which agents need to be called: ["email", "calendar"]
    user_intent: str  # What the user wants to accomplish
    required_actions: List[str]  # Breakdown of actions needed
    
    # Orchestration phase
    tasks: List[AgentTask]  # Tasks to queue
    queue_status: Dict[str, str]  # Status of tasks in queue: {"task_123": "queued"}
    
    # Execution phase
    agent_responses: Dict[str, Any]  # Results from agents: {"email": {...}, "calendar": {...}}
    execution_errors: Dict[str, str]  # Any errors: {"email": "Failed to send"}
    
    # Aggregation phase
    final_response: str  # Response to user
    task_metadata: TaskMetadata
    
    # Logging
    execution_trace: List[Dict[str, Any]]  # For debugging
    
    # Error handling
    insufficient_context: bool
    has_critical_error: bool
    error_message: Optional[str]
    


# Helper function to create initial state
def create_initial_state(
    user_input: str,
    user_id: str,
    session_id: str,
    conversation_history: List[Dict[str, str]] = None
) -> OrchestratorState:
    """Create initial state for orchestrator"""
    return {
        "user_input": user_input,
        "user_id": user_id,
        "session_id": session_id,
        "conversation_history": conversation_history or [],
        "detected_agents": [],
        "user_intent": "",
        "required_actions": [],
        "tasks": [],
        "queue_status": {},
        "agent_responses": {},
        "execution_errors": {},
        "final_response": "",
        "task_metadata": TaskMetadata(
            task_id="",
            user_id=user_id,
            timestamp=datetime.now(),
            session_id=session_id
        ),
        "insufficient_context":False,  
        "has_critical_error":False,    
        "error_message":None,
        "execution_trace": [],
    }
