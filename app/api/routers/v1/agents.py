"""
Agent endpoints - REST API for agent operations
✅ UPDATED: Merged old endpoints with new LangGraph orchestrator

REMOVED: /agents/{agent_id}/execute, /agents/{agent_id}/status
ADDED: /api/v1/execute_task, /api/v1/task_status/{task_id}
KEPT: /agents/list, /agents/{agent_id}/tools, /agents/context/request
"""
import logging
from fastapi import APIRouter, Request, HTTPException, status, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from app.agents.registry import agent_registry
from app.agents.email_agent import EmailAgent
from app.agents.calendar_agent import CalendarAgent
from app.agents.tasks_agent import TasksAgent
from app.database.connection import SessionLocal
from app.database.models import Agent
from app.orchestrator.graph import AgentOrchestratorGraph
from app.services.message_queue import MessageQueueService
from app.services.llm_service import LLMService
# from app.services.context_manager import ContextManager
from app.config import get_settings

logger = logging.getLogger(__name__)

# Create separate routers for organization
agents_router = APIRouter(prefix="/agents", tags=["agents"])
tasks_router = APIRouter(prefix="/api/v1", tags=["tasks"])

security = HTTPBearer()


# ============================================================================
# Pydantic Models
# ============================================================================

class TaskRequest(BaseModel):
    """User request for task execution via orchestrator"""
    user_input: str
    # user_id: str
    # session_id: Optional[str] = None
    # conversation_context: Optional[List[dict]] = None


class TaskResponse(BaseModel):
    """Response from orchestrator"""
    task_id: str
    status: str
    response: str
    agents_involved: List[str]
    timestamp: str


class ToolSchema(BaseModel):
    """Tool definition schema"""
    name: str
    description: str
    inputSchema: Optional[Dict[str, Any]] = None


class AgentInfo(BaseModel):
    """Agent information"""
    agent_id: int
    name: str
    type: str
    description: str
    status: str
    user_id: int
    tools: Optional[List[ToolSchema]] = None


class AgentsListResponse(BaseModel):
    """Response for agents list"""
    agents: List[AgentInfo]
    count: int


class ContextRequest(BaseModel):
    """Context exchange request"""
    target_agent_id: int
    query: str
    source_agent_id: Optional[int] = None


# ============================================================================
# AGENTS ENDPOINTS (Original endpoints - Updated)
# ============================================================================


# ✅ KEPT: List all agents endpoint
@agents_router.get("/list", response_model=AgentsListResponse)
async def list_agents(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    List all agents for current user
    ✅ KEPT: Returns ALL agents (email, calendar, slack)
    Protected endpoint - requires authentication
    
    Args:
        request: FastAPI request with user context
        credentials: Bearer token for authentication
    
    Returns:
        AgentsListResponse with all available agents
    """
    try:
        # Get user from request state (set by auth middleware)
        if not hasattr(request.state, "user"):
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        user_id = request.state.user.get("user_id", 1)
        
        logger.info(f"📋 Listing agents for user {user_id}")
        
        # Get ALL agents from registry
        all_agents = agent_registry.get_all_agents()
        
        # Filter by user_id
        user_agents = [
            agent for agent in all_agents 
            if agent.user_id == user_id
        ]
        
        logger.info(f"✅ Found {len(user_agents)} agents")
        
        # Format response with tools
        agents_data = []
        for agent in user_agents:
            agent_info = agent.get_info()
            
            # Get tools
            tools = []
            if hasattr(agent, 'tools'):
                tools = agent.tools
            
            agents_data.append({
                "agent_id": agent.agent_id,
                "name": agent.name,
                "type": agent.agent_type,
                "description": agent_info.get("description", ""),
                "status": agent_info.get("status", "idle"),
                "user_id": agent.user_id,
                "tools": tools
            })
        
        return AgentsListResponse(
            agents=agents_data,
            count=len(agents_data)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to list agents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list agents: {str(e)}"
        )


# ✅ KEPT: Get agent tools endpoint
@agents_router.get("/{agent_id}/tools")
async def get_agent_tools(
    request: Request,
    agent_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get available tools for agent
    ✅ KEPT: Returns tools that agent can execute
    Protected endpoint - requires authentication
    
    Args:
        request: FastAPI request with user context
        agent_id: ID of the agent
        credentials: Bearer token for authentication
    
    Returns:
        Dictionary with agent_id, tools list, and count
    """
    try:
        if not hasattr(request.state, "user"):
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        user_id = request.state.user.get("user_id", 1)
        
        logger.info(f"🛠️ Getting tools for agent {agent_id}")
        
        # Get agent from registry
        agent = agent_registry.get_agent(agent_id)
        
        if agent is None:
            logger.warning(f"⚠️ Agent {agent_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        # Check authorization
        if agent.user_id != user_id:
            logger.warning(f"⚠️ Unauthorized access to agent {agent_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Unauthorized"
            )
        
        # Get tools
        tools = agent.tools if hasattr(agent, 'tools') else []
        
        logger.info(f"✅ Retrieved {len(tools)} tools for agent {agent_id}")
        
        return {
            "agent_id": agent_id,
            "tools": tools,
            "count": len(tools)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get agent tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent tools: {str(e)}"
        )


# ✅ KEPT: Request context from agent endpoint
@agents_router.post("/context/request")
async def request_context(
    request: Request,
    context_req: ContextRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Request context from another agent (federated context exchange)
    ✅ KEPT: For inter-agent communication
    Protected endpoint - requires authentication
    
    Args:
        request: FastAPI request with user context
        context_req: Context request with target_agent_id and query
        credentials: Bearer token for authentication
    
    Returns:
        Context data from target agent
    """
    try:
        if not hasattr(request.state, "user"):
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        user_id = request.state.user.get("user_id", 1)
        source_agent_id = context_req.source_agent_id or None
        
        logger.info(f"🔄 Requesting context from agent {context_req.target_agent_id}")
        logger.info(f"   Query: {context_req.query}")
        
        # Request context via orchestrator
        result = await orchestrator.request_context_from_agent(
            target_agent_id=context_req.target_agent_id,
            query=context_req.query,
            user_id=user_id,
            source_agent_id=source_agent_id
        )
        
        logger.info(f"✅ Context request completed")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to request context: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to request context: {str(e)}"
        )


# ============================================================================
# TASK ENDPOINTS (New orchestrator-based endpoints)
# ============================================================================

@tasks_router.post("/execute_task", response_model=TaskResponse)
async def execute_task(
    request_body: TaskRequest,
    background_tasks: BackgroundTasks,
):
   
    
    try:
        settings = get_settings()
        # logger.info(f"[API] Execute task request: {request_body.user_input}")
        # logger.info(f"[API] User: {request_body.user_id}, Session: {request_body.session_id}")
        
        # Initialize services
        mq_service = MessageQueueService(settings.redis_url)
        llm_service = LLMService(settings.ollama_url)
        
        # Connect to Redis
        await mq_service.connect()
        logger.info("[API] Connected to message queue")
        
        # Check Ollama health
        ollama_health = await llm_service.health_check()
        if not ollama_health:
            logger.error("[API] Ollama health check failed")
            raise HTTPException(
                status_code=503,
                detail="Ollama service unavailable"
            )
        logger.info("[API] Ollama health check passed")
        
        # Create orchestrator instance
        orchestrator_graph = AgentOrchestratorGraph(mq_service, llm_service)
        logger.info("[API] Orchestrator graph created")
        
        # Generate IDs
        task_id = str(uuid.uuid4())
        # session_id = request_body.session_id or str(uuid.uuid4())
        
        logger.info(f"[API] Starting orchestration - Task ID: {task_id}")
        
        # Execute orchestration (this is the main flow)
        final_state = await orchestrator_graph.orchestrate(
            user_input=request_body.user_input,
            # user_id=request_body.user_id,
            # session_id=session_id,
            # conversation_history=request_body.conversation_context or []
        )
        
        logger.info(f"[API] Orchestration completed successfully")
        logger.info(f"[API] Agents involved: {final_state['detected_agents']}")
        logger.info(f"[API] Final response: {final_state['final_response']}")
        
        # Prepare response
        response = TaskResponse(
            task_id=task_id,
            status="success",
            response=final_state["final_response"],
            agents_involved=final_state["detected_agents"],
            timestamp=final_state["task_metadata"].get("created_at", datetime.now().isoformat())
        )
        
        logger.info(f"[API] Sending response to user")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (like Ollama unavailable)
        raise
        
    except Exception as e:
        logger.error(f"[API] Error executing task: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Task execution failed: {str(e)}"
        )


@tasks_router.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get status of a previously executed task
    
    ✅ NEW: Check status of orchestrated tasks
    
    Args:
        task_id: Task ID returned from /execute_task
    
    Returns:
        Dictionary with:
            - task_id: Task identifier
            - status: "processing", "success", "failure", "completed", "timeout"
            - result: Full result data if available
    
    Example:
        GET /api/v1/task_status/abc-123-def-456
        
        Response (if processing):
        {
            "task_id": "abc-123-def-456",
            "status": "processing"
        }
        
        Response (if completed):
        {
            "task_id": "abc-123-def-456",
            "status": "success",
            "result": {
                "status": "success",
                "agent": "email",
                "data": {...},
                "timestamp": "2025-11-04T14:30:00"
            }
        }
    """
    
    try:
        logger.info(f"[API] Getting status for task {task_id}")
        
        settings = get_settings()
        mq_service = MessageQueueService(settings.redis_url)
        await mq_service.connect()
        
        # Try to retrieve result from Redis
        result = await mq_service.get_result(task_id)
        
        if result:
            logger.info(f"[API] Task {task_id} result found: {result.get('status')}")
            return {
                "task_id": task_id,
                "status": result.get("status", "unknown"),
                "result": result
            }
        else:
            logger.info(f"[API] Task {task_id} still processing")
            return {
                "task_id": task_id,
                "status": "processing",
                "message": "Task is still being processed. Check back in a few seconds."
            }
        
    except Exception as e:
        logger.error(f"[API] Error getting task status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )


# ============================================================================
# Include routers in FastAPI app
# ============================================================================

def setup_endpoints(app):
    """
    Call this function in your main.py to include all endpoints
    
    Example in main.py:
        from app.api.endpoints import setup_endpoints
        
        app = FastAPI()
        setup_endpoints(app)
    """
    app.include_router(agents_router)
    app.include_router(tasks_router)
    
    logger.info("✅ All endpoints registered")
    logger.info("   Old endpoints (kept):")
    logger.info("     - GET /agents/list")
    logger.info("     - GET /agents/{agent_id}/tools")
    logger.info("     - POST /agents/context/request")
    logger.info("     - GET /agents/debug/zapier-tools")
    logger.info("   New endpoints (orchestrator):")
    logger.info("     - POST /api/v1/execute_task")
    logger.info("     - GET /api/v1/task_status/{task_id}")
    logger.info("   Removed endpoints:")
    logger.info("     - POST /agents/{agent_id}/execute (REMOVED)")
    logger.info("     - GET /agents/{agent_id}/status (REMOVED)")
