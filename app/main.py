"""
AgentSync API - Main Application Entry Point
✅ FIXED: Worker service integration + proper shutdown
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.middleware import (
    LoggingMiddleware,
    ErrorHandlerMiddleware,
    RateLimitMiddleware,
    AuthenticationMiddleware,
)
from app.config import get_settings
from app.agents.registry import agent_registry
from app.api.routers import health, v1
from app.api.routers.v1.agents import setup_endpoints
from app.services.message_queue import MessageQueueService
from app.agents.email_agent import EmailAgent
from app.agents.calendar_agent import CalendarAgent
from app.agents.tasks_agent import TasksAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
_mq_service: Optional[MessageQueueService] = None
_worker_service = None  # Will be set in lifespan


# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown"""
    
    global _mq_service, _worker_service
    
    logger.info("=" * 80)
    logger.info("[LIFESPAN] 🚀 AgentSync v2.0 Starting up...")
    logger.info("=" * 80)
    
    # Startup
    try:
        settings = get_settings()
        
        logger.info(f"[LIFESPAN] Environment: {settings.environment}")
        logger.info(f"[LIFESPAN] Ollama URL: {settings.ollama_url}")
        logger.info(f"[LIFESPAN] Redis URL: {settings.redis_url}")
        
        # ========================================================================
        # 1. Initialize Message Queue
        # ========================================================================
        
        logger.info("[LIFESPAN] Initializing message queue...")
        _mq_service = MessageQueueService(settings.redis_url)
        
        try:
            await _mq_service.connect()
            logger.info("[LIFESPAN] ✅ Redis connected")
        except Exception as e:
            logger.error(f"[LIFESPAN] ❌ Failed to connect to Redis: {e}")
            raise
        
        # Check Redis health
        try:
            redis_health = await _mq_service.health_check()
            if redis_health:
                logger.info("[LIFESPAN] ✅ Redis health check passed")
            else:
                logger.error("[LIFESPAN] ❌ Redis health check failed")
        except Exception as e:
            logger.error(f"[LIFESPAN] ⚠️ Redis health check error: {e}")
        
        # ========================================================================
        # 2. Initialize Agents
        # ========================================================================
        
        logger.info("=" * 80)
        logger.info("[LIFESPAN] 📦 Initializing agents...")
        logger.info("=" * 80)
        
        try:
            # Import agents
            logger.info("[LIFESPAN] Step 1: Importing agent classes...")
            logger.info("[LIFESPAN] ✅ EmailAgent imported")
            logger.info("[LIFESPAN] ✅ CalendarAgent imported")
            logger.info("[LIFESPAN] ✅ SlackAgent imported")
            
            # Create agents
            logger.info("[LIFESPAN] Step 2: Creating agent instances...")
            
            email_agent = EmailAgent(
                agent_id=1,
                name="EmailAgent",
                mq_service=_mq_service
            )
            logger.info(f"[LIFESPAN] ✅ EmailAgent created")
            logger.info(f"[LIFESPAN]    - ID: {email_agent.agent_id}")
            logger.info(f"[LIFESPAN]    - Type: {email_agent.agent_type}")
            logger.info(f"[LIFESPAN]    - Tools: {len(email_agent.tools)}")
            
            calendar_agent = CalendarAgent(
                agent_id=2,
                name="CalendarAgent",
                mq_service=_mq_service
            )
            logger.info(f"[LIFESPAN] ✅ CalendarAgent created")
            logger.info(f"[LIFESPAN]    - ID: {calendar_agent.agent_id}")
            logger.info(f"[LIFESPAN]    - Type: {calendar_agent.agent_type}")
            logger.info(f"[LIFESPAN]    - Tools: {len(calendar_agent.tools)}")
            
            tasks_agent = TasksAgent(
                agent_id=3,
                name="TasksAgent",
                mq_service=_mq_service
            )
            logger.info(f"[LIFESPAN] ✅ TasksAgent created")
            logger.info(f"[LIFESPAN]    - ID: {tasks_agent.agent_id}")
            logger.info(f"[LIFESPAN]    - Type: {tasks_agent.agent_type}")
            logger.info(f"[LIFESPAN]    - Tools: {len(tasks_agent.tools)}")
            
            # Register agents
            logger.info("[LIFESPAN] Step 3: Registering agents in registry...")
            
            agent_registry.register(email_agent)
            logger.info("[LIFESPAN] ✅ EmailAgent registered")
            
            agent_registry.register(calendar_agent)
            logger.info("[LIFESPAN] ✅ CalendarAgent registered")
            
            agent_registry.register(tasks_agent)
            logger.info("[LIFESPAN] ✅ SlackAgent registered")
            
            # Verify
            logger.info("[LIFESPAN] Step 4: Verifying registration...")
            all_agents = agent_registry.get_all_agents()
            logger.info(f"[LIFESPAN] ✅ Total agents in registry: {len(all_agents)}")
            
            for i, agent in enumerate(all_agents, 1):
                try:
                    info = agent.get_info()
                    logger.info(f"[LIFESPAN]    [{i}] {info['name']}")
                    logger.info(f"[LIFESPAN]        - Agent ID: {info['agent_id']}")
                    logger.info(f"[LIFESPAN]        - Type: {info['type']}")
                    logger.info(f"[LIFESPAN]        - Status: {info['status']}")
                    logger.info(f"[LIFESPAN]        - Tools: {info['tools_count']}")
                except Exception as e:
                    logger.error(f"[LIFESPAN] ❌ Error getting info for agent {i}: {e}")
            
        except Exception as e:
            logger.error(f"[LIFESPAN] ❌ Agent initialization failed: {e}", exc_info=True)
            raise
        
        # ========================================================================
        # 3. Start Agent Workers ✅ NEW
        # ========================================================================
        
        logger.info("=" * 80)
        logger.info("[LIFESPAN] 🔄 Starting agent workers...")
        logger.info("=" * 80)
        
        try:
            from app.agents.worker_service import get_worker_service
            _worker_service = await get_worker_service(_mq_service)
            await _worker_service.start_workers()
            logger.info("[LIFESPAN] ✅ Agent workers started successfully")
        except Exception as e:
            logger.error(f"[LIFESPAN] ❌ Failed to start workers: {e}", exc_info=True)
            raise
        
        logger.info("=" * 80)
        logger.info("[LIFESPAN] 🎉 AgentSync Ready!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"[LIFESPAN] ❌ Startup error: {e}", exc_info=True)
        raise
    
    yield
    
    # ========================================================================
    # SHUTDOWN
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("[LIFESPAN] 🛑 Shutting down...")
    logger.info("=" * 80)
    
    try:
        # Stop workers first
        if _worker_service:
            logger.info("[LIFESPAN] Stopping agent workers...")
            try:
                await _worker_service.stop_workers()
                logger.info("[LIFESPAN] ✅ Agent workers stopped")
            except Exception as e:
                logger.error(f"[LIFESPAN] ⚠️ Error stopping workers: {e}")
        
        # Disconnect Redis
        if _mq_service:
            logger.info("[LIFESPAN] Disconnecting Redis...")
            try:
                await _mq_service.disconnect()
                logger.info("[LIFESPAN] ✅ Redis disconnected")
            except Exception as e:
                logger.error(f"[LIFESPAN] ⚠️ Error disconnecting Redis: {e}")
        
        # Deregister agents
        logger.info("[LIFESPAN] Deregistering agents...")
        try:
            all_agents = agent_registry.get_all_agents()
            for agent in all_agents:
                agent_registry.deregister(agent.agent_id)
            logger.info(f"[LIFESPAN] ✅ Deregistered {len(all_agents)} agents")
        except Exception as e:
            logger.error(f"[LIFESPAN] ⚠️ Error deregistering agents: {e}")
        
        logger.info("[LIFESPAN] ✅ Shutdown complete")
    except Exception as e:
        logger.error(f"[LIFESPAN] ❌ Shutdown error: {e}", exc_info=True)


# ============================================================================
# FASTAPI APP CREATION
# ============================================================================

app = FastAPI(
    title="AgentSync",
    description="Multi-Agent Orchestration and Context Management System",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# ============================================================================
# OPENAPI SECURITY CONFIGURATION
# ============================================================================

def custom_openapi():
    """Configure OpenAPI security scheme for Swagger UI"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="AgentSync",
        version="0.4.0",
        description="Multi-Agent Orchestration and Context Management System",
        routes=app.routes,
    )
    
    # Add security scheme for Bearer tokens
    openapi_schema["components"]["securitySchemes"] = {
        "HTTPBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your JWT token"
        }
    }
    
    # Apply security globally
    openapi_schema["security"] = [{"HTTPBearer": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ============================================================================
# MIDDLEWARE
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuthenticationMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LoggingMiddleware)


# ============================================================================
# SETUP ENDPOINTS
# ============================================================================

app.include_router(health.router)
app.include_router(v1.auth.router)
setup_endpoints(app)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "AgentSync API is running",
        "version": "0.4.0",
        "status": "active"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    settings = get_settings()  # ✅ FIXED: Get settings here
    return {
        "status": "healthy",
        "version": "0.4.0",
        "mode": settings.environment
    }


@app.get("/config/info")
def config_info():
    """Get non-sensitive configuration information"""
    settings = get_settings()  # ✅ FIXED: Get settings here
    all_agents = agent_registry.get_all_agents()
    
    return {
        "environment": settings.environment,
        "llm_mode": settings.llm_mode,
        "mcp_enabled": bool(settings.ZAPIER_MCP_SERVER_URL),
        "agents": {
            "email": True,
            "calendar": True,
            "slack": True,
            "total_initialized": len(all_agents)
        },
        "agents_initialized": [
            {
                "id": agent.agent_id,
                "name": agent.name,
                "type": agent.agent_type,
                # "user_id": agent.user_id,
                "tools_count": len(agent.tools)
            }
            for agent in all_agents
        ]
    }


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"[APP] HTTP Exception {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"[APP] Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "status_code": 500,
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    settings = get_settings()
    
    logger.info("Starting AgentSync FastAPI application...")
    logger.info(f"  Host: 0.0.0.0")
    logger.info(f"  Port: 8000")
    logger.info(f"  Reload: {settings.debug}")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )
