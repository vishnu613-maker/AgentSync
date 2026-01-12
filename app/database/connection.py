"""
Database connection management for SQLite, ChromaDB, and Redis
Includes connection pooling and health checks
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import chromadb
from chromadb.config import Settings as ChromaSettings
import redis
from typing import Generator
import logging
from sqlalchemy import text 

from app.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Load settings
settings = get_settings()

# ============= SQLite Configuration =============
# Create engine with connection pooling
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    poolclass=StaticPool,  # Use static pool for SQLite
    echo=settings.debug  # Log SQL queries in debug mode
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session
    Use with FastAPI Depends()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============= ChromaDB Configuration =============
try:
    chroma_client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Create default collection for agent contexts
    try:
        context_collection = chroma_client.get_or_create_collection(
            name="agent_contexts",
            metadata={"description": "Stores agent context embeddings"}
        )
        logger.info("ChromaDB initialized successfully")
    except Exception as e:
        logger.warning(f"ChromaDB collection creation failed: {e}")
        context_collection = None
        
except Exception as e:
    logger.error(f"ChromaDB initialization failed: {e}")
    chroma_client = None
    context_collection = None


def get_chroma_client():
    """Get ChromaDB client instance"""
    if chroma_client is None:
        raise ConnectionError("ChromaDB is not initialized")
    return chroma_client


def get_context_collection():
    """Get the agent contexts collection"""
    if context_collection is None:
        raise ConnectionError("ChromaDB context collection not available")
    return context_collection


# ============= Redis Configuration =============
try:
    redis_client = redis.from_url(
        settings.redis_url,
        decode_responses=True,  # Automatically decode responses to strings
        socket_connect_timeout=5,
        socket_timeout=5
    )
    
    # Test connection
    redis_client.ping()
    logger.info("Redis connected successfully")
    
except redis.ConnectionError as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None


def get_redis_client():
    """Get Redis client instance"""
    if redis_client is None:
        raise ConnectionError("Redis is not initialized")
    return redis_client


# ============= Connection Health Checks =============
def check_database_health() -> dict:
    """
    Check health of all database connections
    Returns dict with status of each service
    """
    health_status = {
        "sqlite": False,
        "chromadb": False,
        "redis": False
    }
    
    # Check SQLite
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))  # ✅ Fixed syntax
        db.close()
        health_status["sqlite"] = True
    except Exception as e:
        logger.error(f"SQLite health check failed: {e}")
    
    # Check ChromaDB
    try:
        if chroma_client:
            chroma_client.heartbeat()
            health_status["chromadb"] = True
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
    
    # Check Redis
    try:
        if redis_client:
            redis_client.ping()
            health_status["redis"] = True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
    
    return health_status


# ============= Cleanup on Shutdown =============
def close_db_connections():
    """Close all database connections gracefully"""
    try:
        if redis_client:
            redis_client.close()
            logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis: {e}")
    
    try:
        engine.dispose()
        logger.info("SQLite connections closed")
    except Exception as e:
        logger.error(f"Error closing SQLite: {e}")


# ============= Export for Dependency Injection =============  # ← NEW
# from app.services.context_db_service import get_context_db_service  # ← NEW

__all__ = ["engine", "SessionLocal", "Base", "get_db", "get_context_db_service"]  # ← NEW

