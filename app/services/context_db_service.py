"""
Context Database Service for AgentSync
Manages SQLite operations for AgentContext and ContextRequests tables
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import json

from app.database.models import AgentContext, ContextRequest, Agent
from app.database.connection import SessionLocal

logger = logging.getLogger(__name__)


class ContextDBService:
    """
    Service for managing agent context data in SQLite
    Handles CRUD operations for AgentContext and ContextRequests tables
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize context database service
        
        Args:
            db_session: SQLAlchemy session (uses SessionLocal if not provided)
        """
        self.db = db_session or SessionLocal()
        logger.info("[CONTEXT_DB] ContextDBService initialized")
    
    
    # ==================== AgentContext Operations ====================
    
    def insert_agent_context(
        self,
        agent_id: int,
        context_summary: str,
        vector_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        context_key: Optional[str] = None,
        relevance_score: float = 1.0,
        expires_at: Optional[datetime] = None
    ) -> Optional[AgentContext]:
        """
        Insert new context into AgentContext table
        
        Args:
            agent_id: ID of agent creating context
            context_summary: Human-readable summary
            vector_id: ID from ChromaDB embedding
            metadata: Additional metadata as dict
            context_key: Optional identifier for context
            relevance_score: Relevance score (0-1)
            expires_at: Optional expiration time
            
        Returns:
            AgentContext object or None if failed
        """
        try:
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Create new context record
            new_context = AgentContext(
                agent_id=agent_id,
                context_summary=context_summary,
                embedding_id=vector_id,
                metadata_json=metadata_json,
                context_key=context_key or f"ctx_{agent_id}_{vector_id[:8]}",
                relevance_score=relevance_score,
                expires_at=expires_at,
                created_at=datetime.utcnow()
            )
            
            self.db.add(new_context)
            self.db.commit()
            self.db.refresh(new_context)
            
            logger.info(f"[CONTEXT_DB] Inserted context: {new_context.id} (agent: {agent_id})")
            
            return new_context
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"[CONTEXT_DB] Failed to insert context: {e}", exc_info=True)
            return None
    
    
    def get_agent_contexts(
        self,
        agent_id: int,
        limit: int = 10,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent contexts for a specific agent
        
        Args:
            agent_id: Agent ID to retrieve contexts for
            limit: Maximum number of contexts to return
            include_expired: Whether to include expired contexts
            
        Returns:
            List of context dictionaries
        """
        try:
            query = self.db.query(AgentContext).filter(
                AgentContext.agent_id == agent_id
            )
            
            # Filter out expired contexts if requested
            if not include_expired:
                query = query.filter(
                    (AgentContext.expires_at.is_(None)) | 
                    (AgentContext.expires_at > datetime.utcnow())
                )
            
            # Order by creation date (newest first)
            contexts = query.order_by(
                desc(AgentContext.created_at)
            ).limit(limit).all()
            
            # Format results
            results = []
            for ctx in contexts:
                result = {
                    "id": ctx.id,
                    "agent_id": ctx.agent_id,
                    "context_key": ctx.context_key,
                    "context_summary": ctx.context_summary,
                    "embedding_id": ctx.embedding_id,
                    "metadata": json.loads(ctx.metadata_json) if ctx.metadata_json else {},
                    "relevance_score": ctx.relevance_score,
                    "created_at": ctx.created_at.isoformat() if ctx.created_at else None,
                    "expires_at": ctx.expires_at.isoformat() if ctx.expires_at else None
                }
                results.append(result)
            
            logger.info(f"[CONTEXT_DB] Retrieved {len(results)} contexts for agent {agent_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"[CONTEXT_DB] Failed to get agent contexts: {e}", exc_info=True)
            return []
    
    
    def get_context_by_vector_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific context by ChromaDB vector ID
        
        Args:
            vector_id: ChromaDB embedding ID
            
        Returns:
            Context dictionary or None if not found
        """
        try:
            context = self.db.query(AgentContext).filter(
                AgentContext.embedding_id == vector_id
            ).first()
            
            if context:
                return {
                    "id": context.id,
                    "agent_id": context.agent_id,
                    "context_key": context.context_key,
                    "context_summary": context.context_summary,
                    "embedding_id": context.embedding_id,
                    "metadata": json.loads(context.metadata_json) if context.metadata_json else {},
                    "relevance_score": context.relevance_score,
                    "created_at": context.created_at.isoformat() if context.created_at else None
                }
            
            logger.warning(f"[CONTEXT_DB] Context not found: {vector_id}")
            return None
            
        except Exception as e:
            logger.error(f"[CONTEXT_DB] Failed to get context by vector_id: {e}", exc_info=True)
            return None
    
    
    def get_context_by_id(self, context_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve context by ID
        
        Args:
            context_id: AgentContext table ID
            
        Returns:
            Context dictionary or None
        """
        try:
            context = self.db.query(AgentContext).filter(
                AgentContext.id == context_id
            ).first()
            
            if context:
                return {
                    "id": context.id,
                    "agent_id": context.agent_id,
                    "context_key": context.context_key,
                    "context_summary": context.context_summary,
                    "embedding_id": context.embedding_id,
                    "metadata": json.loads(context.metadata_json) if context.metadata_json else {},
                    "relevance_score": context.relevance_score,
                    "created_at": context.created_at.isoformat() if context.created_at else None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"[CONTEXT_DB] Failed to get context by ID: {e}", exc_info=True)
            return None
    
    
    def delete_agent_context(self, context_id: int) -> bool:
        """
        Delete specific context
        
        Args:
            context_id: AgentContext ID
            
        Returns:
            True if successful
        """
        try:
            context = self.db.query(AgentContext).filter(
                AgentContext.id == context_id
            ).first()
            
            if context:
                self.db.delete(context)
                self.db.commit()
                logger.info(f"[CONTEXT_DB] Deleted context: {context_id}")
                return True
            
            logger.warning(f"[CONTEXT_DB] Context not found: {context_id}")
            return False
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"[CONTEXT_DB] Failed to delete context: {e}", exc_info=True)
            return False
    
    
    def cleanup_expired_contexts(self) -> int:
        """
        Delete all expired contexts
        
        Returns:
            Number of contexts deleted
        """
        try:
            now = datetime.utcnow()
            
            expired_contexts = self.db.query(AgentContext).filter(
                and_(
                    AgentContext.expires_at.isnot(None),
                    AgentContext.expires_at < now
                )
            ).all()
            
            count = len(expired_contexts)
            
            for context in expired_contexts:
                self.db.delete(context)
            
            self.db.commit()
            
            logger.info(f"[CONTEXT_DB] Cleaned up {count} expired contexts")
            
            return count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"[CONTEXT_DB] Failed to cleanup expired contexts: {e}", exc_info=True)
            return 0
        
    def delete_by_embedding_id(self, embedding_id: str) -> None:
        """
        Delete single context by embedding_id from SQLite agent_contexts table
        
        Args:
            embedding_id: The vector ID (column name: embedding_id)
            
        Returns:
            None
            
        Raises:
            Exception: If deletion fails
        """
        try:
            context = self.db.query(AgentContext).filter(
                AgentContext.embedding_id == embedding_id
            ).first()
            
            if context:
                self.db.delete(context)
                self.db.commit()
                logger.info(f"[CONTEXT_DB] Deleted context by embedding_id: {embedding_id}")
            else:
                logger.warning(f"[CONTEXT_DB] Context not found with embedding_id: {embedding_id}")
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"[CONTEXT_DB] Failed to delete by embedding_id: {e}", exc_info=True)
            raise


    def delete_all_contexts(self) -> int:
        """
        Delete ALL contexts from SQLite agent_contexts table
        
        Returns:
            int: Number of contexts deleted
            
        Raises:
            Exception: If deletion fails
        """
        try:
            # Get count before deletion
            count = self.db.query(AgentContext).count()
            
            # Delete all
            self.db.query(AgentContext).delete()
            self.db.commit()
            
            logger.info(f"[CONTEXT_DB] Deleted all {count} contexts from SQLite")
            return count
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"[CONTEXT_DB] Failed to delete all contexts: {e}", exc_info=True)
            raise

    
    
    # ==================== ContextRequest Operations ====================
    
    def create_context_request(
        self,
        requesting_agent_id: int,
        responding_agent_id: int,
        query: str,
        request_type: str = "context_query"
    ) -> Optional[Dict[str, Any]]:
        """
        Create new context request (agent-to-agent communication)
        
        Args:
            requesting_agent_id: Agent ID making the request
            responding_agent_id: Agent ID that will respond
            query: The context query/request
            request_type: Type of request
            
        Returns:
            ContextRequest dictionary or None if failed
        """
        try:
            import uuid
            
            request_id = f"req_{requesting_agent_id}_{responding_agent_id}_{uuid.uuid4().hex[:8]}"
            
            new_request = ContextRequest(
                request_id=request_id,
                requester_id=requesting_agent_id,
                target_id=responding_agent_id,
                query=query,
                request_type=request_type,
                status="pending",
                created_at=datetime.utcnow()
            )
            
            self.db.add(new_request)
            self.db.commit()
            self.db.refresh(new_request)
            
            logger.info(f"[CONTEXT_DB] Created context request: {request_id}")
            
            return {
                "id": new_request.id,
                "request_id": new_request.request_id,
                "requester_id": new_request.requester_id,
                "target_id": new_request.target_id,
                "query": new_request.query,
                "request_type": new_request.request_type,
                "status": new_request.status,
                "created_at": new_request.created_at.isoformat()
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"[CONTEXT_DB] Failed to create context request: {e}", exc_info=True)
            return None
    
    
    def update_context_request(
        self,
        request_id: str,
        response_context_id: Optional[int] = None,
        response_summary: Optional[str] = None,
        status: str = "fulfilled",
        latency_ms: Optional[float] = None
    ) -> bool:
        """
        Update context request with response
        
        Args:
            request_id: ContextRequest ID
            response_context_id: ID of context provided in response
            response_summary: Summary of response
            status: Request status (fulfilled, timeout, error)
            latency_ms: Response time in milliseconds
            
        Returns:
            True if successful
        """
        try:
            request = self.db.query(ContextRequest).filter(
                ContextRequest.request_id == request_id
            ).first()
            
            if request:
                request.status = status
                request.response_context_id = response_context_id
                request.response_summary = response_summary
                request.latency_ms = latency_ms
                request.completed_at = datetime.utcnow()
                
                self.db.commit()
                
                logger.info(f"[CONTEXT_DB] Updated request {request_id}: {status}")
                
                return True
            
            logger.warning(f"[CONTEXT_DB] Request not found: {request_id}")
            return False
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"[CONTEXT_DB] Failed to update request: {e}", exc_info=True)
            return False
    
    
    def get_pending_requests(self, agent_id: int) -> List[Dict[str, Any]]:
        """
        Get pending context requests for an agent
        
        Args:
            agent_id: Agent ID to get pending requests for
            
        Returns:
            List of pending requests
        """
        try:
            requests = self.db.query(ContextRequest).filter(
                and_(
                    ContextRequest.target_id == agent_id,
                    ContextRequest.status == "pending"
                )
            ).order_by(
                desc(ContextRequest.created_at)
            ).all()
            
            results = []
            for req in requests:
                result = {
                    "id": req.id,
                    "request_id": req.request_id,
                    "requester_id": req.requester_id,
                    "query": req.query,
                    "request_type": req.request_type,
                    "created_at": req.created_at.isoformat()
                }
                results.append(result)
            
            logger.info(f"[CONTEXT_DB] Retrieved {len(results)} pending requests for agent {agent_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"[CONTEXT_DB] Failed to get pending requests: {e}", exc_info=True)
            return []
    
    
    def get_request_history(
        self,
        agent_id: int,
        limit: int = 50,
        role: str = "both"  # "requester", "responder", "both"
    ) -> List[Dict[str, Any]]:
        """
        Get context request history for analytics/debugging
        
        Args:
            agent_id: Agent ID
            limit: Maximum results
            role: Whether to show requests made by agent, received, or both
            
        Returns:
            List of requests
        """
        try:
            if role == "requester":
                requests = self.db.query(ContextRequest).filter(
                    ContextRequest.requester_id == agent_id
                )
            elif role == "responder":
                requests = self.db.query(ContextRequest).filter(
                    ContextRequest.target_id == agent_id
                )
            else:  # both
                from sqlalchemy import or_
                requests = self.db.query(ContextRequest).filter(
                    or_(
                        ContextRequest.requester_id == agent_id,
                        ContextRequest.target_id == agent_id
                    )
                )
            
            requests = requests.order_by(
                desc(ContextRequest.created_at)
            ).limit(limit).all()
            
            results = []
            for req in requests:
                result = {
                    "id": req.id,
                    "request_id": req.request_id,
                    "requester_id": req.requester_id,
                    "target_id": req.target_id,
                    "query": req.query,
                    "request_type": req.request_type,
                    "status": req.status,
                    "latency_ms": req.latency_ms,
                    "created_at": req.created_at.isoformat() if req.created_at else None,
                    "completed_at": req.completed_at.isoformat() if req.completed_at else None
                }
                results.append(result)
            
            logger.info(f"[CONTEXT_DB] Retrieved {len(results)} requests for agent {agent_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"[CONTEXT_DB] Failed to get request history: {e}", exc_info=True)
            return []
    
    
    def get_request_stats(self, agent_id: int) -> Dict[str, Any]:
        """
        Get statistics about agent's context requests
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Dictionary with request statistics
        """
        try:
            total_sent = self.db.query(ContextRequest).filter(
                ContextRequest.requester_id == agent_id
            ).count()
            
            total_received = self.db.query(ContextRequest).filter(
                ContextRequest.target_id == agent_id
            ).count()
            
            pending = self.db.query(ContextRequest).filter(
                and_(
                    ContextRequest.requester_id == agent_id,
                    ContextRequest.status == "pending"
                )
            ).count()
            
            fulfilled = self.db.query(ContextRequest).filter(
                and_(
                    ContextRequest.requester_id == agent_id,
                    ContextRequest.status == "fulfilled"
                )
            ).count()
            
            avg_latency = self.db.query(
                ContextRequest.latency_ms
            ).filter(
                and_(
                    ContextRequest.requester_id == agent_id,
                    ContextRequest.latency_ms.isnot(None)
                )
            ).all()
            
            avg_latency_ms = sum([x[0] for x in avg_latency]) / len(avg_latency) if avg_latency else 0
            
            stats = {
                "agent_id": agent_id,
                "requests_sent": total_sent,
                "requests_received": total_received,
                "pending_requests": pending,
                "fulfilled_requests": fulfilled,
                "average_latency_ms": avg_latency_ms,
                "fulfillment_rate": (fulfilled / total_sent * 100) if total_sent > 0 else 0
            }
            
            logger.info(f"[CONTEXT_DB] Stats for agent {agent_id}: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"[CONTEXT_DB] Failed to get stats: {e}", exc_info=True)
            return {}
        
    
    
    
    def close(self):
        """Close database connection"""
        try:
            self.db.close()
            logger.info("[CONTEXT_DB] Database connection closed")
        except Exception as e:
            logger.error(f"[CONTEXT_DB] Error closing connection: {e}", exc_info=True)


# Dependency injection for FastAPI
def get_context_db_service(db_session: Session = None) -> ContextDBService:
    """
    Get or create ContextDBService instance
    
    Args:
        db_session: SQLAlchemy session
        
    Returns:
        ContextDBService instance
    """
    return ContextDBService(db_session=db_session)
