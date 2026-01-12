"""
SQLAlchemy ORM models for AgentSync database
Includes models for agents, contexts, and inter-agent communication
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime


Base = declarative_base()


class Agent(Base):
    """
    Agent model - stores information about each agent
    """
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    agent_type = Column(String, nullable=False)  # email, calendar, slack
    # status = Column(String, default="inactive")  # active, inactive, error
    description = Column(Text, nullable=True)
    
    # Relationships
    contexts = relationship("AgentContext", back_populates="agent")
    sent_requests = relationship(
        "ContextRequest",
        foreign_keys="ContextRequest.requester_id",
        back_populates="requester"
    )
    received_requests = relationship(
        "ContextRequest",
        foreign_keys="ContextRequest.target_id",
        back_populates="target"
    )


class AgentContext(Base):
    """
    Stores context information for agents
    Links to vector embeddings in ChromaDB
    """
    __tablename__ = "agent_contexts"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    context_key = Column(String, nullable=False, index=True)
    context_summary = Column(Text, nullable=False)
    embedding_id = Column(String, nullable=True)  # ID in ChromaDB
    metadata_json = Column(Text, nullable=True)
    relevance_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    agent = relationship("Agent", back_populates="contexts")


class ContextRequest(Base):
    """
    Tracks agent-to-agent context requests
    Used for debugging and analytics
    """
    __tablename__ = "context_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, unique=True, nullable=False, index=True)
    requester_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    target_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    query = Column(Text, nullable=False)
    request_type = Column(String, default="context_query")
    status = Column(String, default="pending")
    response_context_id = Column(Integer, ForeignKey("agent_contexts.id"), nullable=True)
    response_summary = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    requester = relationship(
        "Agent",
        foreign_keys=[requester_id],
        back_populates="sent_requests"
    )
    target = relationship(
        "Agent",
        foreign_keys=[target_id],
        back_populates="received_requests"
    )
