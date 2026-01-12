"""
ChromaDB Service for AgentSync
Manages vector embeddings and semantic search for agent contexts
"""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import json
import uuid

logger = logging.getLogger(__name__)


class ChromaDBService:
    """
    Centralized service for all ChromaDB operations
    Handles context storage, retrieval, and semantic search
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "agent_contexts",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize ChromaDB client with persistent storage
        
        Args:
            persist_directory: Path to ChromaDB persistent storage
            collection_name: Name of the collection for agent contexts
            embedding_model: Sentence transformer model for embeddings
        """
        try:
            # ✅ Use PersistentClient (avoids singleton conflict)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True  # Allows test cleanup
                )
            )
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            
            self.collection_name = collection_name
            self.collection = None
            
            # Create or get collection
            self.create_collection(collection_name)
            
            logger.info(f"[CHROMA] Initialized with persist_directory: {persist_directory}")
            logger.info(f"[CHROMA] Using embedding model: {embedding_model}")
            
        except Exception as e:
            logger.error(f"[CHROMA] Initialization failed: {e}", exc_info=True)
            raise

    
    
    def create_collection(self, name: str) -> chromadb.Collection:
        """
        Create or get existing collection for agent contexts
        
        Args:
            name: Collection name
            
        Returns:
            ChromaDB collection object
        """
        try:
            # Get or create collection with embedding function
            self.collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata={"description": "Agent context storage for federated coordination"}
            )
            
            logger.info(f"[CHROMA] Collection '{name}' ready (count: {self.collection.count()})")
            return self.collection
            
        except Exception as e:
            logger.error(f"[CHROMA] Failed to create collection '{name}': {e}", exc_info=True)
            raise
    
    
    def add_context(
        self,
        agent_id: int,
        context_summary: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None
    ) -> str:
        """
        Store context with automatic embeddings in ChromaDB
        
        Args:
            agent_id: ID of the agent creating this context
            context_summary: Human-readable summary for embedding
            metadata: Additional metadata (action, timestamp, etc.)
            vector_id: Optional custom ID (auto-generated if not provided)
            
        Returns:
            vector_id: Unique identifier in ChromaDB
        """
        try:
            # Generate unique vector ID if not provided
            if not vector_id:
                vector_id = f"ctx_{agent_id}_{uuid.uuid4().hex[:12]}"
            
            # Prepare metadata
            context_metadata = {
                "agent_id": agent_id,
                "created_at": datetime.utcnow().isoformat(),
                "summary_length": len(context_summary)
            }
            
            # Merge with provided metadata
            if metadata:
                context_metadata.update(metadata)
            
            # Convert all metadata values to strings (ChromaDB requirement)
            context_metadata = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in context_metadata.items()
            }
            
            # Add to collection (embedding generated automatically)
            self.collection.add(
                documents=[context_summary],
                metadatas=[context_metadata],
                ids=[vector_id]
            )
            
            logger.info(f"[CHROMA] Stored context: {vector_id} (agent: {agent_id})")
            logger.debug(f"[CHROMA] Summary: {context_summary[:100]}...")
            
            return vector_id
            
        except Exception as e:
            logger.error(f"[CHROMA] Failed to add context: {e}", exc_info=True)
            raise
    
    
    def search_context(
        self,
        query: str,
        top_k: int = 5,
        agent_filter: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for related contexts across all agents
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            agent_filter: Optional filter by specific agent_id
            metadata_filter: Optional metadata filters (e.g., {"action": "send_email"})
            
        Returns:
            List of context results with ids, documents, metadata, distances
        """
        try:
            # Build filter query
            where_filter = {}
            
            if agent_filter:
                where_filter["agent_id"] = agent_filter
            
            if metadata_filter:
                where_filter.update(metadata_filter)
            
            # Perform semantic search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter if where_filter else None
            )
            
            # Format results
            formatted_results = []
            
            if results and results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        "vector_id": results['ids'][0][i],
                        "context_summary": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else None,
                        "relevance_score": 1 - results['distances'][0][i] if 'distances' in results else 1.0
                    })
            
            logger.info(f"[CHROMA] Search for '{query[:50]}...' returned {len(formatted_results)} results")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"[CHROMA] Search failed: {e}", exc_info=True)
            return []
    
    
    def get_agent_contexts(
        self,
        agent_id: int,
        limit: int = 10,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent contexts for a specific agent
        
        Args:
            agent_id: Agent ID to retrieve contexts for
            limit: Maximum number of contexts to return
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            List of contexts with metadata
        """
        try:
            # Query all contexts for this agent
            results = self.collection.get(
                where={"agent_id": agent_id},
                limit=limit,
                include=["documents", "metadatas", "embeddings"] if include_embeddings else ["documents", "metadatas"]
            )
            
            # Format results
            formatted_results = []
            
            if results and results['ids']:
                for i in range(len(results['ids'])):
                    context = {
                        "vector_id": results['ids'][i],
                        "context_summary": results['documents'][i],
                        "metadata": results['metadatas'][i]
                    }
                    
                    if include_embeddings and 'embeddings' in results:
                        context["embedding"] = results['embeddings'][i]
                    
                    formatted_results.append(context)
            
            logger.info(f"[CHROMA] Retrieved {len(formatted_results)} contexts for agent {agent_id}")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"[CHROMA] Failed to get agent contexts: {e}", exc_info=True)
            return []
    
    
    def get_context_by_vector_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific context by vector ID
        
        Args:
            vector_id: Unique identifier in ChromaDB
            
        Returns:
            Context data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[vector_id],
                include=["documents", "metadatas"]
            )
            
            if results and results['ids'] and len(results['ids']) > 0:
                return {
                    "vector_id": results['ids'][0],
                    "context_summary": results['documents'][0],
                    "metadata": results['metadatas'][0]
                }
            
            logger.warning(f"[CHROMA] Context not found: {vector_id}")
            return None
            
        except Exception as e:
            logger.error(f"[CHROMA] Failed to get context by vector_id: {e}", exc_info=True)
            return None
    
    
    def delete_old_contexts(self, days_old: int = 30) -> int:
        """
        Cleanup contexts older than specified days
        
        Args:
            days_old: Delete contexts older than this many days
            
        Returns:
            Number of contexts deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            cutoff_iso = cutoff_date.isoformat()
            
            # Get all contexts
            all_contexts = self.collection.get(include=["metadatas"])
            
            # Find old contexts
            old_context_ids = []
            
            for i, metadata in enumerate(all_contexts['metadatas']):
                created_at = metadata.get('created_at', '')
                
                if created_at and created_at < cutoff_iso:
                    old_context_ids.append(all_contexts['ids'][i])
            
            # Delete old contexts
            if old_context_ids:
                self.collection.delete(ids=old_context_ids)
                logger.info(f"[CHROMA] Deleted {len(old_context_ids)} contexts older than {days_old} days")
            else:
                logger.info(f"[CHROMA] No contexts older than {days_old} days found")
            
            return len(old_context_ids)
            
        except Exception as e:
            logger.error(f"[CHROMA] Failed to delete old contexts: {e}", exc_info=True)
            return 0
    
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample contexts to analyze
            sample = self.collection.peek(limit=100)
            
            # Count contexts per agent
            agent_counts = {}
            if sample and sample['metadatas']:
                for metadata in sample['metadatas']:
                    agent_id = metadata.get('agent_id', 'unknown')
                    agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
            
            stats = {
                "total_contexts": count,
                "collection_name": self.collection_name,
                "agent_distribution": agent_counts,
                "sample_size": len(sample['ids']) if sample and sample['ids'] else 0
            }
            
            logger.info(f"[CHROMA] Collection stats: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"[CHROMA] Failed to get stats: {e}", exc_info=True)
            return {"error": str(e)}
    
    
    def reset_collection(self) -> bool:
        """
        ⚠️ DANGER: Delete all contexts in collection (for testing only)
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"[CHROMA] Deleted collection: {self.collection_name}")
            
            # Recreate empty collection
            self.create_collection(self.collection_name)
            logger.info(f"[CHROMA] Recreated empty collection: {self.collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"[CHROMA] Failed to reset collection: {e}", exc_info=True)
            return False


# Global singleton instance (initialized by config)
_chroma_service_instance = None


def get_chroma_service(
    persist_directory: str = "./data/chroma",
    collection_name: str = "agent_contexts",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> ChromaDBService:
    """
    Get or create singleton ChromaDB service instance
    
    Args:
        persist_directory: Path to ChromaDB storage
        collection_name: Collection name
        embedding_model: Embedding model name
        
    Returns:
        ChromaDBService instance
    """
    global _chroma_service_instance
    
    if _chroma_service_instance is None:
        _chroma_service_instance = ChromaDBService(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model=embedding_model
        )
    
    return _chroma_service_instance
