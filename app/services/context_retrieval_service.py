"""
Context Retrieval Service for AgentSync
Handles semantic search in ChromaDB and cross-agent context requests
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings

from app.config import get_settings
from app.services.chroma_service import get_chroma_service

logger = logging.getLogger(__name__)


class ContextRetrievalService:
    """
    Service for retrieving context from ChromaDB and agents
    Supports semantic search and agent-specific context requests
    """
    
    def __init__(self):
        """Initialize context retrieval service"""
        self.settings = get_settings()
        self.chroma_service = None
        logger.info("[CONTEXT_RETRIEVAL] Service initialized")
    
    def _get_chroma_service(self):
        """Lazy load ChromaDB service"""
        if self.chroma_service is None:
            self.chroma_service = get_chroma_service()
        return self.chroma_service
    
    async def search_chromadb(
        self,
        query: str,
        agent_id: Optional[int] = None,
        agent_type: Optional[str] = None,
        top_k: int = 5,
        min_relevance: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search ChromaDB for relevant contexts using semantic search
        
        Args:
            query: Natural language search query
            agent_id: Optional filter by specific agent ID
            agent_type: Optional filter by agent type (email, tasks, calendar, etc.)
            top_k: Number of top results to return
            min_relevance: Minimum relevance score (0-1)
        
        Returns:
            List of context dictionaries with summary, metadata, and relevance score
        """
        logger.info(f"[CONTEXT_RETRIEVAL] Searching ChromaDB")
        logger.info(f"[CONTEXT_RETRIEVAL] Query: {query}")
        logger.info(f"[CONTEXT_RETRIEVAL] Filters: agent_id={agent_id}, agent_type={agent_type}")
        
        try:
            chroma_service = self._get_chroma_service()
            
            # Prepare where filter for metadata
            where_filter = {}
            if agent_id is not None:
                where_filter["agent_id"] = agent_id
            if agent_type is not None:
                where_filter["agent_type"] = agent_type
            
            # Get collection
            collection = chroma_service.client.get_collection(
                name=self.settings.chroma_collection_name
            )
            
            # Perform semantic search
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Check if results exist
            if not results['ids'] or len(results['ids'][0]) == 0:
                logger.info("[CONTEXT_RETRIEVAL] No contexts found")
                return []
            
            # Format results
            contexts = []
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                relevance_score = 1 - distance  # Convert distance to similarity
                
                # Filter by minimum relevance
                if relevance_score < min_relevance:
                    logger.debug(f"[CONTEXT_RETRIEVAL] Skipping low relevance result: {relevance_score:.3f}")
                    continue
                
                context = {
                    "vector_id": results['ids'][0][i],
                    "summary": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "relevance_score": relevance_score,
                    "distance": distance
                }
                
                contexts.append(context)
                
                logger.info(f"[CONTEXT_RETRIEVAL] Found context: {context['vector_id']} (relevance: {relevance_score:.3f})")
            
            logger.info(f"[CONTEXT_RETRIEVAL] ✅ Retrieved {len(contexts)} relevant context(s)")
            return contexts
            
        except Exception as e:
            logger.error(f"[CONTEXT_RETRIEVAL] ❌ Search failed: {e}", exc_info=True)
            return []
    
    async def search_chromadb_by_agent(
        self,
        agent_type: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for contexts specific to an agent type
        
        Args:
            agent_type: Agent type to search for (email, tasks, calendar, etc.)
            query: Search query
            top_k: Number of results
        
        Returns:
            List of contexts from specified agent
        """
        logger.info(f"[CONTEXT_RETRIEVAL] Searching {agent_type} agent contexts")
        
        return await self.search_chromadb(
            query=query,
            agent_type=agent_type,
            top_k=top_k
        )
    
    async def request_agent_context(
        self,
        target_agent: str,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Request context from a specific agent by searching their stored contexts
        This is for cross-agent context sharing (Tier 3 of your plan)
        
        Args:
            target_agent: Agent type to request context from (email, tasks, calendar, etc.)
            query: What context information is needed
            top_k: Number of relevant contexts to retrieve
        
        Returns:
            Dictionary with agent contexts and metadata
        """
        logger.info(f"[CONTEXT_RETRIEVAL] Requesting context from {target_agent} agent")
        logger.info(f"[CONTEXT_RETRIEVAL] Query: {query}")
        
        try:
            # Search ChromaDB for contexts from target agent
            contexts = await self.search_chromadb_by_agent(
                agent_type=target_agent,
                query=query,
                top_k=top_k
            )
            
            if not contexts:
                logger.warning(f"[CONTEXT_RETRIEVAL] No contexts found from {target_agent} agent")
                return {
                    "success": False,
                    "agent": target_agent,
                    "query": query,
                    "contexts": [],
                    "context_count": 0,
                    "formatted_context": f"No context available from {target_agent} agent.",
                    "message": f"The {target_agent} agent has no relevant context for: '{query}'"
                }
            
            # Format contexts for LLM consumption
            formatted_context = self.format_context_for_llm(contexts)
            
            logger.info(f"[CONTEXT_RETRIEVAL] ✅ Retrieved {len(contexts)} context(s) from {target_agent} agent")
            
            return {
                "success": True,
                "agent": target_agent,
                "query": query,
                "contexts": contexts,
                "context_count": len(contexts),
                "formatted_context": formatted_context,
                "message": f"Successfully retrieved {len(contexts)} context(s) from {target_agent} agent"
            }
            
        except Exception as e:
            logger.error(f"[CONTEXT_RETRIEVAL] ❌ Failed to request context from {target_agent}: {e}", exc_info=True)
            return {
                "success": False,
                "agent": target_agent,
                "query": query,
                "contexts": [],
                "context_count": 0,
                "formatted_context": f"Error retrieving context from {target_agent} agent.",
                "error": str(e)
            }
    
    async def request_multi_agent_context(
        self,
        agent_types: List[str],
        query: str,
        top_k_per_agent: int = 3
    ) -> Dict[str, Any]:
        """
        Request context from multiple agents simultaneously
        Useful when LLM identifies multiple potential context sources
        
        Args:
            agent_types: List of agent types to query
            query: Context query
            top_k_per_agent: Max contexts per agent
        
        Returns:
            Aggregated context from all agents
        """
        logger.info(f"[CONTEXT_RETRIEVAL] Requesting context from multiple agents: {agent_types}")
        
        all_contexts = []
        agent_results = {}
        
        for agent_type in agent_types:
            result = await self.request_agent_context(
                target_agent=agent_type,
                query=query,
                top_k=top_k_per_agent
            )
            
            agent_results[agent_type] = result
            
            if result["success"]:
                all_contexts.extend(result["contexts"])
        
        # Sort by relevance
        all_contexts.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return {
            "success": len(all_contexts) > 0,
            "agents_queried": agent_types,
            "query": query,
            "total_contexts": len(all_contexts),
            "contexts": all_contexts,
            "agent_results": agent_results,
            "formatted_context": self.format_context_for_llm(all_contexts)
        }
    
    def get_agent_context_summary(
        self,
        agent_type: str
    ) -> Dict[str, Any]:
        """
        Get summary information about an agent's stored contexts
        
        Args:
            agent_type: Agent type to summarize
        
        Returns:
            Summary statistics
        """
        logger.info(f"[CONTEXT_RETRIEVAL] Getting context summary for {agent_type}")
        
        try:
            chroma_service = self._get_chroma_service()
            collection = chroma_service.client.get_collection(
                name=self.settings.chroma_collection_name
            )
            
            # Get all contexts for this agent
            results = collection.get(
                where={"agent_type": agent_type},
                include=["metadatas"]
            )
            
            if not results['ids']:
                return {
                    "agent_type": agent_type,
                    "total_contexts": 0,
                    "has_context": False,
                    "message": f"No stored context for {agent_type} agent"
                }
            
            # Analyze metadata
            timestamps = [m.get('timestamp', '') for m in results['metadatas']]
            timestamps = [t for t in timestamps if t]
            timestamps.sort()
            
            return {
                "agent_type": agent_type,
                "total_contexts": len(results['ids']),
                "has_context": True,
                "oldest_context": timestamps[0] if timestamps else None,
                "newest_context": timestamps[-1] if timestamps else None,
                "context_ids": results['ids'][:5],
                "message": f"{agent_type} agent has {len(results['ids'])} stored context(s)"
            }
            
        except Exception as e:
            logger.error(f"[CONTEXT_RETRIEVAL] Error getting summary: {e}", exc_info=True)
            return {
                "agent_type": agent_type,
                "total_contexts": 0,
                "has_context": False,
                "error": str(e)
            }
    
    async def get_recent_contexts(
        self,
        agent_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most recent contexts, optionally filtered by agent
        
        Args:
            agent_type: Optional agent type filter
            limit: Number of contexts to return
        
        Returns:
            List of recent contexts sorted by timestamp
        """
        logger.info(f"[CONTEXT_RETRIEVAL] Getting recent contexts")
        
        try:
            chroma_service = self._get_chroma_service()
            collection = chroma_service.client.get_collection(
                name=self.settings.chroma_collection_name
            )
            
            # Build where filter
            where_filter = None
            if agent_type:
                where_filter = {"agent_type": agent_type}
            
            # Get all contexts (or filtered)
            results = collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
                limit=limit
            )
            
            if not results['ids']:
                logger.info("[CONTEXT_RETRIEVAL] No recent contexts found")
                return []
            
            # Format and sort by timestamp
            contexts = []
            for i in range(len(results['ids'])):
                context = {
                    "vector_id": results['ids'][i],
                    "summary": results['documents'][i],
                    "metadata": results['metadatas'][i],
                    "timestamp": results['metadatas'][i].get('timestamp', '')
                }
                contexts.append(context)
            
            # Sort by timestamp descending
            contexts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            logger.info(f"[CONTEXT_RETRIEVAL] ✅ Retrieved {len(contexts)} recent context(s)")
            return contexts[:limit]
            
        except Exception as e:
            logger.error(f"[CONTEXT_RETRIEVAL] ❌ Failed to get recent contexts: {e}", exc_info=True)
            return []
    
    def format_context_for_llm(
        self,
        contexts: List[Dict[str, Any]],
        max_contexts: int = 5
    ) -> str:
        """
        Format retrieved contexts into readable text for LLM consumption
        
        Args:
            contexts: List of context dictionaries
            max_contexts: Maximum number of contexts to include
        
        Returns:
            Formatted string for LLM prompt
        """
        if not contexts:
            return "No relevant context found."
        
        # Limit number of contexts
        contexts = contexts[:max_contexts]
        
        formatted = "📚 RETRIEVED CONTEXT:\n\n"
        
        for i, ctx in enumerate(contexts, 1):
            metadata = ctx.get('metadata', {})
            summary = ctx.get('summary', 'No summary available')
            relevance = ctx.get('relevance_score', 0)
            
            formatted += f"Context {i} (Relevance: {relevance:.2f}):\n"
            formatted += f"Agent: {metadata.get('agent_type', 'unknown')}\n"
            formatted += f"User Input: {metadata.get('user_input', 'N/A')}\n"
            formatted += f"Timestamp: {metadata.get('timestamp', 'N/A')}\n"
            formatted += f"Summary: {summary[:200]}...\n"
            formatted += "-" * 60 + "\n\n"
        
        return formatted
    
    def format_context_compact(
        self,
        contexts: List[Dict[str, Any]]
    ) -> str:
        """
        Format contexts in compact form for LLM (less verbose)
        
        Args:
            contexts: List of context dictionaries
        
        Returns:
            Compact formatted string
        """
        if not contexts:
            return "No context available."
        
        formatted_parts = []
        
        for ctx in contexts:
            metadata = ctx.get('metadata', {})
            summary = ctx.get('summary', '')
            
            # Extract key info
            agent_type = metadata.get('agent_type', 'unknown')
            user_input = metadata.get('user_input', '')
            
            compact = f"[{agent_type.upper()}] Previous: '{user_input}' → {summary[:100]}"
            formatted_parts.append(compact)
        
        return "\n".join(formatted_parts)


# Singleton instance
_context_retrieval_service: Optional[ContextRetrievalService] = None


def get_context_retrieval_service() -> ContextRetrievalService:
    """
    Get or create singleton instance of ContextRetrievalService
    
    Returns:
        ContextRetrievalService instance
    """
    global _context_retrieval_service
    
    if _context_retrieval_service is None:
        _context_retrieval_service = ContextRetrievalService()
    
    return _context_retrieval_service
