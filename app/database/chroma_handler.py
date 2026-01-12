"""
ChromaDB Handler - Vector Database Operations
Manages embeddings, vector storage, and semantic search
✅ FIXED: Proper metadata handling for ChromaDB compatibility
"""
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChromaHandler:
    """
    Handler for ChromaDB operations
    Manages collections, embeddings, and vector search
    """
    
    def __init__(self):
        """Initialize ChromaDB handler"""
        try:
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Collections dictionary
            self.collections = {}
            self._initialize_collections()
            
            logger.info("✅ ChromaDB handler initialized")
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {str(e)}")
            raise
    
    def _initialize_collections(self):
        """Initialize default collections"""
        collection_names = [
            "agent_contexts",
            "conversation_history",
            "memory_vectors",
            "document_store"
        ]
        
        for name in collection_names:
            try:
                collection = self.client.get_or_create_collection(
                    name=name,
                    metadata={
                        "hnsw:space": "cosine",
                        "description": f"Collection for {name}",
                        "created_at": datetime.utcnow().isoformat()
                    }
                )
                self.collections[name] = collection
                logger.info(f"✅ Collection '{name}' initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize collection '{name}': {str(e)}")
    
    def get_collection(self, collection_name: str):
        """Get or create a collection"""
        if collection_name not in self.collections:
            self.collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self.collections[collection_name]
    
    async def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Add documents to collection with automatic embeddings
        
        Args:
            collection_name: Collection to add to
            documents: List of document texts
            ids: Optional document IDs
            metadatas: Optional metadata for each document
            
        Returns:
            List of document IDs
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Generate IDs if not provided
            if not ids:
                ids = [f"doc_{i}_{int(datetime.utcnow().timestamp())}" for i in range(len(documents))]
            
            # ✅ FIX: ALWAYS create non-empty metadata
            # ChromaDB requires non-empty dicts for metadata
            # NEVER pass empty dict {} to ChromaDB
            metadatas_list = []
            for i, doc in enumerate(documents):
                if metadatas and i < len(metadatas) and metadatas[i]:
                    # Merge existing metadata with required fields
                    meta = {
                        **metadatas[i], 
                        "doc_index": str(i), 
                        "added_at": datetime.utcnow().isoformat()
                    }
                else:
                    # Create default metadata (ALWAYS non-empty)
                    meta = {
                        "doc_index": str(i),
                        "added_at": datetime.utcnow().isoformat(),
                        "default": "true"
                    }
                metadatas_list.append(meta)
            
            # Add documents with guaranteed non-empty metadata
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas_list
            )
            
            logger.info(f"✅ Added {len(ids)} documents to {collection_name}")
            return ids
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {str(e)}")
            raise
    
    async def search(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Search collection with semantic similarity
        
        Args:
            collection_name: Collection to search
            query_text: Query text (will be embedded)
            n_results: Number of results to return
            where: Optional metadata filter
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of search results with scores
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Query collection
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1 - (distance / 2)  # Convert cosine distance to similarity
                    
                    if similarity >= threshold:
                        formatted_results.append({
                            "document": doc,
                            "similarity": similarity,
                            "metadata": results["metadatas"][0][i],
                            "distance": distance
                        })
            
            logger.info(f"✅ Search found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {str(e)}")
            return []
    
    async def update_document(
        self,
        collection_name: str,
        doc_id: str,
        document: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Update a document
        
        Args:
            collection_name: Collection name
            doc_id: Document ID
            document: New document content
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        try:
            collection = self.get_collection(collection_name)
            
            # ✅ FIX: Ensure metadata is not empty
            if not metadata:
                metadata = {
                    "default": "true",
                    "updated_at": datetime.utcnow().isoformat()
                }
            else:
                metadata = {
                    **metadata,
                    "default": "true",
                    "updated_at": datetime.utcnow().isoformat()
                }
            
            collection.update(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata]
            )
            logger.info(f"✅ Updated document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update document: {str(e)}")
            return False
    
    async def delete_document(
        self,
        collection_name: str,
        doc_id: str
    ) -> bool:
        """
        Delete a document
        
        Args:
            collection_name: Collection name
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            collection = self.get_collection(collection_name)
            collection.delete(ids=[doc_id])
            logger.info(f"✅ Deleted document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {str(e)}")
            return False
    
    async def get_document(
        self,
        collection_name: str,
        doc_id: str
    ) -> Optional[Dict]:
        """
        Get a document by ID
        
        Args:
            collection_name: Collection name
            doc_id: Document ID
            
        Returns:
            Document dict or None
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if result["documents"]:
                return {
                    "id": doc_id,
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get document: {str(e)}")
            return None
    
    async def get_collection_count(self, collection_name: str) -> int:
        """
        Get number of documents in collection
        
        Args:
            collection_name: Collection name
            
        Returns:
            Number of documents
        """
        try:
            collection = self.get_collection(collection_name)
            return collection.count()
        except Exception as e:
            logger.error(f"❌ Failed to get count: {str(e)}")
            return 0
    
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete entire collection
        
        Args:
            collection_name: Collection name
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            logger.info(f"✅ Deleted collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete collection: {str(e)}")
            return False
    
    async def batch_add_documents(
        self,
        collection_name: str,
        documents: List[str],
        batch_size: int = 50
    ) -> List[str]:
        """
        Add documents in batches (more efficient)
        
        Args:
            collection_name: Collection name
            documents: List of documents
            batch_size: Batch size
            
        Returns:
            List of all document IDs
        """
        all_ids = []
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                ids = await self.add_documents(
                    collection_name=collection_name,
                    documents=batch
                )
                all_ids.extend(ids)
            
            logger.info(f"✅ Batch added {len(all_ids)} documents")
            return all_ids
        except Exception as e:
            logger.error(f"❌ Batch add failed: {str(e)}")
            raise


# Global instance
chroma_handler = ChromaHandler()