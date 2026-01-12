"""
ChromaDB Collection Inspection Tool for AgentSync
Directly inspects ChromaDB vector database
"""
import chromadb
from chromadb.config import Settings
import json
from datetime import datetime
from typing import Dict, Any, List


class ChromaDBInspector:
    """Inspector for ChromaDB collections"""
    
    def __init__(self, persist_directory: str = "./data/chroma"):
        """Initialize ChromaDB client"""
        print(f"\n{'='*120}")
        print("🔍 CHROMADB INSPECTOR INITIALIZING")
        print(f"{'='*120}")
        
        try:
            # Use same client configuration as your app
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            print(f"✅ Connected to ChromaDB at: {persist_directory}")
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            raise
    
    def list_collections(self):
        """List all collections in ChromaDB"""
        print(f"\n{'='*120}")
        print("📚 COLLECTIONS")
        print(f"{'='*120}")
        
        try:
            collections = self.client.list_collections()
            
            if not collections:
                print("\n⚠️  No collections found")
                return []
            
            print(f"\n✅ Found {len(collections)} collection(s):\n")
            
            for col in collections:
                count = col.count()
                print(f"   📁 {col.name}")
                print(f"      └─ Items: {count}")
                
                # Get metadata
                if hasattr(col, 'metadata') and col.metadata:
                    print(f"      └─ Metadata: {col.metadata}")
            
            return collections
            
        except Exception as e:
            print(f"❌ Error listing collections: {e}")
            return []
    
    def inspect_collection(self, collection_name: str = "agent_contexts"):
        """Inspect specific collection in detail"""
        print(f"\n{'='*120}")
        print(f"🔍 INSPECTING COLLECTION: {collection_name}")
        print(f"{'='*120}")
        
        try:
            # Get collection
            collection = self.client.get_collection(name=collection_name)
            
            # Get total count
            total_count = collection.count()
            print(f"\n📊 Total contexts: {total_count}")
            
            if total_count == 0:
                print("\n⚠️  Collection is empty")
                return None
            
            # Get all data
            all_data = collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
            print(f"\n{'─'*120}")
            print("📋 CONTEXT DETAILS")
            print(f"{'─'*120}\n")
            
            # Display each context
            for i in range(len(all_data['ids'])):
                self._display_context(
                    index=i + 1,
                    vector_id=all_data['ids'][i],
                    summary=all_data['documents'][i],
                    metadata=all_data['metadatas'][i],
                    embedding=all_data.get('embeddings', [[]])[i] if all_data.get('embeddings') else None
                )
            
            # Statistics
            self._display_statistics(all_data)
            
            return all_data
            
        except Exception as e:
            print(f"❌ Error inspecting collection: {e}")
            return None
    
    def _display_context(
        self,
        index: int,
        vector_id: str,
        summary: str,
        metadata: Dict[str, Any],
        embedding: List[float] = None
    ):
        """Display single context details"""
        print(f"{'▼'*100}")
        print(f"CONTEXT #{index}")
        print(f"{'▼'*100}")
        
        # Vector ID
        print(f"\n🔑 Vector ID: {vector_id}")
        
        # Summary (truncated)
        print(f"\n📝 Context Description:")
        
        print(f"   {summary}")
        
        # Metadata
        print(f"\n📊 Metadata:")
        for key, value in metadata.items():
            if isinstance(value, str) and len(value) > 60:
                print(f"   • {key}: {value[:60]}...")
            else:
                print(f"   • {key}: {value}")
        
        # Embedding info
        if embedding:
            print(f"\n🧮 Embedding:")
            print(f"   • Dimensions: {len(embedding)}")
            print(f"   • Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
            print(f"   • Magnitude: {sum(x**2 for x in embedding)**0.5:.4f}")
        
        print()
    
    def _display_statistics(self, all_data: Dict[str, Any]):
        """Display collection statistics"""
        print(f"\n{'='*120}")
        print("📈 STATISTICS")
        print(f"{'='*120}")
        
        # Count by agent
        agent_counts = {}
        for metadata in all_data['metadatas']:
            agent_id = metadata.get('agent_id', 'unknown')
            agent_type = metadata.get('agent_type', 'unknown')
            key = f"Agent {agent_id} ({agent_type})"
            agent_counts[key] = agent_counts.get(key, 0) + 1
        
        print(f"\n📊 Contexts by Agent:")
        for agent, count in sorted(agent_counts.items()):
            print(f"   • {agent}: {count} context(s)")
        
        # Embedding dimensions
        if all_data.get('embeddings') and len(all_data['embeddings']) > 0:
            embedding_dim = len(all_data['embeddings'][0])
            print(f"\n🧮 Embedding Dimensions: {embedding_dim}")
        
        # Oldest and newest
        timestamps = []
        for metadata in all_data['metadatas']:
            if 'timestamp' in metadata:
                timestamps.append(metadata['timestamp'])
        
        if timestamps:
            timestamps.sort()
            print(f"\n📅 Time Range:")
            print(f"   • Oldest: {timestamps[0]}")
            print(f"   • Newest: {timestamps[-1]}")
    
    def search_contexts(
        self,
        query: str,
        collection_name: str = "agent_contexts",
        top_k: int = 5
    ):
        """Search contexts using semantic search"""
        print(f"\n{'='*120}")
        print(f"🔎 SEMANTIC SEARCH")
        print(f"{'='*120}")
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            print(f"\nQuery: \"{query}\"")
            print(f"Top K: {top_k}\n")
            
            # Perform search
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['ids'] or len(results['ids'][0]) == 0:
                print("⚠️  No results found")
                return None
            
            print(f"✅ Found {len(results['ids'][0])} result(s)\n")
            
            # Display results
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                print(f"{'─'*120}")
                print(f"Result #{i+1} - Relevance: {similarity:.3f} (distance: {distance:.3f})")
                print(f"{'─'*120}")
                
                print(f"Vector ID: {results['ids'][0][i]}")
                print(f"\nSummary: {results['documents'][0][i][:150]}...")
                print(f"\nMetadata:")
                for key, value in results['metadatas'][0][i].items():
                    print(f"   • {key}: {value}")
                print()
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return None
    
    def check_specific_context(
        self,
        vector_id: str,
        collection_name: str = "agent_contexts"
    ):
        """Check if specific context exists"""
        print(f"\n{'='*120}")
        print(f"🔍 CHECKING SPECIFIC CONTEXT")
        print(f"{'='*120}")
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            print(f"\nLooking for Vector ID: {vector_id}\n")
            
            result = collection.get(
                ids=[vector_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not result['ids'] or len(result['ids']) == 0:
                print("❌ Context NOT FOUND")
                return False
            
            print("✅ Context FOUND!\n")
            
            self._display_context(
                index=1,
                vector_id=result['ids'][0],
                summary=result['documents'][0],
                metadata=result['metadatas'][0],
                embedding=result.get('embeddings', [[]])[0] if result.get('embeddings') else None
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    """Main inspection workflow"""
    
    print("\n" + "="*120)
    print("🔍 AGENTSYNC CHROMADB INSPECTOR")
    print("="*120)
    
    # Initialize inspector
    try:
        inspector = ChromaDBInspector(persist_directory="./data/chroma")
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        return
    
    # List all collections
    collections = inspector.list_collections()
    
    if not collections:
        print("\n⚠️  No collections to inspect")
        return
    
    # Inspect agent_contexts collection
    inspector.inspect_collection("agent_contexts")
    
    # # Test semantic search
    # print("\n" + "="*60)
    # print("🔎 TESTING SEMANTIC SEARCH")
    # print("="*60)
    
    # test_queries = [
    #     "Interview Preparation task",
    # ]
    
    # for query in test_queries:
    #     inspector.search_contexts(query, top_k=3)
    
    # # Check specific context (example with your actual vector_id)
    # # Uncomment and replace with actual ID
    # inspector.check_specific_context("ctx_1_30be16d5cdcb")
    
    print("\n" + "="*120)
    print("✅ INSPECTION COMPLETE")
    print("="*120 + "\n")


if __name__ == "__main__":
    main()
