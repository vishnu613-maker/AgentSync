"""
Initialize ChromaDB
Creates the ChromaDB database in data/chroma directory
"""
import os
import chromadb
from chromadb.config import Settings

print("🚀 Initializing ChromaDB...")
print()

# Create data directory if it doesn't exist
data_dir = "./data/chroma"
os.makedirs(data_dir, exist_ok=True)
print(f"✅ Created directory: {data_dir}")

# Initialize ChromaDB with persistence
client = chromadb.PersistentClient(
    path=data_dir,
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

print(f"✅ ChromaDB initialized at: {data_dir}")
print()

# Create a test collection
collection = client.get_or_create_collection(
    name="agent_contexts",
    metadata={"hnsw:space": "cosine"}
)

print(f"✅ Created collection: agent_contexts")
print(f"   Documents: {collection.count()}")
print()

# Add a test document
collection.add(
    ids=["test_1"],
    documents=["This is a test document for AgentSync context management"],
    metadatas=[{
        "user_id": "1",
        "agent_type": "system",
        "context_type": "initialization"
    }]
)

print(f"✅ Added test document")
print(f"   Total documents: {collection.count()}")
print()

print("🎉 ChromaDB initialization complete!")
print()
print("📂 Database files created in:")
print(f"   {os.path.abspath(data_dir)}")
print()
print("📊 You can now:")
print("   1. Run your AgentSync application")
print("   2. Agents will automatically use this ChromaDB instance")
print("   3. Context will be persisted across restarts")
