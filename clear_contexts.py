"""
Clear contexts from ChromaDB and SQLite database
Supports clearing all contexts or specific contexts by embedding_id

USAGE:
    python clear_contexts.py
    
    Then follow the menu:
    1. Enter option 1 to clear specific contexts (provide comma-separated embedding IDs)
    2. Enter option 2 to clear ALL contexts (will ask for confirmation)
"""
import sys
import logging
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_specific_contexts(embedding_ids: List[str], chroma_service, context_db_service) -> None:
    """
    Clear specific contexts from ChromaDB and SQLite by embedding_ids
    
    Args:
        embedding_ids: List of vector IDs (embedding_id) to delete
        chroma_service: ChromaDB service instance
        context_db_service: Context DB service instance
    """
    if not embedding_ids:
        logger.warning("[CLEAR] No embedding IDs provided")
        return
    
    logger.info(f"[CLEAR] Starting deletion of {len(embedding_ids)} context(s)...")
    
    # Delete from ChromaDB
    try:
        logger.info("[CLEAR] Deleting from ChromaDB...")
        deleted_count_chroma = 0
        
        for embedding_id in embedding_ids:
            try:
                # ChromaDB delete by ID
                chroma_service.collection.delete(ids=[embedding_id])
                deleted_count_chroma += 1
                logger.info(f"[CLEAR] ✅ Deleted from ChromaDB: {embedding_id}")
            except Exception as e:
                logger.warning(f"[CLEAR] ⚠️  Failed to delete from ChromaDB ({embedding_id}): {e}")
        
        logger.info(f"[CLEAR] ✅ Deleted {deleted_count_chroma}/{len(embedding_ids)} contexts from ChromaDB")
        
    except Exception as e:
        logger.error(f"[CLEAR] ❌ Error deleting from ChromaDB: {e}", exc_info=True)
    
    # Delete from SQLite
    try:
        logger.info("[CLEAR] Deleting from SQLite (agent_contexts table)...")
        deleted_count_sqlite = 0
        
        for embedding_id in embedding_ids:
            try:
                # Delete from SQLite by embedding_id
                context_db_service.delete_by_embedding_id(embedding_id)
                deleted_count_sqlite += 1
                logger.info(f"[CLEAR] ✅ Deleted from SQLite: {embedding_id}")
            except Exception as e:
                logger.warning(f"[CLEAR] ⚠️  Failed to delete from SQLite ({embedding_id}): {e}")
        
        logger.info(f"[CLEAR] ✅ Deleted {deleted_count_sqlite}/{len(embedding_ids)} contexts from SQLite")
        
    except Exception as e:
        logger.error(f"[CLEAR] ❌ Error deleting from SQLite: {e}", exc_info=True)


def clear_all_contexts(chroma_service, context_db_service) -> None:
    """
    Clear ALL contexts from ChromaDB and SQLite database
    
    Args:
        chroma_service: ChromaDB service instance
        context_db_service: Context DB service instance
    """
    logger.info("[CLEAR] Starting deletion of ALL contexts...")
    
    # Delete all from ChromaDB
    try:
        logger.info("[CLEAR] Deleting ALL contexts from ChromaDB...")
        
        # Get collection and delete all
        collection = chroma_service.collection
        all_ids = collection.get(include=[])["ids"]
        
        if all_ids:
            collection.delete(ids=all_ids)
            logger.info(f"[CLEAR] ✅ Deleted {len(all_ids)} contexts from ChromaDB")
        else:
            logger.info("[CLEAR] ℹ️  No contexts found in ChromaDB")
        
    except Exception as e:
        logger.error(f"[CLEAR] ❌ Error deleting ALL from ChromaDB: {e}", exc_info=True)
    
    # Delete all from SQLite
    try:
        logger.info("[CLEAR] Deleting ALL contexts from SQLite...")
        
        deleted_count = context_db_service.delete_all_contexts()
        logger.info(f"[CLEAR] ✅ Deleted {deleted_count} contexts from SQLite")
        
    except Exception as e:
        logger.error(f"[CLEAR] ❌ Error deleting ALL from SQLite: {e}", exc_info=True)


def get_user_choice() -> tuple[int, Optional[List[str]]]:
    """
    Display menu and get user choice
    
    Returns:
        tuple: (choice, embedding_ids)
            - choice: 1 for specific, 2 for all
            - embedding_ids: List of IDs if choice is 1, None if choice is 2
    """
    print("\n" + "="*60)
    print("CONTEXT MANAGEMENT - CLEAR OPTIONS")
    print("="*60)
    print("1. Clear specific context(s) by embedding ID")
    print("2. Clear ALL contexts")
    print("="*60)
    
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == "1":
                embedding_ids = input("\nEnter embedding ID(s) (comma-separated): ").strip()
                if not embedding_ids:
                    print("❌ No IDs provided. Please try again.")
                    continue
                
                ids_list = [id.strip() for id in embedding_ids.split(",")]
                print(f"\n✅ You selected: Clear {len(ids_list)} specific context(s)")
                print(f"   Embedding IDs: {ids_list}")
                return 1, ids_list
            
            elif choice == "2":
                confirm = input("\n⚠️  WARNING: This will delete ALL contexts. Continue? (yes/no): ").strip().lower()
                if confirm == "yes":
                    print("\n✅ You selected: Clear ALL contexts")
                    return 2, None
                else:
                    print("❌ Cancelled.")
                    continue
            
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
                continue
        
        except KeyboardInterrupt:
            print("\n\n❌ Operation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


def main():
    """
    Main function to handle context clearing
    """
    try:
        # Import services
        from app.services.chroma_service import ChromaDBService
        from app.services.context_db_service import ContextDBService
        
        logger.info("[CLEAR] Initializing services...")
        
        # Initialize services
        chroma_service = ChromaDBService()
        context_db_service = ContextDBService()
        
        logger.info("[CLEAR] Services initialized successfully")
        
        # Get user choice
        choice, embedding_ids = get_user_choice()
        
        # Execute based on choice
        if choice == 1:
            logger.info(f"[CLEAR] Processing {len(embedding_ids)} specific context(s)...")
            clear_specific_contexts(embedding_ids, chroma_service, context_db_service)
        
        elif choice == 2:
            logger.info("[CLEAR] Processing deletion of ALL contexts...")
            clear_all_contexts(chroma_service, context_db_service)
        
        logger.info("[CLEAR] ✅ Operation completed successfully")
        print("\n✅ Context clearing completed successfully!")
        
    except ImportError as e:
        logger.error(f"[CLEAR] ❌ Import error: {e}", exc_info=True)
        print(f"\n❌ Error importing services: {e}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"[CLEAR] ❌ Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()