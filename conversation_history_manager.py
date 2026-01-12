"""
Conversation History Manager - GROUPED VERSION
Works with Docker Redis without needing MessageQueueService

GROUPED DISPLAY: User and Assistant messages share the same index
"""

import json
import logging
import asyncio
import redis.asyncio as redis
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class ConversationHistoryManager:
    """
    Manages conversation history stored in Redis
    
    Features:
    - Load conversation history
    - Display history with GROUPED format (user+assistant pairs)
    - Delete specific conversation pair by index
    - Delete all history with confirmation
    """
    
    def __init__(self, redis_client):
        """
        Initialize history manager
        
        Args:
            redis_client: Redis client instance
        """
        self.redis_client = redis_client
        self.history_key = "conversation_history"
    
    
    # ==================== LOAD HISTORY ====================
    
    async def load_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Load conversation history from Redis
        """
        try:
            history_json = await self.redis_client.get(self.history_key)
            
            if history_json:
                if isinstance(history_json, bytes):
                    history_json = history_json.decode('utf-8')
                
                history = json.loads(history_json)
                print(f"[HISTORY] ✅ Loaded {len(history)} messages from Redis")
                logger.info(f"[HISTORY] ✅ Loaded {len(history)} messages from Redis")
                return history
            
            print("[HISTORY] No previous history found in Redis")
            logger.info("[HISTORY] No previous history found in Redis")
            return []
            
        except Exception as e:
            print(f"[HISTORY] ⚠️  Could not load history from Redis: {e}")
            logger.warning(f"[HISTORY] ⚠️  Could not load history from Redis: {e}")
            return []
    
    
    # ==================== HELPER: GET CONVERSATION PAIRS ====================
    
    def _get_conversation_pairs(self, history: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """
        Helper method to identify conversation pair boundaries
        
        Returns:
            List of tuples: [(start_idx, end_idx), ...]
            Example: [(0, 2), (2, 4), (4, 5)] means:
                - Pair 0: indices 0-1 (user + assistant)
                - Pair 1: indices 2-3 (user + assistant)  
                - Pair 2: index 4 (single user message)
        """
        pairs = []
        i = 0
        
        while i < len(history):
            start = i
            message = history[i]
            role = message.get("role", "unknown").upper()
            
            # Check if current is USER and next is ASSISTANT
            if (role == "USER" and 
                i + 1 < len(history) and 
                history[i + 1].get("role", "unknown").upper() == "ASSISTANT"):
                # This is a user+assistant pair
                pairs.append((start, i + 2))  # (start_idx, end_idx)
                i += 2
            else:
                # Single message or orphaned assistant
                pairs.append((start, i + 1))
                i += 1
        
        return pairs
    
    
    # ==================== DISPLAY HISTORY (GROUPED) ====================
    
    async def display_conversation_history(self) -> None:
        """
        Load and display conversation history in GROUPED format
        Groups user and assistant messages together with same index
        """
        try:
            history = await self.load_conversation_history()
            
            if not history:
                print("\n" + "="*70)
                print("CONVERSATION HISTORY")
                print("="*70)
                print("📭 No history found\n")
                return
            
            pairs = self._get_conversation_pairs(history)
            
            print("\n" + "="*70)
            print("CONVERSATION HISTORY")
            print("="*70)
            print(f"Total messages: {len(history)}")
            print(f"Total conversation pairs: {len(pairs)}\n")
            
            # Display each pair grouped
            for pair_idx, (start, end) in enumerate(pairs):
                # First message in pair
                message = history[start]
                role = message.get("role", "unknown").upper()
                content = message.get("content", "")
                timestamp = message.get("timestamp", "N/A")
                
                display_content = content
                role_symbol = "👤" if role == "USER" else "🤖" if role == "ASSISTANT" else "❓"
                
                # Print first message with pair index
                print(f"[{pair_idx}] {role_symbol} {role}")
                print(f"    Content: {display_content}")
                if timestamp != "N/A":
                    print(f"    Time:    {timestamp}")
                
                # If this is a pair (user+assistant), show the second message
                if end - start == 2:
                    print(f"    ↓")
                    
                    next_message = history[start + 1]
                    next_role = next_message.get("role", "unknown").upper()
                    next_content = next_message.get("content", "")
                    next_timestamp = next_message.get("timestamp", "N/A")
                    
                    next_display_content = next_content
                    next_role_symbol = "🤖" if next_role == "ASSISTANT" else "👤" if next_role == "USER" else "❓"
                    
                    print(f"    {next_role_symbol} {next_role}")
                    print(f"    Content: {next_display_content}")
                    if next_timestamp != "N/A":
                        print(f"    Time:    {next_timestamp}")
                
                print()
                print("="*120)
                print()
            
            logger.info(f"[HISTORY] Displayed {len(history)} messages in {len(pairs)} pairs")
            
        except Exception as e:
            logger.error(f"[HISTORY] Failed to display history: {e}", exc_info=True)
            print(f"\n❌ Error displaying history: {e}\n")
    
    
    # ==================== DELETE BY PAIR INDEX ====================
    
    async def delete_history_by_index(self, pair_index: int) -> bool:
        """
        Delete specific conversation PAIR from history by pair index
        When deleting a pair, deletes both user and assistant messages
        """
        try:
            history = await self.load_conversation_history()
            
            if not history:
                print("[HISTORY] Cannot delete - no history found")
                print("❌ No history to delete\n")
                logger.warning("[HISTORY] Cannot delete - no history found")
                return False
            
            pairs = self._get_conversation_pairs(history)
            
            if pair_index < 0 or pair_index >= len(pairs):
                print(f"[HISTORY] Invalid pair index: {pair_index} (total pairs: {len(pairs)})")
                print(f"❌ Invalid index: {pair_index}. Valid range: 0-{len(pairs)-1}\n")
                logger.warning(f"[HISTORY] Invalid pair index: {pair_index}")
                return False
            
            # Get the pair to delete
            start, end = pairs[pair_index]
            pair_size = end - start
            
            # Get message info for logging
            deleted_messages = history[start:end]
            first_role = deleted_messages[0].get("role", "unknown")
            first_content = deleted_messages[0].get("content", "")[:50]
            
            # Delete messages in reverse order to maintain indices
            for i in range(end - 1, start - 1, -1):
                history.pop(i)
            
            # Save updated history
            history_json = json.dumps(history)
            await self.redis_client.setex(
                self.history_key,
                86400,
                history_json
            )
            
            print(f"[HISTORY] ✅ Deleted conversation pair at index {pair_index}")
            print(f"✅ Deleted pair [{pair_index}] ({pair_size} messages: {first_role}...)\n")
            logger.info(f"[HISTORY] ✅ Deleted conversation pair at index {pair_index}")
            return True
            
        except Exception as e:
            logger.error(f"[HISTORY] Failed to delete pair by index: {e}", exc_info=True)
            print(f"❌ Error deleting pair: {e}\n")
            return False
    
    
    # ==================== DELETE ALL HISTORY ====================
    
    async def delete_all_history(self, confirm: bool = False) -> bool:
        """
        Delete ALL conversation history from Redis
        """
        try:
            history = await self.load_conversation_history()
            
            if not history:
                print("[HISTORY] No history to delete")
                print("ℹ️  No history found\n")
                logger.info("[HISTORY] No history to delete")
                return True
            
            count = len(history)
            pairs = self._get_conversation_pairs(history)
            
            if not confirm:
                print(f"\n⚠️  WARNING: This will delete all {count} messages ({len(pairs)} pairs) permanently!")
                confirmation = input("Continue? (yes/no): ").strip().lower()
                
                if confirmation != "yes":
                    print("❌ Cancelled\n")
                    logger.info("[HISTORY] User cancelled delete all operation")
                    return False
            
            await self.redis_client.delete(self.history_key)
            
            print(f"[HISTORY] ✅ Deleted all {count} messages ({len(pairs)} pairs) from Redis")
            print(f"✅ Deleted all {count} messages\n")
            logger.info(f"[HISTORY] ✅ Deleted all {count} messages from Redis")
            return True
            
        except Exception as e:
            logger.error(f"[HISTORY] Failed to delete all history: {e}", exc_info=True)
            print(f"❌ Error deleting history: {e}\n")
            return False
    
    
    # ==================== UTILITY FUNCTIONS ====================
    
    async def get_history_count(self) -> int:
        """
        Get total number of messages in history
        """
        try:
            history = await self.load_conversation_history()
            return len(history)
        except Exception as e:
            logger.error(f"[HISTORY] Failed to get history count: {e}")
            return 0
    
    
    async def get_conversation_pairs_count(self) -> int:
        """
        Get total number of conversation pairs (user+assistant groups)
        """
        try:
            history = await self.load_conversation_history()
            if not history:
                return 0
            
            pairs = self._get_conversation_pairs(history)
            return len(pairs)
        except Exception as e:
            logger.error(f"[HISTORY] Failed to get pair count: {e}")
            return 0
    
    
    async def get_message_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get specific message from history by index
        """
        try:
            history = await self.load_conversation_history()
            
            if 0 <= index < len(history):
                return history[index]
            
            print(f"[HISTORY] Invalid index: {index}")
            logger.warning(f"[HISTORY] Invalid index: {index}")
            return None
            
        except Exception as e:
            logger.error(f"[HISTORY] Failed to get message by index: {e}")
            return None
    
    
    async def clear_old_messages(self, keep_count: int = 10) -> int:
        """
        Keep only the last N conversation PAIRS, delete older ones
        """
        try:
            history = await self.load_conversation_history()
            
            if not history:
                print(f"[HISTORY] No messages to cleanup")
                logger.info(f"[HISTORY] No messages to cleanup")
                return 0
            
            pairs = self._get_conversation_pairs(history)
            
            if len(pairs) <= keep_count:
                print(f"[HISTORY] No cleanup needed ({len(pairs)} pairs)")
                logger.info(f"[HISTORY] No cleanup needed ({len(pairs)} pairs)")
                return 0
            
            # Calculate how many pairs to delete
            pairs_to_delete_count = len(pairs) - keep_count
            
            # Get the start index of the first pair to keep
            pairs_to_delete = pairs[:pairs_to_delete_count]
            first_pair_to_keep_start, _ = pairs[pairs_to_delete_count]
            
            # Count deleted messages
            deleted_message_count = first_pair_to_keep_start
            
            # Create new history with only last keep_count pairs
            new_history = history[first_pair_to_keep_start:]
            
            history_json = json.dumps(new_history)
            await self.redis_client.setex(
                self.history_key,
                86400,
                history_json
            )
            
            print(f"[HISTORY] ✅ Cleaned up {deleted_message_count} messages ({pairs_to_delete_count} pairs)")
            print(f"✅ Cleaned up {deleted_message_count} messages (kept {keep_count} pairs)\n")
            logger.info(f"[HISTORY] ✅ Cleaned up {deleted_message_count} messages")
            return deleted_message_count
            
        except Exception as e:
            logger.error(f"[HISTORY] Failed to cleanup old messages: {e}", exc_info=True)
            return 0


# ==================== INTERACTIVE CLI ====================

async def history_management_cli(redis_client):
    """
    Interactive CLI for managing conversation history
    """
    manager = ConversationHistoryManager(redis_client)
    
    while True:
        print("\n" + "="*70)
        print("CONVERSATION HISTORY MANAGEMENT")
        print("="*70)
        print("1. Display all history")
        print("2. Delete specific conversation pair by index")
        print("3. Delete all history")
        print("4. Get history statistics")
        print("5. Keep only last N conversation pairs")
        print("6. Exit")
        print("="*70)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            await manager.display_conversation_history()
        
        elif choice == "2":
            pair_count = await manager.get_conversation_pairs_count()
            if pair_count > 0:
                try:
                    idx = int(input(f"Enter pair index to delete (0-{pair_count-1}): ").strip())
                    await manager.delete_history_by_index(idx)
                except ValueError:
                    print("❌ Invalid index\n")
            else:
                print("❌ No conversation pairs to delete\n")
        
        elif choice == "3":
            await manager.delete_all_history(confirm=False)
        
        elif choice == "4":
            total_messages = await manager.get_history_count()
            total_pairs = await manager.get_conversation_pairs_count()
            print(f"\n📊 Statistics:")
            print(f"   Total messages: {total_messages}")
            print(f"   Conversation pairs: {total_pairs}\n")
        
        elif choice == "5":
            try:
                keep = int(input("How many recent conversation pairs to keep? ").strip())
                await manager.clear_old_messages(keep_count=keep)
            except ValueError:
                print("❌ Invalid number\n")
        
        elif choice == "6":
            print("\n✅ Goodbye!\n")
            break
        
        else:
            print("❌ Invalid choice\n")


# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    """
    Direct execution entry point - GROUPED VERSION
    
    Works with Docker Redis
    Groups user-assistant messages with same index
    """
    
    async def main():
        print("\n" + "="*70)
        print("CONVERSATION HISTORY MANAGER (GROUPED)")
        print("="*70)
        print("Connecting to Redis...\n")
        
        try:
            # Connect to Docker Redis
            # Try common Docker Redis URLs
            redis_urls = [
                "redis://localhost:6379",      # Local Docker
                "redis://host.docker.internal:6379",  # Docker Desktop
                "redis://redis:6379",          # Docker compose service name
            ]
            
            redis_client = None
            
            for url in redis_urls:
                try:
                    print(f"[INIT] Trying: {url}")
                    redis_client = await redis.from_url(
                        url,
                        encoding="utf8",
                        decode_responses=True,
                        socket_connect_timeout=3
                    )
                    await redis_client.ping()
                    print(f"[INIT] ✅ Connected to Redis at {url}\n")
                    break
                except Exception as e:
                    print(f"[INIT] ❌ Failed: {e}")
                    continue
            
            if redis_client is None:
                print("\n[INIT] ❌ Could not connect to Redis")
                print("Please make sure Redis is running on Docker")
                print("\nFor Docker:")
                print("  docker run -d -p 6379:6379 redis:latest")
                return
            
            print("="*70)
            print("Starting interactive CLI...\n")
            
            await history_management_cli(redis_client)
            
            print("[CLEANUP] Closing Redis connection...")
            await redis_client.close()
            print("[CLEANUP] ✅ Done\n")
            
        except KeyboardInterrupt:
            print("\n\n✅ Interrupted by user\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()

    asyncio.run(main())