"""
Redis-based message queue service for task distribution
"""
import json
import logging
from typing import Dict, Any, Optional
import redis.asyncio as redis
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MessageQueueService:
    
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Connect to Redis"""
        if not self.redis_client:
            self.redis_client = await redis.from_url(self.redis_url, decode_responses=True)
            logger.info(f"[MQ] Connected to Redis at {self.redis_url}")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("[MQ] Disconnected from Redis")
    
    async def enqueue_task(self, queue_key: str, task: Dict[str, Any]) -> str:
        
        if not self.redis_client:
            await self.connect()
        
        task_json = json.dumps(task)
        task_id = task.get("task_id", "")
        
        # Push to queue
        await self.redis_client.rpush(queue_key, task_json)
        logger.info(f"[MQ] Enqueued task {task_id} to {queue_key}")
        
        return task_id
    
    async def dequeue_task(self, queue_key: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
        
        if not self.redis_client:
            await self.connect()
        
        try:
            # Use BLPOP for blocking pop
            result = await self.redis_client.blpop(queue_key, timeout=timeout)
            
            if result:
                queue_name, task_json = result
                task = json.loads(task_json)
                logger.info(f"[MQ] Dequeued task {task.get('task_id')} from {queue_key}")
                return task
            
            return None
            
        except Exception as e:
            logger.error(f"[MQ] Error dequeuing from {queue_key}: {e}")
            return None
    
    async def store_result(
        self,
        task_id: str,
        result: Dict[str, Any],
        ttl: int = 300
    ) -> bool:
        """
        Store task result in Redis
        
        Args:
            task_id: Task identifier
            result: Result dictionary
            ttl: Time to live in seconds
        
        Returns:
            True if successful
        """
        if not self.redis_client:
            await self.connect()
        
        try:
            result_key = f"{task_id}_result"
            result_json = json.dumps(result)
            
            # Store with TTL
            await self.redis_client.setex(result_key, ttl, result_json)
            logger.info(f"[MQ] Stored result for task {task_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"[MQ] Error storing result for {task_id}: {e}")
            return False
    
    async def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve task result from Redis
        
        Args:
            task_id: Task identifier (or full key with _result suffix)
        
        Returns:
            Result dictionary or None
        """
        if not self.redis_client:
            await self.connect()
        
        try:
            # Handle both plain task_id and full key
            result_key = task_id if task_id.endswith("_result") else f"{task_id}_result"
            
            result_json = await self.redis_client.get(result_key)
            
            if result_json:
                result = json.loads(result_json)
                logger.info(f"[MQ] Retrieved result for {task_id}")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"[MQ] Error retrieving result for {task_id}: {e}")
            return None
    
    async def get_queue_length(self, queue_key: str) -> int:
        """Get number of tasks in queue"""
        if not self.redis_client:
            await self.connect()
        
        try:
            length = await self.redis_client.llen(queue_key)
            return length
        except Exception as e:
            logger.error(f"[MQ] Error getting queue length for {queue_key}: {e}")
            return 0
    
    async def clear_queue(self, queue_key: str) -> bool:
        """Clear all tasks from queue"""
        if not self.redis_client:
            await self.connect()
        
        try:
            await self.redis_client.delete(queue_key)
            logger.info(f"[MQ] Cleared queue {queue_key}")
            return True
        except Exception as e:
            logger.error(f"[MQ] Error clearing queue {queue_key}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check Redis connection health"""
        if not self.redis_client:
            await self.connect()
        
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"[MQ] Redis health check failed: {e}")
            return False
