
import asyncio
import logging
import json
from typing import Optional

from app.services.message_queue import MessageQueueService
from app.agents.registry import agent_registry

logger = logging.getLogger(__name__)


class AgentWorkerService:
    """
    Worker service that processes agent tasks from Redis
    Listens to email_tasks, calendar_tasks, slack_tasks queues
    """
    
    def __init__(self, mq_service: MessageQueueService):
        self.mq_service = mq_service
        self.running = False
        self.workers = {}
    
    async def start_workers(self):
        """Start worker tasks for each agent"""
        logger.info("[WORKER] Starting agent workers...")
        
        self.running = True
        
        # Get all registered agents
        agents = agent_registry.get_all_agents()
        
        for agent in agents:
            queue_key = f"{agent.agent_type}_tasks"
            task = asyncio.create_task(self._worker_loop(agent, queue_key))
            self.workers[agent.agent_type] = task
            logger.info(f"[WORKER] Started worker for {agent.agent_type} (queue: {queue_key})")
    
    async def stop_workers(self):
        """Stop all worker tasks"""
        logger.info("[WORKER] Stopping workers...")
        self.running = False
        
        for agent_type, task in self.workers.items():
            task.cancel()
            logger.info(f"[WORKER] Stopped {agent_type} worker")
    
    async def _worker_loop(self, agent, queue_key: str):
        """
        Worker loop for a specific agent
        Continuously polls Redis queue for tasks
        """
        logger.info(f"[WORKER] {agent.agent_type.upper()} worker started")
        
        while self.running:
            try:
                # Dequeue task
                task_data = await self.mq_service.dequeue_task(queue_key)
                
                if task_data:
                    task_id = task_data.get('task_id')
                    logger.info(f"[WORKER] Processing task for {agent.agent_type}: {task_data.get('action')}")
                    
                    try:
                        # Execute task with agent
                        result = await agent.execute(task_data)
                        
                        # ✅ FIXED: Pass task_id (not task_id_result)
                        # store_result() will add _result suffix internally
                        await self.mq_service.store_result(task_id, result)
                        
                        logger.info(f"[WORKER] Task {task_id} completed")
                        
                    except Exception as e:
                        logger.error(f"[WORKER] Error executing task: {e}", exc_info=True)
                        
                        # Store error result
                        error_result = {
                            "status": "failure",
                            "error": str(e),
                            "task_id": task_id
                        }
                        # ✅ FIXED: Pass task_id (not task_id_result)
                        await self.mq_service.store_result(task_id, error_result)
                else:
                    # No task, wait before polling again
                    await asyncio.sleep(0.5)
                    
            except asyncio.CancelledError:
                logger.info(f"[WORKER] {agent.agent_type} worker cancelled")
                break
            except Exception as e:
                logger.error(f"[WORKER] Unexpected error: {e}", exc_info=True)
                await asyncio.sleep(1)


# Global worker service instance
_worker_service: Optional[AgentWorkerService] = None


async def get_worker_service(mq_service: MessageQueueService) -> AgentWorkerService:
    """Get or create worker service"""
    global _worker_service
    if _worker_service is None:
        _worker_service = AgentWorkerService(mq_service)
    return _worker_service
