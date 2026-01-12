from fastapi import APIRouter
from app.database.connection import check_database_health

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
def health_check():
    # Composite health status for DB, Redis, ChromaDB
    status = check_database_health()
    return {"status": "healthy" if all(status.values()) else "unhealthy", "details": status}
