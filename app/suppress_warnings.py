"""
Suppress non-critical warnings and noisy logging
"""
import warnings
import logging

def setup_warnings():
    """
    Suppress ChromaDB telemetry warnings and other non-critical output
    """
    # Suppress ChromaDB telemetry event warnings
    warnings.filterwarnings("ignore", message=".*telemetry event.*")
    
    # Suppress ChromaDB logging (very noisy)
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("chromadb.db.impl").setLevel(logging.ERROR)
    
    # Suppress Starlette WebSocket warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
