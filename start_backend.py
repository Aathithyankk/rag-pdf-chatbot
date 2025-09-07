#!/usr/bin/env python3
"""
Startup script for the RAG PDF Chatbot backend server
"""

import uvicorn
import os
import sys
import multiprocessing
from pathlib import Path
from config import Config

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Start the FastAPI backend server"""
    # Get configuration
    workers = Config.WORKERS
    host = Config.HOST
    port = Config.PORT
    reload = Config.RELOAD
    
    print("Starting RAG PDF Chatbot Backend Server...")
    print(f"Backend will be available at: http://{host}:{port}")
    print(f"API documentation at: http://{host}:{port}/docs")
    print(f"Health check at: http://{host}:{port}/health")
    print(f"Workers: {workers} (CPU cores: {multiprocessing.cpu_count()})")
    print(f"Reload mode: {'Enabled' if reload else 'Disabled'}")
    print("\n" + "="*50)
    
    try:
        if workers > 1 and not reload:
            # Production mode with multiple workers
            print(f"Starting with {workers} workers for better performance...")
            uvicorn.run(
                "app_backend:app",
                host=host,
                port=port,
                workers=workers,
                log_level="info",
                access_log=True
            )
        else:
            # Development mode with single worker and reload
            print("Starting in development mode (single worker with reload)...")
            uvicorn.run(
                "app_backend:app",
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
    except KeyboardInterrupt:
        print("\n Backend server stopped.")
    except Exception as e:
        print(f" Error starting backend server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
