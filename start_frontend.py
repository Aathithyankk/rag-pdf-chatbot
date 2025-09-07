#!/usr/bin/env python3
"""
Startup script for the RAG PDF Chatbot frontend application
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the Streamlit frontend application"""
    print("Starting RAG PDF Chatbot Frontend...")
    print("Frontend will be available at: http://localhost:8501")
    print("Make sure the backend server is running first!")
    print("\n" + "="*50)
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app_streamlit.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("Frontend application stopped.")
    except Exception as e:
        print(f"Error starting frontend application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
