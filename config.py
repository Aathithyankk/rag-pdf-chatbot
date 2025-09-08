import os
import multiprocessing
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the RAG PDF Chatbot"""
    
    # API Keys
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
    
    # Backend Configuration
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    # Model Configurations
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")
    
    # Vector Store Configuration
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./storage/chroma_db")
    
    # PDF Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # RAG Configuration
    MAX_RETRIEVAL_RESULTS = int(os.getenv("MAX_RETRIEVAL_RESULTS", "5"))
    
    # Server Configuration
    WORKERS = int(os.getenv("WORKERS", multiprocessing.cpu_count()))
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    RELOAD = os.getenv("RELOAD", "false").lower() == "true"
    
    # Local Embedding Configuration
    @classmethod
    def get_embedding_device(cls):
        """Get the appropriate device for embeddings"""
        device_setting = os.getenv("EMBEDDING_DEVICE", "auto")
        
        if device_setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device_setting in ["cpu", "cuda", "mps"]:
            return device_setting
        else:
            # Fallback to auto-detection
            return "cuda" if torch.cuda.is_available() else "cpu"
    
    USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"