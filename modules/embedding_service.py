import logging
from typing import List, Dict
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddingService:
    """Local embedding service using sentence-transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = None):
        """
        Initialize local embedding service
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load the sentence transformer model
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Successfully loaded model: {model_name}")
            
            # Get embedding dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert numpy array to list
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            else:
                return list(embedding)
                
        except Exception as e:
            logger.error(f"Error getting embedding for text: {e}")
            raise Exception(f"Failed to get embedding: {str(e)}")
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            # Generate embeddings for all texts at once
            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
            
            # Convert to list of lists
            if len(embeddings.shape) == 1:
                # Single text case
                return [embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)]
            else:
                # Multiple texts case
                return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
                
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {e}")
            raise Exception(f"Failed to get batch embeddings: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dimension
    
    def test_model(self) -> bool:
        """
        Test the model by generating a test embedding
        
        Returns:
            True if model is working, False otherwise
        """
        try:
            test_text = "This is a test sentence for embedding generation."
            embedding = self.get_embedding(test_text)
            return len(embedding) > 0 and len(embedding) == self.embedding_dimension
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.embedding_dimension,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'Unknown'),
            "model_type": "sentence-transformers"
        }

# For backward compatibility, create an alias
HuggingFaceEmbeddingService = LocalEmbeddingService