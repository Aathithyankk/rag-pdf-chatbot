import google.generativeai as genai
import logging
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLLMService:
    """Gemini 2.5 Flash LLM service for generating responses"""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini LLM service
        
        Args:
            api_key: Google AI API key
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key or os.getenv("GOOGLE_AI_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("Google AI API key is required. Set GOOGLE_AI_API_KEY environment variable.")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        
        # Configure generation parameters
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        logger.info(f"Initialized Gemini LLM service with model: {model_name}")
    
    def generate_response(self, prompt: str, context: str = "", chat_history: List[Dict] = None) -> str:
        """
        Generate a response using Gemini
        
        Args:
            prompt: User's question/prompt
            context: Relevant context from RAG
            chat_history: Previous conversation history
            
        Returns:
            Generated response
        """
        try:
            # Build the full prompt with context and history
            full_prompt = self._build_prompt(prompt, context, chat_history)
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=self.generation_config
            )
            
            if response.text:
                logger.info("Successfully generated response from Gemini")
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini")
                return "I apologize, but I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _build_prompt(self, prompt: str, context: str = "", chat_history: List[Dict] = None) -> str:
        """
        Build the full prompt with context and chat history
        
        Args:
            prompt: User's current question
            context: Relevant context from documents
            chat_history: Previous conversation
            
        Returns:
            Formatted prompt
        """
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents. 
        Follow these guidelines:
        1. Use only the information provided in the context to answer questions
        2. If the context doesn't contain enough information, say so clearly
        3. Be concise but comprehensive in your answers
        4. If asked about something not in the context, politely explain that you can only answer based on the uploaded documents
        5. Maintain a helpful and professional tone
        """
        
        full_prompt = f"{system_prompt}\n\n"
        
        # Add context if available
        if context:
            full_prompt += f"Context from documents:\n{context}\n\n"
        
        # Add chat history if available
        if chat_history:
            full_prompt += "Previous conversation:\n"
            for msg in chat_history[-5:]:  # Only include last 5 messages to avoid token limits
                role = "User" if msg.get("role") == "user" else "Assistant"
                full_prompt += f"{role}: {msg.get('content', '')}\n"
            full_prompt += "\n"
        
        # Add current question
        full_prompt += f"Current question: {prompt}\n\n"
        full_prompt += "Answer:"
        
        return full_prompt
    
    def test_connection(self) -> bool:
        """
        Test the connection to Gemini API
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_prompt = "Hello, this is a test message."
            response = self.generate_response(test_prompt)
            return len(response) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def update_generation_config(self, **kwargs):
        """
        Update generation configuration
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.generation_config.update(kwargs)
        logger.info(f"Updated generation config: {self.generation_config}")
