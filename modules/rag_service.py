import logging
from typing import List, Dict, Optional
from .pdf_loader import PDFLoader
from .vector_store import ChromaVectorStore
from .embedding_service import LocalEmbeddingService
from .llm_service import GeminiLLMService
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """RAG (Retrieval-Augmented Generation) service combining vector search and LLM"""
    
    def __init__(self, 
                 huggingface_api_key: str = None,
                 google_ai_api_key: str = None,
                 embedding_model: str = None,
                 llm_model: str = None,
                 vector_store_path: str = None,
                 use_local_embeddings: bool = None):
        """
        Initialize RAG service
        
        Args:
            huggingface_api_key: Hugging Face API key (not needed for local embeddings)
            google_ai_api_key: Google AI API key
            embedding_model: Embedding model name
            llm_model: Gemini model name
            vector_store_path: Path for Chroma vector store
            use_local_embeddings: Whether to use local embeddings (default: True)
        """
        # Use config defaults if not provided
        embedding_model = embedding_model or Config.EMBEDDING_MODEL
        llm_model = llm_model or Config.LLM_MODEL
        vector_store_path = vector_store_path or Config.VECTOR_STORE_PATH
        use_local_embeddings = use_local_embeddings if use_local_embeddings is not None else Config.USE_LOCAL_EMBEDDINGS
        
        # Initialize components
        self.pdf_loader = PDFLoader()
        self.vector_store = ChromaVectorStore(persist_directory=vector_store_path)
        
        # Initialize embedding service (local by default)
        if use_local_embeddings:
            logger.info("Using local embedding service")
            self.embedding_service = LocalEmbeddingService(
                model_name=embedding_model,
                device=Config.EMBEDDING_DEVICE
            )
        else:
            logger.info("Using Hugging Face API embedding service")
            self.embedding_service = LocalEmbeddingService(
                model_name=embedding_model,
                device=Config.EMBEDDING_DEVICE
            )
        
        # Initialize LLM service
        self.llm_service = GeminiLLMService(
            api_key=google_ai_api_key,
            model_name=llm_model
        )
        
        logger.info("RAG service initialized successfully")
    
    def process_pdf(self, pdf_file) -> Dict:
        """
        Process a PDF file and add it to the vector store
        
        Args:
            pdf_file: Uploaded PDF file
            
        Returns:
            Processing result with status and details
        """
        try:
            # Load PDF
            logger.info(f"Processing PDF: {pdf_file.filename}")
            pages = self.pdf_loader.load_pdf(pdf_file)
            
            if not pages:
                return {
                    "success": False,
                    "message": "No content found in PDF",
                    "pages_processed": 0,
                    "chunks_created": 0
                }
            
            # Chunk the text
            chunks = self.pdf_loader.chunk_text(pages)
            
            if not chunks:
                return {
                    "success": False,
                    "message": "Failed to create text chunks",
                    "pages_processed": len(pages),
                    "chunks_created": 0
                }
            
            # Get embeddings for chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_service.get_embeddings_batch(chunk_texts)
            
            # Add to vector store
            success = self.vector_store.add_documents(chunks, embeddings)
            
            if success:
                return {
                    "success": True,
                    "message": f"Successfully processed PDF: {pdf_file.filename}",
                    "pages_processed": len(pages),
                    "chunks_created": len(chunks),
                    "filename": pdf_file.filename
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to add documents to vector store",
                    "pages_processed": len(pages),
                    "chunks_created": len(chunks)
                }
                
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {
                "success": False,
                "message": f"Error processing PDF: {str(e)}",
                "pages_processed": 0,
                "chunks_created": 0
            }
    
    def ask_question(self, question: str, chat_history: List[Dict] = None, 
                    n_results: int = 5) -> Dict:
        """
        Ask a question and get an answer using RAG
        
        Args:
            question: User's question
            chat_history: Previous conversation history
            n_results: Number of similar documents to retrieve
            
        Returns:
            Answer with sources and metadata
        """
        try:
            # Get embedding for the question
            question_embedding = self.embedding_service.get_embedding(question)
            
            # Search for similar documents
            similar_docs = self.vector_store.search_similar(
                query_embedding=question_embedding,
                n_results=n_results
            )
            
            if not similar_docs:
                return {
                    "success": False,
                    "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    "sources": [],
                    "context_used": ""
                }
            
            # Build context from similar documents
            context = self._build_context(similar_docs)
            
            # Generate answer using LLM
            answer = self.llm_service.generate_response(
                prompt=question,
                context=context,
                chat_history=chat_history
            )
            
            # Prepare sources
            sources = []
            for doc in similar_docs:
                sources.append({
                    "content": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    "metadata": doc['metadata'],
                    "similarity_score": 1 - doc['distance'] if 'distance' in doc else 0
                })
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "context_used": context,
                "similar_docs_found": len(similar_docs)
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "success": False,
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "context_used": ""
            }
    
    def _build_context(self, similar_docs: List[Dict]) -> str:
        """
        Build context string from similar documents
        
        Args:
            similar_docs: List of similar documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(similar_docs, 1):
            metadata = doc.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'Unknown')
            
            context_parts.append(
                f"Source {i} (from {source}, page {page}):\n{doc['content']}\n"
            )
        
        return "\n".join(context_parts)
    
    def get_vector_store_info(self) -> Dict:
        """Get information about the vector store"""
        return self.vector_store.get_collection_info()
    
    def reset_vector_store(self) -> bool:
        """Reset the vector store (delete all documents)"""
        return self.vector_store.reset_collection()
    
    def test_services(self) -> Dict:
        """
        Test all services to ensure they're working
        
        Returns:
            Test results for each service
        """
        results = {}
        
        # Test embedding service
        try:
            test_embedding = self.embedding_service.get_embedding("test")
            results["embedding_service"] = {
                "status": "success",
                "embedding_dimension": len(test_embedding),
                "model_info": self.embedding_service.get_model_info()
            }
        except Exception as e:
            results["embedding_service"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test LLM service
        try:
            test_response = self.llm_service.generate_response("Hello, this is a test.")
            results["llm_service"] = {
                "status": "success",
                "response_length": len(test_response)
            }
        except Exception as e:
            results["llm_service"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test vector store
        try:
            store_info = self.vector_store.get_collection_info()
            results["vector_store"] = {
                "status": "success",
                "document_count": store_info.get("document_count", 0)
            }
        except Exception as e:
            results["vector_store"] = {
                "status": "error",
                "error": str(e)
            }
        
        return results