from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import os
from dotenv import load_dotenv

# Import our modules
from modules.rag_service import RAGService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG PDF Chatbot API",
    description="Backend API for RAG-based PDF chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    document_id: str
    chat_history: Optional[List[Dict]] = []
    n_results: Optional[int] = 5

class QuestionResponse(BaseModel):
    success: bool
    answer: str
    sources: List[Dict]
    context_used: str
    similar_docs_found: Optional[int] = 0
    document_id: str

class ProcessPDFResponse(BaseModel):
    success: bool
    message: str
    pages_processed: int
    chunks_created: int
    filename: Optional[str] = None
    document_id: str

class VectorStoreInfo(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[Dict]
    persist_directory: str

class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    chunk_count: int
    exists: bool

class TestResponse(BaseModel):
    embedding_service: Dict
    llm_service: Dict
    vector_store: Dict

# Initialize RAG service
rag_service = None

def get_rag_service():
    """Dependency to get RAG service instance"""
    global rag_service
    if rag_service is None:
        try:
            rag_service = RAGService(
                huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY"),
                google_ai_api_key=os.getenv("GOOGLE_AI_API_KEY")
            )
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG service: {str(e)}")
    return rag_service

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG PDF Chatbot API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.post("/upload-pdf", response_model=ProcessPDFResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    rag: RAGService = Depends(get_rag_service)
):
    """
    Upload and process a PDF file
    
    Args:
        file: PDF file to upload
        
    Returns:
        Processing result with document_id
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Process the PDF (document_id will be generated automatically)
        result = rag.process_pdf(file)
        
        if result["success"]:
            logger.info(f"Successfully processed PDF: {file.filename} with document_id: {result['document_id']}")
            return ProcessPDFResponse(**result)
        else:
            logger.error(f"Failed to process PDF: {result['message']}")
            raise HTTPException(status_code=400, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    rag: RAGService = Depends(get_rag_service)
):
    """
    Ask a question and get an answer using RAG
    
    Args:
        request: Question request with question text, document_id, and chat history
        
    Returns:
        Answer with sources and context
    """
    try:
        result = rag.ask_question(
            question=request.question,
            document_id=request.document_id,
            chat_history=request.chat_history,
            n_results=request.n_results
        )
        
        if result["success"]:
            logger.info(f"Successfully answered question for document {request.document_id}: {request.question[:50]}...")
            return QuestionResponse(**result)
        else:
            logger.warning(f"Failed to answer question for document {request.document_id}: {result.get('answer', 'Unknown error')}")
            return QuestionResponse(**result)
            
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.get("/vector-store-info", response_model=VectorStoreInfo)
async def get_vector_store_info(rag: RAGService = Depends(get_rag_service)):
    """
    Get information about the vector store
    
    Returns:
        Vector store information
    """
    try:
        info = rag.get_vector_store_info()
        return VectorStoreInfo(**info)
    except Exception as e:
        logger.error(f"Error getting vector store info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting vector store info: {str(e)}")

@app.delete("/reset-vector-store")
async def reset_vector_store(rag: RAGService = Depends(get_rag_service)):
    """
    Reset the vector store (delete all documents)
    
    Returns:
        Success message
    """
    try:
        success = rag.reset_vector_store()
        if success:
            return {"message": "Vector store reset successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset vector store")
    except Exception as e:
        logger.error(f"Error resetting vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting vector store: {str(e)}")

@app.get("/documents", response_model=List[Dict])
async def list_documents(rag: RAGService = Depends(get_rag_service)):
    """
    List all available documents
    
    Returns:
        List of documents with metadata
    """
    try:
        documents = rag.list_documents()
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/documents/{document_id}", response_model=DocumentInfo)
async def get_document_info(document_id: str, rag: RAGService = Depends(get_rag_service)):
    """
    Get information about a specific document
    
    Args:
        document_id: Document identifier
        
    Returns:
        Document information
    """
    try:
        info = rag.get_vector_store_info(document_id)
        return DocumentInfo(**info)
    except Exception as e:
        logger.error(f"Error getting document info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document info: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, rag: RAGService = Depends(get_rag_service)):
    """
    Delete a specific document
    
    Args:
        document_id: Document identifier
        
    Returns:
        Success message
    """
    try:
        success = rag.delete_document(document_id)
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/test-services", response_model=TestResponse)
async def test_services(rag: RAGService = Depends(get_rag_service)):
    """
    Test all services to ensure they're working
    
    Returns:
        Test results for each service
    """
    try:
        results = rag.test_services()
        return TestResponse(**results)
    except Exception as e:
        logger.error(f"Error testing services: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing services: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)