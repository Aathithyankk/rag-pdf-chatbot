# RAG PDF Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot that allows you to upload PDF documents and have isolated conversations about each document. Each chat session is tied to a specific document, ensuring clean separation of context and preventing cross-document contamination.

##  Features

- **Document Isolation**: Each chat window handles only one document
- **Multiple Document Support**: Upload and manage multiple PDF documents simultaneously
- **Persistent Chat History**: Save and load previous conversations
- **Source Attribution**: View exact sources and page numbers for each answer
- **Real-time Processing**: Fast PDF processing with chunking and embedding generation
- **Modern UI**: Clean, responsive interface built with Streamlit

##  Setup Steps

### Prerequisites

- Python 3.8 or higher
- SQLite 3.35.0 or higher (for ChromaDB compatibility)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-pdf-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   # Required API Keys
   HUGGINGFACE_API_KEY=your_huggingface_token_here
   GOOGLE_AI_API_KEY=your_google_ai_api_key_here
   
   # Optional Configuration
   BACKEND_URL=http://localhost:8000
   EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
   LLM_MODEL=gemini-2.0-flash
   VECTOR_STORE_PATH=./storage/chroma_db
   USE_LOCAL_EMBEDDINGS=true
   ```

5. **Get API Keys**
   - **Google AI**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Running the Application

1. **Start the backend server**
   ```bash
   python start_backend.py
   ```
   The API will be available at `http://localhost:8000`

2. **Start the frontend (in a new terminal)**
   ```bash
   python start_frontend.py
   ```
   The web interface will be available at `http://localhost:8501`

##  Architecture & Models

### Models Used

- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
  - Local model for fast, offline embedding generation
  - 768-dimensional embeddings
  - Optimized for semantic similarity search

- **LLM Model**: `gemini-2.0-flash`
  - Google's Gemini model via API
  - Fast response times with high-quality text generation
  - Context-aware responses with conversation history

### Chunking Strategy

- **Text Extraction**: PyPDF2 for reliable PDF text extraction
- **Chunking Method**: Fixed-size chunks with overlap
  - Chunk size: 1000 characters
  - Overlap: 200 characters
  - Preserves context across chunk boundaries
- **Metadata Preservation**: Page numbers, source filenames, and chunk IDs

### Vector Database

- **ChromaDB**: Persistent vector database
- **Document Isolation**: Each document gets its own collection (`doc_{document_id}`)
- **Similarity Search**: Cosine similarity for semantic matching
- **Persistence**: Data survives application restarts

##  Conversation History Management

### Chat Isolation
- Each chat session is tied to a specific document ID
- Switching documents automatically clears the current chat
- Chat history preserves document associations

### Session State Management
- **Frontend (Streamlit)**: Maintains chat history in session state
- **Backend**: Stateless API design for scalability
- **Persistence**: Chat history stored in browser session (not persistent across browser restarts)

### Chat Data Structure
```json
{
  "id": "chat_timestamp",
  "title": "First question preview...",
  "messages": [...],
  "created_at": "2024-01-01 12:00:00",
  "message_count": 5,
  "document_id": "uuid-string",
  "document_name": "filename.pdf"
}
```

## API Endpoints

- `POST /upload-pdf` - Upload and process PDF documents
- `POST /ask` - Ask questions (requires document_id)
- `GET /documents` - List all available documents
- `GET /documents/{document_id}` - Get specific document info
- `DELETE /documents/{document_id}` - Delete a document
- `GET /vector-store-info` - Get vector store statistics
- `DELETE /reset-vector-store` - Reset all documents
- `GET /health` - Health check

##  How It Works

1. **Document Upload**: PDF is processed, chunked, and embedded
2. **Document Selection**: User selects a document from the sidebar
3. **Question Processing**: User question is embedded and searched within the selected document
4. **Context Retrieval**: Most relevant chunks are retrieved from the document's collection
5. **Response Generation**: LLM generates answer using retrieved context
6. **Source Attribution**: Sources with page numbers are displayed

##  Known Limitations

### Technical Limitations
- **Memory Usage**: Large PDFs may consume significant memory during processing
- **API Rate Limits**: Google AI API has rate limits for free tier
- **Browser Storage**: Chat history is not persistent across browser sessions
- **Python environment**: Two different environments has to be created for frontend and backend due to dependency conflicts between streamlit, photobuf and google-generativeai libraries

### Functional Limitations
- **Single Document per Chat**: Cannot ask questions across multiple documents simultaneously
- **PDF Only**: Currently supports only PDF format doesn't support OCR currently
- **Text Extraction**: May not handle complex PDF layouts perfectly
- **No User Authentication**: No user management or access control
- **Hybrid Chunking**: Hybrid chunking with both paragraph and fixed length chunking whenever the paragraph length is too large could have been used.

### Performance Considerations
- **First Query Delay**: Initial embedding generation may take time since it runs in CPU
- **Large Documents**: Very large PDFs (>100 pages) may take longer to process
- **Concurrent Users**: Backend is not optimized for high concurrency

##  Troubleshooting

### Common Issues
1. **API Key Issues**
   - Verify API keys in `.env` file
   - Check API key permissions and quotas

2. **Memory Issues**
   - Reduce chunk size in `config.py`
   - Process smaller documents

3. **Backend Connection**
   - Ensure backend is running on correct port

