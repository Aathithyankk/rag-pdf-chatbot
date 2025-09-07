# RAG PDF Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot that allows you to upload PDF documents and ask questions about their content. Built with Streamlit frontend and FastAPI backend.

## Features

- 📄 **PDF Upload & Processing**: Upload PDF documents and extract text content
- 🤖 **AI-Powered Chat**: Ask questions and get intelligent answers using Gemini 2.5 Flash
- 🔍 **Vector Search**: Use Chroma vector database for semantic search
- 💬 **Chat History**: Save and manage multiple chat sessions
- 📚 **Source Attribution**: View sources for each answer with page references
- 🎨 **Modern UI**: Clean and intuitive Streamlit interface

## Architecture

- **Frontend**: Streamlit application (`app_streamlit.py`)
- **Backend**: FastAPI server (`app_backend.py`)
- **Vector Store**: Chroma database for document embeddings
- **Embeddings**: Hugging Face API for text embeddings
- **LLM**: Google Gemini 2.5 Flash for response generation

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

You'll need API keys for:

1. **Hugging Face**: Get your API key from [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. **Google AI**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
# API Keys (Required)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
GOOGLE_AI_API_KEY=your_google_ai_api_key_here

# Backend Configuration (Optional)
BACKEND_URL=http://localhost:8000

# Model Configurations (Optional)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gemini-2.0-flash-exp

# Vector Store Configuration (Optional)
VECTOR_STORE_PATH=./storage/chroma_db

# PDF Processing Configuration (Optional)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG Configuration (Optional)
MAX_RETRIEVAL_RESULTS=5
```

### 4. Run the Application

#### Start the Backend Server

```bash
python app_backend.py
```

The backend will be available at `http://localhost:8000`

#### Start the Frontend Application

In a new terminal:

```bash
streamlit run app_streamlit.py
```

The frontend will be available at `http://localhost:8501`

## Usage

1. **Upload PDF**: Use the file uploader to upload a PDF document
2. **Process Document**: Click "Process PDF" to extract and index the content
3. **Ask Questions**: Type your questions in the chat interface
4. **View Sources**: Expand the "Sources" section to see where answers came from
5. **Manage Chats**: Use the sidebar to save and switch between chat sessions

## API Endpoints

The FastAPI backend provides the following endpoints:

- `POST /upload-pdf`: Upload and process a PDF file
- `POST /ask`: Ask a question and get an answer
- `GET /vector-store-info`: Get information about the vector store
- `DELETE /reset-vector-store`: Reset the vector store
- `GET /health`: Health check endpoint

## Project Structure

```
rag-pdf-chatbot/
├── modules/
│   ├── pdf_loader.py          # PDF text extraction and chunking
│   ├── vector_store.py        # Chroma vector database operations
│   ├── embedding_service.py   # Hugging Face embedding API
│   ├── llm_service.py         # Gemini LLM integration
│   └── rag_service.py         # Main RAG orchestration
├── storage/                   # Vector database storage
├── app_backend.py            # FastAPI backend server
├── app_streamlit.py          # Streamlit frontend application
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Configuration Options

### PDF Processing
- `CHUNK_SIZE`: Size of text chunks for vector storage (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

### RAG Settings
- `MAX_RETRIEVAL_RESULTS`: Number of similar documents to retrieve (default: 5)
- `EMBEDDING_MODEL`: Hugging Face embedding model to use
- `LLM_MODEL`: Gemini model to use for responses

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your API keys are correctly set in the `.env` file
2. **Connection Errors**: Make sure the backend server is running before starting the frontend
3. **PDF Processing Errors**: Check that the PDF file is not corrupted and contains extractable text
4. **Memory Issues**: For large PDFs, consider reducing `CHUNK_SIZE` or `MAX_RETRIEVAL_RESULTS`
