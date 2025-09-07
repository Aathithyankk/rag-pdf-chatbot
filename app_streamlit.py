import streamlit as st
import requests
import json
from typing import List, Dict
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .source-card {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #4caf50;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []
    if "vector_store_info" not in st.session_state:
        st.session_state.vector_store_info = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

def test_backend_connection():
    """Test connection to backend API"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_pdf_to_backend(file):
    """Upload PDF to backend for processing"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{BACKEND_URL}/upload-pdf", files=files, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "message": f"Error: {response.text}"}
    except Exception as e:
        return {"success": False, "message": f"Connection error: {str(e)}"}

def ask_question_to_backend(question: str, chat_history: List[Dict]):
    """Send question to backend and get response"""
    try:
        payload = {
            "question": question,
            "chat_history": chat_history,
            "n_results": 5
        }
        response = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "answer": f"Error: {response.text}"}
    except Exception as e:
        return {"success": False, "answer": f"Connection error: {str(e)}"}

def get_vector_store_info():
    """Get vector store information from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/vector-store-info", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def display_chat_message(message: Dict, is_user: bool = False):
    """Display a chat message with proper styling"""
    css_class = "user-message" if is_user else "assistant-message"
    role = "You" if is_user else "Assistant"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{role}:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources if available
        if not is_user and 'sources' in message and message['sources']:
            with st.expander("📚 Sources"):
                for i, source in enumerate(message['sources'], 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Source {i}:</strong><br>
                        <em>From: {source.get('metadata', {}).get('source', 'Unknown')}, 
                        Page: {source.get('metadata', {}).get('page', 'Unknown')}</em><br>
                        {source.get('content', '')[:300]}...
                    </div>
                    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🤖 RAG PDF Chatbot</div>', unsafe_allow_html=True)
    
    # Check backend connection
    if not test_backend_connection():
        st.error(f"❌ Cannot connect to backend API at {BACKEND_URL}. Please ensure the backend is running.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📁 Chat History</div>', unsafe_allow_html=True)
        
        # Chat history management
        if st.button("🗑️ Clear Current Chat"):
            st.session_state.current_chat = []
            st.rerun()
        
        if st.button("🗑️ Clear All History"):
            st.session_state.chat_history = []
            st.session_state.current_chat = []
            st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("**Previous Chats:**")
            for i, chat in enumerate(st.session_state.chat_history):
                if st.button(f"💬 Chat {i+1} ({len(chat)} messages)", key=f"chat_{i}"):
                    st.session_state.current_chat = chat
                    st.rerun()
        
        # Vector store info
        st.markdown('<div class="sidebar-header">📊 Vector Store</div>', unsafe_allow_html=True)
        if st.button("🔄 Refresh Info"):
            st.session_state.vector_store_info = get_vector_store_info()
        
        if st.session_state.vector_store_info:
            info = st.session_state.vector_store_info
            st.metric("Documents", info.get('document_count', 0))
            st.text(f"Collection: {info.get('collection_name', 'N/A')}")
        
        # Reset vector store
        if st.button("🗑️ Reset Vector Store"):
            try:
                response = requests.delete(f"{BACKEND_URL}/reset-vector-store", timeout=10)
                if response.status_code == 200:
                    st.success("Vector store reset successfully!")
                    st.session_state.vector_store_info = get_vector_store_info()
                else:
                    st.error("Failed to reset vector store")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # PDF Upload Section
        st.markdown("### 📄 Upload PDF Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to start asking questions"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Process PDF"):
                with st.spinner("Processing PDF..."):
                    result = upload_pdf_to_backend(uploaded_file)
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.info(f"📊 Processed {result['pages_processed']} pages, created {result['chunks_created']} chunks")
                        st.session_state.vector_store_info = get_vector_store_info()
                        st.session_state.uploaded_files.append(result.get('filename', uploaded_file.name))
                    else:
                        st.error(f"❌ {result['message']}")
        
        # Chat Section
        st.markdown("### 💬 Chat with your PDF")
        
        # Display current chat
        for message in st.session_state.current_chat:
            display_chat_message(message, is_user=(message['role'] == 'user'))
        
        # Chat input
        user_input = st.text_input(
            "Ask a question about your PDF:",
            placeholder="Type your question here...",
            key="user_input"
        )
        
        if st.button("Send", type="primary") and user_input:
            # Add user message to chat
            user_message = {"role": "user", "content": user_input}
            st.session_state.current_chat.append(user_message)
            
            # Get response from backend
            with st.spinner("Thinking..."):
                response = ask_question_to_backend(user_input, st.session_state.current_chat[:-1])
                
                if response["success"]:
                    assistant_message = {
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    }
                    st.session_state.current_chat.append(assistant_message)
                else:
                    error_message = {
                        "role": "assistant",
                        "content": response.get("answer", "Sorry, I couldn't process your question.")
                    }
                    st.session_state.current_chat.append(error_message)
            
            st.rerun()
    
    with col2:
        # Information panel
        st.markdown("### ℹ️ Information")
        
        # Backend status
        if test_backend_connection():
            st.success("🟢 Backend Connected")
        else:
            st.error("🔴 Backend Disconnected")
        
        # Current chat info
        st.metric("Messages in Current Chat", len(st.session_state.current_chat))
        st.metric("Total Chat Sessions", len(st.session_state.chat_history))
        
        # Save current chat
        if st.session_state.current_chat and st.button("💾 Save Current Chat"):
            st.session_state.chat_history.append(st.session_state.current_chat.copy())
            st.success("Chat saved to history!")
        
        # Instructions
        st.markdown("""
        ### 📋 How to use:
        1. Upload a PDF document
        2. Wait for processing to complete
        3. Ask questions about the document
        4. View sources for each answer
        5. Save important chats to history
        """)
        
        # API endpoints info
        with st.expander("🔧 API Endpoints"):
            st.code(f"""
            Backend URL: {BACKEND_URL}
            
            Available endpoints:
            - POST /upload-pdf
            - POST /ask
            - GET /vector-store-info
            - DELETE /reset-vector-store
            - GET /health
            """)

if __name__ == "__main__":
    main()
