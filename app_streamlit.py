import streamlit as st
import requests
import json
from typing import List, Dict
import time
import os
from datetime import datetime
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
        color: #000000;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
        color: #000000;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
        color: #000000;
    }
    .source-card {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #4caf50;
        color: #000000;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    .chat-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        background-color: #f0f2f6;
        cursor: pointer;
    }
    .chat-item:hover {
        background-color: #e1e5e9;
    }
    .current-chat {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "current_document_id" not in st.session_state:
        st.session_state.current_document_id = None
    if "current_document_name" not in st.session_state:
        st.session_state.current_document_name = None
    if "vector_store_info" not in st.session_state:
        st.session_state.vector_store_info = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "available_documents" not in st.session_state:
        st.session_state.available_documents = []

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
            result = response.json()
            # Store document info in session state
            if result.get("success") and result.get("document_id"):
                st.session_state.current_document_id = result["document_id"]
                st.session_state.current_document_name = result.get("filename", file.name)
            return result
        else:
            return {"success": False, "message": f"Error: {response.text}"}
    except Exception as e:
        return {"success": False, "message": f"Connection error: {str(e)}"}

def ask_question_to_backend(question: str, chat_history: List[Dict], document_id: str):
    """Send question to backend and get response"""
    try:
        payload = {
            "question": question,
            "document_id": document_id,
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

def get_available_documents():
    """Get list of available documents from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def select_document_for_chat(document_id: str, document_name: str):
    """Select a document for the current chat"""
    st.session_state.current_document_id = document_id
    st.session_state.current_document_name = document_name
    # Clear current chat when switching documents
    st.session_state.current_chat = []
    st.session_state.current_chat_id = None
    st.rerun()

def create_new_chat():
    """Create a new chat session"""
    if st.session_state.current_chat and len(st.session_state.current_chat) > 0:
        # Save current chat if it has messages
        save_current_chat()
    
    # Create new chat (but keep the same document)
    st.session_state.current_chat = []
    st.session_state.current_chat_id = None
    st.rerun()

def save_current_chat():
    """Save the current chat to history"""
    if st.session_state.current_chat and len(st.session_state.current_chat) > 0:
        chat_data = {
            "id": st.session_state.current_chat_id or f"chat_{int(time.time())}",
            "title": st.session_state.current_chat[0]['content'][:50] + "..." if st.session_state.current_chat else "New Chat",
            "messages": st.session_state.current_chat.copy(),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message_count": len(st.session_state.current_chat),
            "document_id": st.session_state.current_document_id,
            "document_name": st.session_state.current_document_name
        }
        
        # Update existing chat or add new one
        if st.session_state.current_chat_id:
            # Update existing chat
            for i, chat in enumerate(st.session_state.chat_history):
                if chat["id"] == st.session_state.current_chat_id:
                    st.session_state.chat_history[i] = chat_data
                    break
        else:
            # Add new chat
            st.session_state.current_chat_id = chat_data["id"]
            st.session_state.chat_history.append(chat_data)
        
        st.success("Chat saved!")

def load_chat(chat_id: str):
    """Load a specific chat from history"""
    for chat in st.session_state.chat_history:
        if chat["id"] == chat_id:
            st.session_state.current_chat = chat["messages"].copy()
            st.session_state.current_chat_id = chat_id
            # Load document info if available
            if "document_id" in chat:
                st.session_state.current_document_id = chat["document_id"]
            if "document_name" in chat:
                st.session_state.current_document_name = chat["document_name"]
            st.rerun()
            break

def delete_chat(chat_id: str):
    """Delete a chat from history"""
    st.session_state.chat_history = [chat for chat in st.session_state.chat_history if chat["id"] != chat_id]
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat = []
        st.session_state.current_chat_id = None
    st.rerun()

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
        st.markdown('<div class="sidebar-header">📄 Document Selection</div>', unsafe_allow_html=True)
        
        # Refresh documents button
        if st.button("🔄 Refresh Documents", use_container_width=True):
            st.session_state.available_documents = get_available_documents()
            st.session_state.vector_store_info = get_vector_store_info()
        
        # Current document info
        if st.session_state.current_document_id:
            st.success(f"📄 **Current Document:** {st.session_state.current_document_name}")
            st.text(f"ID: {st.session_state.current_document_id[:8]}...")
        else:
            st.warning("⚠️ No document selected")
        
        # Available documents
        if st.session_state.available_documents:
            st.markdown("**Available Documents:**")
            for doc in st.session_state.available_documents:
                is_current = st.session_state.current_document_id == doc["document_id"]
                button_style = "🔵" if is_current else "⚪"
                
                if st.button(
                    f"{button_style} {doc['document_name']}",
                    key=f"select_doc_{doc['document_id']}",
                    help=f"Chunks: {doc['chunk_count']}",
                    use_container_width=True
                ):
                    select_document_for_chat(doc["document_id"], doc["document_name"])
        else:
            st.markdown("*No documents available*")
        
        st.markdown("---")
        
        # Chat Management
        st.markdown('<div class="sidebar-header">💬 Chat Management</div>', unsafe_allow_html=True)
        
        # New Chat Button
        if st.button("➕ New Chat", type="primary", use_container_width=True):
            create_new_chat()
        
        # Current Chat Info
        if st.session_state.current_chat:
            st.markdown(f"**Current Chat:** {len(st.session_state.current_chat)} messages")
            if st.button("💾 Save Current Chat", use_container_width=True):
                save_current_chat()
        else:
            st.markdown("**Current Chat:** Empty")
        
        st.markdown("---")
        
        # Chat History
        st.markdown('<div class="sidebar-header">📁 Chat History</div>', unsafe_allow_html=True)
        
        if st.session_state.chat_history:
            for chat in reversed(st.session_state.chat_history):  # Show newest first
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    is_current = st.session_state.current_chat_id == chat["id"]
                    button_style = "🔵" if is_current else "⚪"
                    
                    # Show document name in chat title if available
                    title = chat['title']
                    if "document_name" in chat:
                        title = f"{title} ({chat['document_name']})"
                    
                    if st.button(
                        f"{button_style} {title}",
                        key=f"load_{chat['id']}",
                        help=f"Created: {chat['created_at']} | Messages: {chat['message_count']}",
                        use_container_width=True
                    ):
                        load_chat(chat["id"])
                
                with col2:
                    if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete chat"):
                        delete_chat(chat["id"])
        else:
            st.markdown("*No previous chats*")
        
        st.markdown("---")
        
        # Vector Store Info
        st.markdown('<div class="sidebar-header">📊 Vector Store</div>', unsafe_allow_html=True)
        
        if st.session_state.vector_store_info:
            info = st.session_state.vector_store_info
            st.metric("Total Documents", info.get('total_documents', 0))
            st.metric("Total Chunks", info.get('total_chunks', 0))
        
        # Reset vector store
        if st.button("🗑️ Reset All Documents"):
            try:
                response = requests.delete(f"{BACKEND_URL}/reset-vector-store", timeout=10)
                if response.status_code == 200:
                    st.success("All documents reset successfully!")
                    st.session_state.vector_store_info = get_vector_store_info()
                    st.session_state.available_documents = get_available_documents()
                    # Clear current document and chat
                    st.session_state.current_document_id = None
                    st.session_state.current_document_name = None
                    st.session_state.current_chat = []
                    st.session_state.current_chat_id = None
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
                        st.session_state.available_documents = get_available_documents()
                        st.session_state.uploaded_files.append(result.get('filename', uploaded_file.name))
                    else:
                        st.error(f"❌ {result['message']}")
        
        # Chat Section
        if st.session_state.current_document_id:
            st.markdown(f"### 💬 Chat with: {st.session_state.current_document_name}")
            
            # Display current chat
            for message in st.session_state.current_chat:
                display_chat_message(message, is_user=(message['role'] == 'user'))
            
            # Chat input
            user_input = st.text_input(
                f"Ask a question about {st.session_state.current_document_name}:",
                placeholder="Type your question here...",
                key="user_input"
            )
            
            if st.button("Send", type="primary") and user_input:
                # Add user message to chat
                user_message = {"role": "user", "content": user_input}
                st.session_state.current_chat.append(user_message)
                
                # Get response from backend
                with st.spinner("Thinking..."):
                    response = ask_question_to_backend(
                        user_input, 
                        st.session_state.current_chat[:-1], 
                        st.session_state.current_document_id
                    )
                    
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
        else:
            st.markdown("### 💬 Chat")
            st.info("👆 Please select a document from the sidebar to start chatting, or upload a new PDF document.")
    
    with col2:
        # Information panel
        st.markdown("### ℹ️ Information")
        
        # Backend status
        if test_backend_connection():
            st.success("🟢 Backend Connected")
        else:
            st.error("🔴 Backend Disconnected")
        
        # Current document info
        if st.session_state.current_document_id:
            st.metric("Current Document", st.session_state.current_document_name)
            st.metric("Messages in Current Chat", len(st.session_state.current_chat))
        else:
            st.metric("Messages in Current Chat", len(st.session_state.current_chat))
        
        st.metric("Total Chat Sessions", len(st.session_state.chat_history))
        
        # Document-specific info
        if st.session_state.current_document_id and st.session_state.available_documents:
            current_doc = next((doc for doc in st.session_state.available_documents 
                              if doc["document_id"] == st.session_state.current_document_id), None)
            if current_doc:
                st.metric("Document Chunks", current_doc["chunk_count"])
        
        # Instructions
        st.markdown("""
        ### 📋 How to use:
        1. Upload a PDF document
        2. Select the document from the sidebar
        3. Ask questions about the document
        4. Each chat is isolated to one document
        5. Create new chats for different documents
        6. View sources for each answer
        7. Save and manage chat history
        """)
        
        # API endpoints info
        with st.expander("🔧 API Endpoints"):
            st.code(f"""
            Backend URL: {BACKEND_URL}
            
            Available endpoints:
            - POST /upload-pdf
            - POST /ask (requires document_id)
            - GET /documents
            - GET /documents/{{document_id}}
            - DELETE /documents/{{document_id}}
            - GET /vector-store-info
            - DELETE /reset-vector-store
            - GET /health
            """)

if __name__ == "__main__":
    main()