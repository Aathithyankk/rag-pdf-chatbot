import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Optional
import uuid
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Chroma vector database for storing and retrieving document embeddings"""
    
    def __init__(self, persist_directory: str = "./storage/chroma_db"):
        """
        Initialize Chroma vector store
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Store collections by document ID
        self.collections = {}
        self.collection_metadata = {}
    
    def _get_or_create_collection(self, document_id: str, document_name: str = None) -> chromadb.Collection:
        """
        Get or create a collection for a specific document
        
        Args:
            document_id: Unique identifier for the document
            document_name: Human-readable name for the document
            
        Returns:
            Chroma collection object
        """
        if document_id not in self.collections:
            collection_name = f"doc_{document_id}"
            try:
                # Try to get existing collection
                collection = self.client.get_collection(name=collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except:
                # Create new collection
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "description": f"PDF document chunks for RAG - {document_name or document_id}",
                        "document_id": document_id,
                        "document_name": document_name or document_id
                    }
                )
                logger.info(f"Created new collection: {collection_name}")
            
            self.collections[document_id] = collection
            self.collection_metadata[document_id] = {
                "document_name": document_name or document_id,
                "created_at": str(uuid.uuid4())  # Simple timestamp placeholder
            }
        
        return self.collections[document_id]
    
    def add_documents(self, chunks: List[Dict[str, str]], embeddings: List[List[float]], 
                     document_id: str, document_name: str = None) -> bool:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of document chunks with content and metadata
            embeddings: List of embeddings for each chunk
            document_id: Unique identifier for the document
            document_name: Human-readable name for the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get or create collection for this document
            collection = self._get_or_create_collection(document_id, document_name)
            
            # Prepare data for Chroma
            ids = [chunk['chunk_id'] for chunk in chunks]
            documents = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # Add to collection
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], document_id: str, n_results: int = 5, 
                      filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            document_id: Document ID to search within
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Get the collection for this document
            if document_id not in self.collections:
                logger.warning(f"No collection found for document {document_id}")
                return []
            
            collection = self.collections[document_id]
            
            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    similar_docs.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0,
                        'id': results['ids'][0][i] if results['ids'] else None
                    })
            
            logger.info(f"Found {len(similar_docs)} similar documents in document {document_id}")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_collection_info(self, document_id: str = None) -> Dict:
        """Get information about collections"""
        try:
            if document_id:
                # Get info for specific document
                if document_id not in self.collections:
                    return {
                        'document_id': document_id,
                        'document_name': 'Not found',
                        'chunk_count': 0,
                        'exists': False
                    }
                
                collection = self.collections[document_id]
                chunk_count = collection.count()
                metadata = self.collection_metadata.get(document_id, {})
                
                return {
                    'document_id': document_id,
                    'document_name': metadata.get('document_name', document_id),
                    'chunk_count': chunk_count,
                    'exists': True
                }
            else:
                # Get info for all collections
                total_chunks = 0
                document_count = 0
                documents_info = []
                
                # Get all existing collections from Chroma
                all_collections = self.client.list_collections()
                
                for collection_info in all_collections:
                    if collection_info.name.startswith('doc_'):
                        document_id = collection_info.name[4:]  # Remove 'doc_' prefix
                        collection = self.client.get_collection(name=collection_info.name)
                        chunk_count = collection.count()
                        
                        if chunk_count > 0:
                            total_chunks += chunk_count
                            document_count += 1
                            
                            metadata = self.collection_metadata.get(document_id, {})
                            documents_info.append({
                                'document_id': document_id,
                                'document_name': metadata.get('document_name', document_id),
                                'chunk_count': chunk_count
                            })
                
                return {
                    'total_documents': document_count,
                    'total_chunks': total_chunks,
                    'documents': documents_info,
                    'persist_directory': self.persist_directory
                }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'documents': [],
                'persist_directory': self.persist_directory
            }
    
    def delete_document_collection(self, document_id: str) -> bool:
        """Delete a specific document collection"""
        try:
            collection_name = f"doc_{document_id}"
            self.client.delete_collection(name=collection_name)
            
            # Remove from local cache
            if document_id in self.collections:
                del self.collections[document_id]
            if document_id in self.collection_metadata:
                del self.collection_metadata[document_id]
            
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def reset_all_collections(self) -> bool:
        """Reset all collections (delete all document collections)"""
        try:
            # Get all existing collections from Chroma
            all_collections = self.client.list_collections()
            
            for collection_info in all_collections:
                if collection_info.name.startswith('doc_'):
                    document_id = collection_info.name[4:]  # Remove 'doc_' prefix
                    self.delete_document_collection(document_id)
            
            # Clear local cache
            self.collections = {}
            self.collection_metadata = {}
            
            logger.info("Reset all document collections")
            return True
        except Exception as e:
            logger.error(f"Error resetting collections: {e}")
            return False
    
    def list_documents(self) -> List[Dict]:
        """List all available documents"""
        try:
            documents = []
            all_collections = self.client.list_collections()
            
            for collection_info in all_collections:
                if collection_info.name.startswith('doc_'):
                    document_id = collection_info.name[4:]  # Remove 'doc_' prefix
                    collection = self.client.get_collection(name=collection_info.name)
                    chunk_count = collection.count()
                    
                    if chunk_count > 0:
                        metadata = self.collection_metadata.get(document_id, {})
                        documents.append({
                            'document_id': document_id,
                            'document_name': metadata.get('document_name', document_id),
                            'chunk_count': chunk_count
                        })
            
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []