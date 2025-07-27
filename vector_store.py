# vector_store/base.py
"""Local vector storage replacing Supabase"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    document: Document
    score: float

class VectorStore(ABC):
    """Base interface for vector storage"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete(self, document_ids: List[str]) -> None:
        """Delete documents"""
        pass

# vector_store/faiss_store.py
"""FAISS-based local vector store"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .base import VectorStore, Document, SearchResult

class FAISSVectorStore(VectorStore):
    def __init__(self, storage_path: str = "./artifacts/vector_store"):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.storage_path / "faiss.index"
        self.documents_file = self.storage_path / "documents.pkl"
        
        # Initialize or load index
        self.documents: Dict[str, Document] = {}
        self.index: Optional[faiss.Index] = None
        self.dimension = None
        
        self._load_store()
        
        logger.info(f"FAISS vector store initialized at {storage_path}")
    
    def _load_store(self):
        """Load existing index and documents"""
        try:
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                self.dimension = self.index.d
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            if self.documents_file.exists():
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} documents")
                
        except Exception as e:
            logger.warning(f"Failed to load existing store: {e}")
            self.documents = {}
            self.index = None
    
    def _save_store(self):
        """Save index and documents to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
            
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
                
        except Exception as e:
            logger.error(f"Failed to save store: {e}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents with embeddings to FAISS index"""
        if not documents:
            return
        
        # Extract embeddings
        embeddings = []
        valid_docs = []
        
        for doc in documents:
            if doc.embedding:
                embeddings.append(doc.embedding)
                valid_docs.append(doc)
                self.documents[doc.id] = doc
        
        if not embeddings:
            logger.warning("No documents with embeddings to add")
            return
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Initialize index if needed
        if self.index is None:
            self.dimension = embeddings_array.shape[1]
            # Use flat L2 index for simplicity (can be upgraded to HNSW for large datasets)
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Created new FAISS index with dimension {self.dimension}")
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Save to disk
        self._save_store()
        
        logger.info(f"Added {len(valid_docs)} documents to vector store")
    
    def search(self, query_embedding: List[float], k: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Search for similar documents"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("No documents in vector store")
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search FAISS index
        scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        results = []
        doc_list = list(self.documents.values())
        
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(doc_list):  # Valid index
                # Convert L2 distance to similarity score (higher = more similar)
                similarity = 1.0 / (1.0 + score)
                
                if similarity >= threshold:
                    results.append(SearchResult(
                        document=doc_list[idx],
                        score=similarity
                    ))
        
        return results
    
    def delete(self, document_ids: List[str]) -> None:
        """Delete documents (requires rebuilding index)"""
        # Remove from documents dict
        for doc_id in document_ids:
            self.documents.pop(doc_id, None)
        
        # Rebuild index with remaining documents
        remaining_docs = list(self.documents.values())
        self.index = None  # Reset index
        
        if remaining_docs:
            self.add_documents(remaining_docs)
        else:
            self._save_store()
        
        logger.info(f"Deleted {len(document_ids)} documents")

# vector_store/hybrid_search.py
"""Hybrid search combining BM25 and vector similarity"""

from typing import List, Dict, Any, Optional
import re
import math
from collections import defaultdict, Counter

class BM25:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        # Tokenize and process documents
        self.documents = documents
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Calculate document frequencies
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0
        
        self._calculate_frequencies()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _calculate_frequencies(self):
        """Calculate term frequencies and IDF"""
        # Document frequencies
        df = defaultdict(int)
        
        for tokens in self.tokenized_docs:
            self.doc_freqs.append(Counter(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1
        
        # Calculate IDF
        num_docs = len(self.documents)
        for term, freq in df.items():
            self.idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5))
        
        # Average document length
        self.avgdl = sum(len(tokens) for tokens in self.tokenized_docs) / num_docs
    
    def search(self, query: str, k: int = 10) -> List[tuple[int, float]]:
        """Search and return (doc_index, score) pairs"""
        query_tokens = self._tokenize(query)
        
        scores = []
        for i, doc_tokens in enumerate(self.tokenized_docs):
            score = 0
            doc_len = len(doc_tokens)
            
            for token in query_tokens:
                if token in self.doc_freqs[i]:
                    tf = self.doc_freqs[i][token]
                    idf = self.idf.get(token, 0)
                    
                    score += idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    )
            
            scores.append((i, score))
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

class HybridSearchStore:
    def __init__(self, vector_store: VectorStore, embedding_client):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.bm25: Optional[BM25] = None
        self._rebuild_bm25()
    
    def _rebuild_bm25(self):
        """Rebuild BM25 index from current documents"""
        documents_text = [doc.content for doc in self.vector_store.documents.values()]
        if documents_text:
            self.bm25 = BM25(documents_text)
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.7) -> List[SearchResult]:
        """
        Hybrid search combining BM25 and vector similarity
        alpha: weight for vector search (1-alpha for BM25)
        """
        if not self.vector_store.documents:
            return []
        
        # Get query embedding
        query_embedding_response = self.embedding_client.embed([query])
        query_embedding = query_embedding_response.embeddings[0]
        
        # Vector search
        vector_results = self.vector_store.search(query_embedding, k=k*2)  # Get more candidates
        
        # BM25 search
        bm25_results = []
        if self.bm25:
            bm25_scores = self.bm25.search(query, k=k*2)
            doc_list = list(self.vector_store.documents.values())
            
            for idx, score in bm25_scores:
                if idx < len(doc_list):
                    bm25_results.append(SearchResult(
                        document=doc_list[idx],
                        score=score
                    ))
        
        # Combine scores
        combined_scores = {}
        
        # Add vector scores
        for result in vector_results:
            doc_id = result.document.id
            combined_scores[doc_id] = alpha * result.score
        
        # Add BM25 scores
        max_bm25_score = max([r.score for r in bm25_results], default=1.0)
        for result in bm25_results:
            doc_id = result.document.id
            normalized_bm25 = result.score / max_bm25_score
            
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * normalized_bm25
            else:
                combined_scores[doc_id] = (1 - alpha) * normalized_bm25
        
        # Sort by combined score
        sorted_results = []
        for doc_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
            if doc_id in self.vector_store.documents:
                sorted_results.append(SearchResult(
                    document=self.vector_store.documents[doc_id],
                    score=score
                ))
        
        return sorted_results[:k]

# vector_store/factory.py
"""Vector store factory"""

import os
from .base import VectorStore
from .faiss_store import FAISSVectorStore

def create_vector_store(store_type: Optional[str] = None, **kwargs) -> VectorStore:
    """Create vector store based on configuration"""
    
    if store_type is None:
        store_type = os.getenv("VECTOR_STORE_TYPE", "faiss")
    
    if store_type.lower() == "faiss":
        storage_path = kwargs.get("storage_path", "./artifacts/vector_store")
        return FAISSVectorStore(storage_path)
    
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")

# Example usage
if __name__ == "__main__":
    from llm_client.factory import create_llm_client
    from llm_client.local_sentence_transformers import LocalEmbeddingClient
    
    # Create local vector store
    store = create_vector_store("faiss")
    
    # Create embedding client
    embedding_client = LocalEmbeddingClient()
    
    # Add some test documents
    documents = [
        Document(
            id="doc1",
            content="Python function to calculate fibonacci numbers",
            metadata={"type": "code", "language": "python"}
        ),
        Document(
            id="doc2", 
            content="JavaScript async/await pattern for API calls",
            metadata={"type": "code", "language": "javascript"}
        )
    ]
    
    # Generate embeddings
    texts = [doc.content for doc in documents]
    embeddings_response = embedding_client.embed(texts)
    
    # Add embeddings to documents
    for doc, embedding in zip(documents, embeddings_response.embeddings):
        doc.embedding = embedding
    
    # Store documents
    store.add_documents(documents)
    
    # Search
    query_embedding = embedding_client.embed(["fibonacci function"]).embeddings[0]
    results = store.search(query_embedding, k=5)
    
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"- {result.document.content} (score: {result.score:.3f})")