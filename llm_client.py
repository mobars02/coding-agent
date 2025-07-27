# llm_client/base.py
"""Provider-agnostic LLM interface for free/local backends"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str

@dataclass
class LLMResponse:
    content: str
    usage: Dict[str, int]  # tokens used
    model: str
    finish_reason: str

@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]
    usage: Dict[str, int]
    model: str

class LLMClient(ABC):
    """Base interface for all LLM providers"""
    
    @abstractmethod
    def generate(self, 
                messages: List[ChatMessage],
                max_tokens: int = 2048,
                temperature: float = 0.1,
                stop: Optional[List[str]] = None) -> LLMResponse:
        """Generate text completion"""
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> EmbeddingResponse:
        """Generate embeddings"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate)"""
        pass

# llm_client/ollama_client.py
"""Ollama local LLM client"""

import requests
import json
from typing import List, Optional
from .base import LLMClient, ChatMessage, LLMResponse, EmbeddingResponse

class OllamaClient(LLMClient):
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 chat_model: str = "codellama:7b",
                 embedding_model: str = "nomic-embed-text"):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        
        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Connected to Ollama at {base_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def generate(self, 
                messages: List[ChatMessage],
                max_tokens: int = 2048,
                temperature: float = 0.1,
                stop: Optional[List[str]] = None) -> LLMResponse:
        
        # Convert messages to Ollama format
        if len(messages) == 1:
            prompt = messages[0].content
        else:
            # Multi-turn conversation
            prompt = ""
            for msg in messages:
                if msg.role == "system":
                    prompt += f"System: {msg.content}\n"
                elif msg.role == "user":
                    prompt += f"User: {msg.content}\n"
                elif msg.role == "assistant":
                    prompt += f"Assistant: {msg.content}\n"
            prompt += "Assistant: "
        
        payload = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": stop or []
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            
            return LLMResponse(
                content=result["response"].strip(),
                usage={"total_tokens": len(prompt.split()) + len(result["response"].split())},
                model=self.chat_model,
                finish_reason="stop"
            )
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    def embed(self, texts: List[str]) -> EmbeddingResponse:
        embeddings = []
        total_tokens = 0
        
        for text in texts:
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                embeddings.append(result["embedding"])
                total_tokens += len(text.split())
                
            except Exception as e:
                logger.error(f"Ollama embedding failed for text: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 768)  # Common embedding size
        
        return EmbeddingResponse(
            embeddings=embeddings,
            usage={"total_tokens": total_tokens},
            model=self.embedding_model
        )
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars per token heuristic)"""
        return len(text) // 4

# llm_client/local_sentence_transformers.py
"""Local sentence transformers for embeddings"""

from typing import List
from .base import LLMClient, ChatMessage, LLMResponse, EmbeddingResponse

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class LocalEmbeddingClient:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Loaded local embedding model: {model_name}")
    
    def embed(self, texts: List[str]) -> EmbeddingResponse:
        """Generate embeddings locally"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            return EmbeddingResponse(
                embeddings=embeddings.tolist(),
                usage={"total_tokens": sum(len(text.split()) for text in texts)},
                model=self.model_name
            )
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise

# llm_client/factory.py
"""LLM client factory for easy switching"""

import os
from typing import Optional
from .base import LLMClient
from .ollama_client import OllamaClient

def create_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Create LLM client based on environment or explicit provider"""
    
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "ollama")
    
    provider = provider.lower()
    
    if provider == "ollama":
        return OllamaClient(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            chat_model=os.getenv("OLLAMA_CHAT_MODEL", "codellama:7b"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        )
    
    elif provider == "hf_local":
        # Could add HuggingFace transformers pipeline client
        raise NotImplementedError("HF local client not implemented yet")
    
    elif provider == "vllm":
        # Could add vLLM client
        raise NotImplementedError("vLLM client not implemented yet")
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

# Usage example
if __name__ == "__main__":
    # Test the client
    client = create_llm_client("ollama")
    
    messages = [ChatMessage(role="user", content="Write a simple Python function to add two numbers")]
    response = client.generate(messages)
    print(f"Response: {response.content}")
    
    # Test embeddings
    embeddings = client.embed(["Hello world", "Code generation"])
    print(f"Generated {len(embeddings.embeddings)} embeddings")