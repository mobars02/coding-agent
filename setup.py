#!/usr/bin/env python3
import subprocess
import sys

def install_deps():
    deps = [
        "sentence-transformers", "faiss-cpu", "libcst", 
        "ruff", "bandit", "pytest", "GitPython", "requests", "python-dotenv"
    ]
    for dep in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", dep])

def setup_ollama():
    print("Setting up Ollama...")
    try:
        # Check if ollama exists
        subprocess.run(["ollama", "version"], check=True, capture_output=True)
        
        # Pull models
        subprocess.run(["ollama", "pull", "codellama:7b"])
        subprocess.run(["ollama", "pull", "nomic-embed-text"]) 
        print("✅ Ollama setup complete")
    except:
        print("❌ Ollama not found. Install from https://ollama.ai")

if __name__ == "__main__":
    install_deps()
    setup_ollama()
    
    # Create .env
    with open(".env", "w") as f:
        f.write("""LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=codellama:7b
VECTOR_STORE_TYPE=faiss
""")
    print("🎉 Setup complete!")