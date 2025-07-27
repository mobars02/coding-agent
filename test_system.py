#!/usr/bin/env python3
"""
test_system.py - Test the enhanced_tools.py functionality
"""

import os
import tempfile
from pathlib import Path

def test_local_llm():
    """Test local LLM functionality"""
    print("🧠 Testing Local LLM...")
    
    try:
        # Import the function from enhanced_tools
        from enhanced_tools import _get_local_llm_response
        
        # Test with a simple prompt
        response = _get_local_llm_response("Say hello in one word", max_tokens=50)
        
        if response and len(response) > 0:
            print(f"✅ Local LLM response: {response[:50]}...")
            return True
        else:
            print("❌ Local LLM returned empty response")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

def test_vector_store():
    """Test local vector store"""
    print("🔍 Testing Local Vector Store...")
    
    try:
        from enhanced_tools import LocalVectorStore
        
        # Create a temporary vector store
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalVectorStore(temp_dir)
            
            # Add a test document
            store.add_document("Hello world test document", {"type": "test"})
            
            # Try to search
            results = store.search("hello", k=1)
            
            print(f"✅ Vector store created and searched. Found {len(results)} results")
            return True
            
    except ImportError as e:
        print(f"❌ Import error (likely missing dependencies): {e}")
        print("   Install with: pip install sentence-transformers faiss-cpu")
        return False
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_file_operations():
    """Test enhanced file operations"""
    print("📁 Testing File Operations...")
    
    try:
        # Import the actual functions, not the decorated versions
        import enhanced_tools
        
        # Access the underlying functions from the tool objects
        read_func = enhanced_tools.read_file_advanced.func if hasattr(enhanced_tools.read_file_advanced, 'func') else enhanced_tools.read_file_advanced
        write_func = enhanced_tools.write_file_smart.func if hasattr(enhanced_tools.write_file_smart, 'func') else enhanced_tools.write_file_smart
        
        # Create a test file
        test_content = """def hello():
    print("Hello world")
    return "hello"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            test_file = f.name
        
        try:
            # Test advanced file reading
            result = read_func(test_file)
            
            if result.get("content") == test_content:
                print("✅ Advanced file reading works")
                
                # Test smart file writing
                new_content = test_content + "\n# Added comment"
                write_result = write_func(test_file, new_content, backup=True)
                
                if write_result.get("success"):
                    print("✅ Smart file writing works")
                    return True
                else:
                    print(f"❌ File writing failed: {write_result}")
                    return False
            else:
                print("❌ File reading failed")
                return False
                
        finally:
            # Clean up
            os.unlink(test_file)
            backup_file = test_file + ".backup"
            if os.path.exists(backup_file):
                os.unlink(backup_file)
        
    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        return False

def test_git_operations():
    """Test git operations (if in a git repo)"""
    print("🔧 Testing Git Operations...")
    
    try:
        import enhanced_tools
        
        # Get the underlying function
        status_func = enhanced_tools.git_status_enhanced.func if hasattr(enhanced_tools.git_status_enhanced, 'func') else enhanced_tools.git_status_enhanced
        
        # Test enhanced status
        status = status_func()
        
        if "current_branch" in status:
            print(f"✅ Git operations work. Current branch: {status['current_branch']}")
            return True
        else:
            print("❌ Git status failed")
            return False
            
    except Exception as e:
        print(f"❌ Git test failed (might not be in a git repo): {e}")
        return False

def test_quality_tools():
    """Test code quality analysis"""
    print("🛡️ Testing Quality Tools...")
    
    try:
        import enhanced_tools
        
        # Get the underlying function
        quality_func = enhanced_tools.analyze_code_quality.func if hasattr(enhanced_tools.analyze_code_quality, 'func') else enhanced_tools.analyze_code_quality
        
        # Test on current directory
        result = quality_func(".")
        
        if "total_files" in result and "quality_score" in result:
            print(f"✅ Quality analysis works. Found {result['total_files']} files, score: {result['quality_score']:.1f}")
            return True
        else:
            print("❌ Quality analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Quality tools test failed: {e}")
        return False

def test_linting():
    """Test linting functionality"""
    print("🔍 Testing Linting...")
    
    try:
        import enhanced_tools
        
        # Get the underlying function
        lint_func = enhanced_tools.run_comprehensive_lint.func if hasattr(enhanced_tools.run_comprehensive_lint, 'func') else enhanced_tools.run_comprehensive_lint
        
        # Run linting
        result = lint_func(fix_issues=False)
        
        if "overall_score" in result:
            print(f"✅ Linting works. Overall score: {result['overall_score']:.1f}/100")
            return True
        else:
            print("❌ Linting failed")
            return False
            
    except Exception as e:
        print(f"❌ Linting test failed: {e}")
        return False

def check_ollama():
    """Check if Ollama is running"""
    print("🦙 Checking Ollama Status...")
    
    try:
        import requests
        
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            tags = response.json()
            models = [model.get("name", "unknown") for model in tags.get("models", [])]
            print(f"✅ Ollama is running with models: {models}")
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ollama not accessible: {e}")
        print("   Start with: ollama serve")
        print("   Install models: ollama pull codellama:7b")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("📦 Checking Dependencies...")
    
    deps = {
        "sentence_transformers": "sentence-transformers",
        "faiss": "faiss-cpu", 
        "git": "GitPython",
        "requests": "requests"
    }
    
    missing = []
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - install with: pip install {package}")
            missing.append(package)
    
    return len(missing) == 0

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Tools System")
    print("=" * 50)
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠️ Missing dependencies. Install them first:")
        print("pip install sentence-transformers faiss-cpu GitPython requests")
        return
    
    # Check Ollama
    ollama_ok = check_ollama()
    
    # Run tests
    tests = [
        ("File Operations", test_file_operations),
        ("Git Operations", test_git_operations), 
        ("Quality Tools", test_quality_tools),
        ("Linting", test_linting),
        ("Vector Store", test_vector_store),
    ]
    
    # Only test LLM if Ollama is running
    if ollama_ok:
        tests.append(("Local LLM", test_local_llm))
    
    passed = 0
    total = len(tests)
    
    print(f"\n🚀 Running {total} tests...")
    print("-" * 30)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your enhanced tools are working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        
    if not ollama_ok:
        print("\n💡 To enable AI features:")
        print("   1. Install Ollama: https://ollama.ai")
        print("   2. Start service: ollama serve")
        print("   3. Pull models: ollama pull codellama:7b")

if __name__ == "__main__":
    main()