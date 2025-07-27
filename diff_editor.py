# diff_editor/base.py
"""Diff-based code editing for safer LLM modifications"""

import os
import subprocess
import tempfile
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DiffEdit:
    file_path: str
    original_content: str
    modified_content: str
    unified_diff: str
    description: str

@dataclass
class DiffResult:
    success: bool
    applied_files: List[str]
    failed_files: List[str]
    errors: List[str]
    diff_summary: str

class DiffBasedEditor:
    """Safe diff-based code editing system"""
    
    def __init__(self, workspace_path: str = "."):
        self.workspace_path = Path(workspace_path)
        self.artifacts_path = self.workspace_path / "artifacts" / f"run_{int(time.time())}"
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Diff editor initialized with artifacts at {self.artifacts_path}")
    
    def apply_llm_edit(self, 
                       file_path: str, 
                       llm_output: str, 
                       description: str = "",
                       context_lines: int = 3) -> DiffResult:
        """
        Apply LLM-generated edit as a diff patch
        Expects LLM output to be a unified diff format
        """
        
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return DiffResult(
                success=False,
                applied_files=[],
                failed_files=[file_path],
                errors=[f"File not found: {file_path}"],
                diff_summary=""
            )
        
        try:
            # Read original content
            original_content = file_path_obj.read_text(encoding='utf-8')
            
            # Parse LLM output as diff or full content
            if self._is_unified_diff(llm_output):
                diff_patch = llm_output
                modified_content = self._apply_diff_patch(original_content, diff_patch)
            else:
                # Treat as full file content and generate diff
                modified_content = llm_output
                diff_patch = self._generate_unified_diff(
                    original_content, 
                    modified_content, 
                    file_path,
                    context_lines
                )
            
            if modified_content is None:
                return DiffResult(
                    success=False,
                    applied_files=[],
                    failed_files=[file_path],
                    errors=["Failed to apply diff patch"],
                    diff_summary=""
                )
            
            # Create diff edit record
            diff_edit = DiffEdit(
                file_path=file_path,
                original_content=original_content,
                modified_content=modified_content,
                unified_diff=diff_patch,
                description=description
            )
            
            # Apply in safe workspace
            return self._apply_diff_safely(diff_edit)
            
        except Exception as e:
            logger.error(f"Failed to apply LLM edit to {file_path}: {e}")
            return DiffResult(
                success=False,
                applied_files=[],
                failed_files=[file_path],
                errors=[str(e)],
                diff_summary=""
            )
    
    def _is_unified_diff(self, content: str) -> bool:
        """Check if content is a unified diff format"""
        lines = content.strip().split('\n')
        
        # Look for diff headers
        has_diff_header = any(
            line.startswith(('---', '+++', '@@', 'diff --git'))
            for line in lines[:10]
        )
        
        # Look for change markers
        has_change_markers = any(
            line.startswith(('+', '-', ' '))
            for line in lines
        )
        
        return has_diff_header or has_change_markers
    
    def _generate_unified_diff(self, 
                              original: str, 
                              modified: str, 
                              filename: str,
                              context_lines: int = 3) -> str:
        """Generate unified diff between original and modified content"""
        
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=context_lines
        )
        
        return ''.join(diff)
    
    def _apply_diff_patch(self, original_content: str, diff_patch: str) -> Optional[str]:
        """Apply unified diff patch to original content"""
        
        try:
            # Write original content to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False) as f:
                f.write(original_content)
                temp_file = f.name
            
            # Write diff to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(diff_patch)
                patch_file = f.name
            
            # Apply patch using git apply
            result = subprocess.run(
                ['git', 'apply', '--whitespace=nowarn', patch_file],
                cwd=str(Path(temp_file).parent),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Read modified content
                modified_content = Path(temp_file).read_text(encoding='utf-8')
                
                # Clean up
                os.unlink(temp_file)
                os.unlink(patch_file)
                
                return modified_content
            else:
                logger.error(f"Patch application failed: {result.stderr}")
                
                # Clean up
                os.unlink(temp_file)
                os.unlink(patch_file)
                
                return None
                
        except Exception as e:
            logger.error(f"Error applying patch: {e}")
            return None
    
    def _apply_diff_safely(self, diff_edit: DiffEdit) -> DiffResult:
        """Apply diff in a safe workspace with validation"""
        
        file_path = Path(diff_edit.file_path)
        
        try:
            # Store original content for rollback
            backup_path = self.artifacts_path / f"{file_path.name}.backup"
            backup_path.write_text(diff_edit.original_content, encoding='utf-8')
            
            # Save diff for review
            diff_path = self.artifacts_path / f"{file_path.name}.diff"
            diff_path.write_text(diff_edit.unified_diff, encoding='utf-8')
            
            # Apply changes to actual file
            file_path.write_text(diff_edit.modified_content, encoding='utf-8')
            
            # Validate the changes
            validation_errors = self._validate_changes(file_path, diff_edit)
            
            if validation_errors:
                # Rollback on validation failure
                file_path.write_text(diff_edit.original_content, encoding='utf-8')
                
                return DiffResult(
                    success=False,
                    applied_files=[],
                    failed_files=[str(file_path)],
                    errors=validation_errors,
                    diff_summary=diff_edit.unified_diff
                )
            
            # Success
            return DiffResult(
                success=True,
                applied_files=[str(file_path)],
                failed_files=[],
                errors=[],
                diff_summary=diff_edit.unified_diff
            )
            
        except Exception as e:
            logger.error(f"Error applying diff safely: {e}")
            return DiffResult(
                success=False,
                applied_files=[],
                failed_files=[str(file_path)],
                errors=[str(e)],
                diff_summary=diff_edit.unified_diff
            )
    
    def _validate_changes(self, file_path: Path, diff_edit: DiffEdit) -> List[str]:
        """Validate that changes don't break the code"""
        errors = []
        
        # Syntax validation based on file type
        if file_path.suffix == '.py':
            errors.extend(self._validate_python_syntax(file_path))
        elif file_path.suffix in ['.js', '.ts', '.jsx', '.tsx']:
            errors.extend(self._validate_js_syntax(file_path))
        elif file_path.suffix == '.json':
            errors.extend(self._validate_json_syntax(file_path))
        
        # Size validation (prevent massive files)
        file_size = file_path.stat().st_size
        if file_size > 1024 * 1024:  # 1MB
            errors.append(f"File too large after edit: {file_size} bytes")
        
        # Check for dangerous patterns
        content = file_path.read_text(encoding='utf-8')
        dangerous_patterns = [
            'eval(',
            'exec(',
            '__import__',
            'subprocess.call',
            'os.system'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in content and pattern not in diff_edit.original_content:
                errors.append(f"Potentially dangerous code added: {pattern}")
        
        return errors
    
    def _validate_python_syntax(self, file_path: Path) -> List[str]:
        """Validate Python syntax"""
        import ast
        
        try:
            content = file_path.read_text(encoding='utf-8')
            ast.parse(content)
            return []
        except SyntaxError as e:
            return [f"Python syntax error: {e}"]
        except Exception as e:
            return [f"Python validation error: {e}"]
    
    def _validate_js_syntax(self, file_path: Path) -> List[str]:
        """Validate JavaScript/TypeScript syntax"""
        try:
            # Try with TypeScript compiler
            result = subprocess.run(
                ['npx', 'tsc', '--noEmit', '--skipLibCheck', str(file_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return []
            else:
                return [f"TypeScript syntax error: {result.stderr.strip()}"]
                
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback: just check if file is readable
            try:
                file_path.read_text(encoding='utf-8')
                return []
            except Exception as e:
                return [f"File encoding error: {e}"]
    
    def _validate_json_syntax(self, file_path: Path) -> List[str]:
        """Validate JSON syntax"""
        import json
        
        try:
            content = file_path.read_text(encoding='utf-8')
            json.loads(content)
            return []
        except json.JSONDecodeError as e:
            return [f"JSON syntax error: {e}"]
        except Exception as e:
            return [f"JSON validation error: {e}"]

# diff_editor/llm_diff_prompter.py
"""LLM prompting system for generating proper diffs"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import textwrap

class LLMDiffPrompter:
    """Generate prompts that encourage LLM to output proper diffs"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def generate_code_edit_prompt(self, 
                                 file_path: str, 
                                 original_content: str, 
                                 edit_request: str,
                                 context_lines: int = 3) -> str:
        """Generate prompt for LLM to create a diff-based edit"""
        
        # Show file structure for context
        lines = original_content.split('\n')
        total_lines = len(lines)
        
        # Show first few and last few lines for context
        context_preview = []
        if total_lines > 20:
            context_preview.extend(lines[:5])
            context_preview.append(f"... ({total_lines - 10} more lines) ...")
            context_preview.extend(lines[-5:])
        else:
            context_preview = lines
        
        prompt = f"""
You are a senior software engineer making precise code changes. Your task is to modify the file `{file_path}` according to the request.

CRITICAL INSTRUCTIONS:
1. Output ONLY a unified diff format (like `git diff`)
2. Use exactly this format:
   ```diff
   --- a/{file_path}
   +++ b/{file_path}
   @@ -start_line,num_lines +start_line,num_lines @@
   -removed_line
   +added_line
    unchanged_line
   ```
3. Include {context_lines} lines of context around changes
4. Make minimal, surgical changes only
5. Do NOT output the entire file - only the diff

CURRENT FILE CONTENT ({file_path}):
```
{chr(10).join(f"{i+1:4d}: {line}" for i, line in enumerate(context_preview))}
```

EDIT REQUEST:
{edit_request}

OUTPUT REQUIREMENTS:
- Start with `--- a/{file_path}`
- Use proper diff format with @@ headers
- Show only changed sections with context
- Ensure changes are precise and minimal
- Test that your diff would apply cleanly

Generate the unified diff now:
"""
        return prompt.strip()
    
    def request_diff_edit(self, 
                         file_path: str, 
                         original_content: str, 
                         edit_request: str) -> Optional[str]:
        """Request LLM to generate a diff edit"""
        
        prompt = self.generate_code_edit_prompt(file_path, original_content, edit_request)
        
        try:
            from llm_client.base import ChatMessage
            
            messages = [
                ChatMessage(role="system", content="You are a precision code editor that outputs unified diffs."),
                ChatMessage(role="user", content=prompt)
            ]
            
            response = self.llm_client.generate(
                messages=messages,
                temperature=0.1,  # Low temperature for precision
                max_tokens=2048
            )
            
            diff_content = response.content.strip()
            
            # Clean up the response (remove code block markers if present)
            if diff_content.startswith('```diff'):
                lines = diff_content.split('\n')
                diff_content = '\n'.join(lines[1:-1])  # Remove first and last line
            elif diff_content.startswith('```'):
                lines = diff_content.split('\n')
                diff_content = '\n'.join(lines[1:-1])
            
            return diff_content
            
        except Exception as e:
            logger.error(f"Failed to generate diff with LLM: {e}")
            return None

# diff_editor/tools.py
"""CrewAI tools for diff-based editing"""

from crewai.tools import tool
from typing import Dict, Any
import tempfile
import time

# Global diff editor instance
_diff_editor = None

def get_diff_editor():
    global _diff_editor
    if _diff_editor is None:
        _diff_editor = DiffBasedEditor()
    return _diff_editor

@tool("apply_code_diff")
def apply_code_diff(file_path: str, edit_description: str, diff_patch: str = "") -> Dict[str, Any]:
    """
    Apply a code change using diff format for safety and reviewability.
    
    Args:
        file_path: Path to the file to edit
        edit_description: Description of what changes are being made
        diff_patch: Unified diff patch (optional - will generate if not provided)
    
    Returns:
        Dict with success status, applied files, and diff summary
    """
    editor = get_diff_editor()
    
    if diff_patch:
        # Apply provided diff directly
        result = editor.apply_llm_edit(file_path, diff_patch, edit_description)
    else:
        # Generate diff using LLM
        from llm_client.factory import create_llm_client
        
        llm_client = create_llm_client()
        prompter = LLMDiffPrompter(llm_client)
        
        # Read current file
        from pathlib import Path
        current_content = Path(file_path).read_text(encoding='utf-8')
        
        # Generate diff
        generated_diff = prompter.request_diff_edit(file_path, current_content, edit_description)
        
        if generated_diff:
            result = editor.apply_llm_edit(file_path, generated_diff, edit_description)
        else:
            return {
                "success": False,
                "error": "Failed to generate diff patch",
                "applied_files": [],
                "diff_summary": ""
            }
    
    return {
        "success": result.success,
        "applied_files": result.applied_files,
        "failed_files": result.failed_files,
        "errors": result.errors,
        "diff_summary": result.diff_summary,
        "artifacts_path": str(editor.artifacts_path)
    }

@tool("optimize_code_with_diff")
def optimize_code_with_diff(file_path: str, optimization_type: str = "performance") -> Dict[str, Any]:
    """
    Optimize code using AI with diff-based output for safety.
    
    Args:
        file_path: Path to file to optimize
        optimization_type: Type of optimization (performance, readability, maintainability)
    
    Returns:
        Dict with optimization results and diff
    """
    from llm_client.factory import create_llm_client
    from pathlib import Path
    
    if not Path(file_path).exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "diff_summary": ""
        }
    
    try:
        llm_client = create_llm_client()
        prompter = LLMDiffPrompter(llm_client)
        
        current_content = Path(file_path).read_text(encoding='utf-8')
        
        optimization_request = f"""
Optimize this code for {optimization_type}. Focus on:
- {optimization_type.title()} improvements
- Code clarity and maintainability  
- Best practices for the language
- Proper error handling

Make minimal, surgical changes. Do not rewrite the entire file.
"""
        
        diff_patch = prompter.request_diff_edit(file_path, current_content, optimization_request)
        
        if diff_patch:
            editor = get_diff_editor()
            result = editor.apply_llm_edit(file_path, diff_patch, f"AI optimization: {optimization_type}")
            
            return {
                "success": result.success,
                "optimization_type": optimization_type,
                "applied_files": result.applied_files,
                "errors": result.errors,
                "diff_summary": result.diff_summary,
                "artifacts_path": str(editor.artifacts_path)
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate optimization diff",
                "diff_summary": ""
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "diff_summary": ""
        }

@tool("refactor_with_diff")
def refactor_with_diff(file_path: str, refactor_description: str) -> Dict[str, Any]:
    """
    Refactor code using AI with diff-based output.
    
    Args:
        file_path: Path to file to refactor
        refactor_description: Description of desired refactoring
    
    Returns:
        Dict with refactoring results and diff
    """
    return apply_code_diff(file_path, f"Refactoring: {refactor_description}")

@tool("fix_code_with_diff") 
def fix_code_with_diff(file_path: str, issue_description: str) -> Dict[str, Any]:
    """
    Fix code issues using AI with diff-based output.
    
    Args:
        file_path: Path to file with issues
        issue_description: Description of the issue to fix
    
    Returns:
        Dict with fix results and diff
    """
    return apply_code_diff(file_path, f"Fix: {issue_description}")

@tool("review_diff_artifacts")
def review_diff_artifacts(artifacts_path: str = None) -> Dict[str, Any]:
    """
    Review all diff artifacts from recent edits.
    
    Args:
        artifacts_path: Path to artifacts directory (optional)
    
    Returns:
        Dict with summary of all diffs and changes
    """
    if artifacts_path is None:
        editor = get_diff_editor()
        artifacts_path = str(editor.artifacts_path)
    
    from pathlib import Path
    
    artifacts_dir = Path(artifacts_path)
    if not artifacts_dir.exists():
        return {
            "success": False,
            "error": f"Artifacts directory not found: {artifacts_path}"
        }
    
    summary = {
        "artifacts_path": artifacts_path,
        "diff_files": [],
        "backup_files": [],
        "total_changes": 0
    }
    
    # Collect diff files
    for diff_file in artifacts_dir.glob("*.diff"):
        diff_content = diff_file.read_text(encoding='utf-8')
        lines_changed = len([line for line in diff_content.split('\n') 
                           if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))])
        
        summary["diff_files"].append({
            "file": diff_file.name,
            "lines_changed": lines_changed,
            "diff_preview": diff_content[:200] + "..." if len(diff_content) > 200 else diff_content
        })
        summary["total_changes"] += lines_changed
    
    # Collect backup files
    for backup_file in artifacts_dir.glob("*.backup"):
        summary["backup_files"].append(backup_file.name)
    
    return {
        "success": True,
        "summary": summary
    }

# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import os
    
    # Test the diff-based editor
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        test_content = '''def hello_world():
    print("Hello, World!")
    return "hello"

def add_numbers(a, b):
    result = a + b
    return result
'''
        f.write(test_content)
        test_file = f.name
    
    try:
        # Test diff application
        editor = DiffBasedEditor()
        
        # Test with a simple diff
        diff_patch = f'''--- a/{os.path.basename(test_file)}
+++ b/{os.path.basename(test_file)}
@@ -1,7 +1,8 @@
 def hello_world():
-    print("Hello, World!")
+    print("Hello, World!")
+    print("This is a test modification")
     return "hello"
 
 def add_numbers(a, b):
     result = a + b
     return result
'''
        
        result = editor.apply_llm_edit(test_file, diff_patch, "Test modification")
        
        print(f"✅ Diff application: {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            print(f"Applied to: {result.applied_files}")
            print(f"Diff summary:\n{result.diff_summary}")
        else:
            print(f"Errors: {result.errors}")
        
        # Check the result
        modified_content = Path(test_file).read_text()
        print(f"\nModified content:\n{modified_content}")
        
    finally:
        # Clean up
        os.unlink(test_file)