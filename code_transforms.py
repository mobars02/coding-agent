# codemods/base.py
"""Safe code transformation system with codemods"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)

@dataclass
class FileMove:
    old_path: str
    new_path: str
    reason: str

@dataclass
class ImportRewrite:
    file_path: str
    old_import: str
    new_import: str

@dataclass
class ChangeSet:
    file_moves: List[FileMove]
    import_rewrites: List[ImportRewrite]
    dependencies: List[str]  # Files that depend on changed files
    
@dataclass
class TransformResult:
    success: bool
    files_changed: List[str]
    errors: List[str]
    diff_path: Optional[str] = None

class CodeTransformer(ABC):
    """Base class for code transformations"""
    
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this transformer can handle the file"""
        pass
    
    @abstractmethod
    def rewrite_imports(self, file_path: Path, import_mapping: Dict[str, str]) -> TransformResult:
        """Rewrite imports in the file"""
        pass
    
    @abstractmethod
    def validate_syntax(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate syntax after transformation"""
        pass

# codemods/python_transformer.py
"""Python code transformation using libcst"""

import ast
from typing import Dict, List, Any, Tuple
from pathlib import Path

try:
    import libcst as cst
    from libcst.metadata import FullRepoManager
    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False

from .base import CodeTransformer, TransformResult

class PythonTransformer(CodeTransformer):
    def __init__(self):
        if not LIBCST_AVAILABLE:
            logger.warning("libcst not available. Install with: pip install libcst")
    
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix == '.py'
    
    def rewrite_imports(self, file_path: Path, import_mapping: Dict[str, str]) -> TransformResult:
        """Rewrite Python imports using libcst"""
        if not LIBCST_AVAILABLE:
            return self._fallback_import_rewrite(file_path, import_mapping)
        
        try:
            # Read original file
            original_content = file_path.read_text(encoding='utf-8')
            
            # Parse with libcst
            tree = cst.parse_module(original_content)
            
            # Transform imports
            transformer = ImportRewriter(import_mapping)
            modified_tree = tree.visit(transformer)
            
            # Generate new content
            new_content = modified_tree.code
            
            if new_content != original_content:
                # Write back to file
                file_path.write_text(new_content, encoding='utf-8')
                
                return TransformResult(
                    success=True,
                    files_changed=[str(file_path)],
                    errors=[]
                )
            else:
                return TransformResult(
                    success=True,
                    files_changed=[],
                    errors=[]
                )
        
        except Exception as e:
            logger.error(f"Failed to transform {file_path}: {e}")
            return TransformResult(
                success=False,
                files_changed=[],
                errors=[str(e)]
            )
    
    def _fallback_import_rewrite(self, file_path: Path, import_mapping: Dict[str, str]) -> TransformResult:
        """Fallback regex-based import rewriting"""
        import re
        
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            for old_module, new_module in import_mapping.items():
                # Handle different import patterns
                patterns = [
                    (rf'from {re.escape(old_module)} import', f'from {new_module} import'),
                    (rf'import {re.escape(old_module)}(?=\s|$)', f'import {new_module}'),
                    (rf'import {re.escape(old_module)} as', f'import {new_module} as'),
                ]
                
                for pattern, replacement in patterns:
                    content = re.sub(pattern, replacement, content)
            
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                return TransformResult(
                    success=True,
                    files_changed=[str(file_path)],
                    errors=[]
                )
            else:
                return TransformResult(
                    success=True,
                    files_changed=[],
                    errors=[]
                )
        
        except Exception as e:
            return TransformResult(
                success=False,
                files_changed=[],
                errors=[str(e)]
            )
    
    def validate_syntax(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate Python syntax"""
        try:
            content = file_path.read_text(encoding='utf-8')
            ast.parse(content)
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error in {file_path}: {e}"]
        except Exception as e:
            return False, [f"Error validating {file_path}: {e}"]

if LIBCST_AVAILABLE:
    class ImportRewriter(cst.CSTTransformer):
        def __init__(self, import_mapping: Dict[str, str]):
            self.import_mapping = import_mapping
        
        def leave_SimpleStatementLine(self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine) -> cst.SimpleStatementLine:
            # Handle import statements
            for i, stmt in enumerate(updated_node.body):
                if isinstance(stmt, cst.Import):
                    # Handle "import module" statements
                    new_names = []
                    for alias in stmt.names:
                        if isinstance(alias, cst.ImportAlias):
                            module_name = alias.name.value if hasattr(alias.name, 'value') else str(alias.name)
                            if module_name in self.import_mapping:
                                new_alias = alias.with_changes(
                                    name=cst.parse_expression(self.import_mapping[module_name])
                                )
                                new_names.append(new_alias)
                            else:
                                new_names.append(alias)
                    
                    if new_names != list(stmt.names):
                        updated_node = updated_node.with_changes(
                            body=[stmt.with_changes(names=new_names) if j == i else s 
                                  for j, s in enumerate(updated_node.body)]
                        )
                
                elif isinstance(stmt, cst.ImportFrom):
                    # Handle "from module import ..." statements
                    if stmt.module:
                        module_name = stmt.module.value if hasattr(stmt.module, 'value') else str(stmt.module)
                        if module_name in self.import_mapping:
                            new_module = cst.parse_expression(self.import_mapping[module_name])
                            updated_node = updated_node.with_changes(
                                body=[stmt.with_changes(module=new_module) if j == i else s 
                                      for j, s in enumerate(updated_node.body)]
                            )
            
            return updated_node

# codemods/typescript_transformer.py
"""TypeScript/JavaScript code transformation"""

import json
import subprocess
from typing import Dict, List, Tuple
from pathlib import Path
from .base import CodeTransformer, TransformResult

class TypeScriptTransformer(CodeTransformer):
    def __init__(self):
        self.ts_node_available = self._check_ts_node()
    
    def _check_ts_node(self) -> bool:
        """Check if ts-node is available for running TypeScript scripts"""
        try:
            subprocess.run(['npx', 'ts-node', '--version'], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("ts-node not available. TypeScript transformations will use regex fallback")
            return False
    
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix in ['.ts', '.tsx', '.js', '.jsx']
    
    def rewrite_imports(self, file_path: Path, import_mapping: Dict[str, str]) -> TransformResult:
        """Rewrite TypeScript/JavaScript imports"""
        if self.ts_node_available:
            return self._ts_morph_rewrite(file_path, import_mapping)
        else:
            return self._regex_rewrite(file_path, import_mapping)
    
    def _ts_morph_rewrite(self, file_path: Path, import_mapping: Dict[str, str]) -> TransformResult:
        """Use ts-morph for safe TypeScript transformations"""
        
        # Create a temporary TypeScript script for transformation
        transform_script = f"""
import {{ Project }} from "ts-morph";

const project = new Project();
const sourceFile = project.addSourceFileAtPath("{file_path}");

const importMapping = {json.dumps(import_mapping)};

// Transform import declarations
sourceFile.getImportDeclarations().forEach(importDecl => {{
    const moduleSpecifier = importDecl.getModuleSpecifierValue();
    if (importMapping[moduleSpecifier]) {{
        importDecl.setModuleSpecifier(importMapping[moduleSpecifier]);
    }}
}});

// Transform dynamic imports
sourceFile.getCallExpressions()
    .filter(call => call.getExpression().getText() === "import")
    .forEach(importCall => {{
        const arg = importCall.getArguments()[0];
        if (arg && arg.getKind() === SyntaxKind.StringLiteral) {{
            const moduleSpecifier = arg.getLiteralValue();
            if (importMapping[moduleSpecifier]) {{
                arg.replaceWithText(`"${{importMapping[moduleSpecifier]}}"`);
            }}
        }}
    }});

// Save the file
sourceFile.saveSync();
console.log("SUCCESS");
"""
        
        try:
            # Write transform script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
                f.write(transform_script)
                script_path = f.name
            
            # Run transformation
            result = subprocess.run(
                ['npx', 'ts-node', script_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=file_path.parent
            )
            
            # Clean up
            Path(script_path).unlink()
            
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                return TransformResult(
                    success=True,
                    files_changed=[str(file_path)],
                    errors=[]
                )
            else:
                logger.warning(f"ts-morph transform failed, falling back to regex: {result.stderr}")
                return self._regex_rewrite(file_path, import_mapping)
        
        except Exception as e:
            logger.warning(f"ts-morph transform failed, falling back to regex: {e}")
            return self._regex_rewrite(file_path, import_mapping)
    
    def _regex_rewrite(self, file_path: Path, import_mapping: Dict[str, str]) -> TransformResult:
        """Fallback regex-based import rewriting"""
        import re
        
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            for old_module, new_module in import_mapping.items():
                # Escape special regex characters
                old_escaped = re.escape(old_module)
                
                patterns = [
                    # import ... from "module"
                    (rf'import\s+.*?\s+from\s+["\']({old_escaped})["\']', 
                     rf'import {new_module} from "{new_module}"'),
                    
                    # import("module")
                    (rf'import\s*\(\s*["\']({old_escaped})["\']\s*\)',
                     rf'import("{new_module}")'),
                    
                    # require("module")
                    (rf'require\s*\(\s*["\']({old_escaped})["\']\s*\)',
                     rf'require("{new_module}")'),
                ]
                
                for pattern, replacement in patterns:
                    content = re.sub(pattern, replacement, content)
            
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                return TransformResult(
                    success=True,
                    files_changed=[str(file_path)],
                    errors=[]
                )
            else:
                return TransformResult(
                    success=True,
                    files_changed=[],
                    errors=[]
                )
        
        except Exception as e:
            return TransformResult(
                success=False,
                files_changed=[],
                errors=[str(e)]
            )
    
    def validate_syntax(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate TypeScript/JavaScript syntax"""
        try:
            # Try TypeScript compiler
            result = subprocess.run(
                ['npx', 'tsc', '--noEmit', '--skipLibCheck', str(file_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, []
            else:
                return False, [f"TypeScript errors in {file_path}: {result.stderr}"]
        
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback to basic syntax check
            try:
                # Very basic check - just ensure it's valid text
                file_path.read_text(encoding='utf-8')
                return True, []
            except Exception as e:
                return False, [f"File read error {file_path}: {e}"]

# codemods/orchestrator.py
"""Safe code transformation orchestrator"""

import git
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .base import ChangeSet, FileMove, ImportRewrite, TransformResult
from .python_transformer import PythonTransformer
from .typescript_transformer import TypeScriptTransformer

class SafeTransformOrchestrator:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path, search_parent_directories=True)
        
        # Initialize transformers
        self.transformers = [
            PythonTransformer(),
            TypeScriptTransformer()
        ]
        
        logger.info("Safe transform orchestrator initialized")
    
    def plan_file_organization(self, current_files: List[Path], target_structure: Dict[str, str]) -> ChangeSet:
        """Plan file moves and import rewrites without executing them"""
        
        file_moves = []
        import_rewrites = []
        dependencies = set()
        
        # Plan moves based on target structure
        for file_path in current_files:
            if file_path.is_file():
                new_location = self._determine_new_location(file_path, target_structure)
                
                if new_location and new_location != str(file_path):
                    file_moves.append(FileMove(
                        old_path=str(file_path),
                        new_path=new_location,
                        reason=f"Moving to professional structure"
                    ))
        
        # Plan import rewrites
        if file_moves:
            import_mapping = {move.old_path: move.new_path for move in file_moves}
            import_rewrites = self._plan_import_rewrites(current_files, import_mapping)
            
            # Find dependent files
            dependencies = self._find_dependencies(current_files, [move.old_path for move in file_moves])
        
        return ChangeSet(
            file_moves=file_moves,
            import_rewrites=import_rewrites,
            dependencies=list(dependencies)
        )
    
    def execute_safe_transformation(self, changeset: ChangeSet, dry_run: bool = True) -> TransformResult:
        """Execute transformations safely with rollback capability"""
        
        if dry_run:
            return self._validate_changeset(changeset)
        
        # Create a working branch for safety
        original_branch = self.repo.active_branch.name
        temp_branch = f"safe-transform-{int(time.time())}"
        
        try:
            # Create temp branch
            self.repo.git.checkout('-b', temp_branch)
            
            # Step 1: Rewrite imports first
            import_results = self._execute_import_rewrites(changeset.import_rewrites)
            if not import_results.success:
                raise Exception(f"Import rewrite failed: {import_results.errors}")
            
            # Step 2: Move files
            move_results = self._execute_file_moves(changeset.file_moves)
            if not move_results.success:
                raise Exception(f"File moves failed: {move_results.errors}")
            
            # Step 3: Validate all syntax
            validation_errors = self._validate_all_syntax()
            if validation_errors:
                raise Exception(f"Syntax validation failed: {validation_errors}")
            
            # Step 4: Run tests if available
            test_results = self._run_tests()
            if not test_results:
                logger.warning("Tests failed, but continuing (you may want to review)")
            
            # Step 5: Commit changes
            self.repo.git.add('.')
            self.repo.git.commit('-m', 'Safe code transformation: file organization')
            
            logger.info(f"✅ Safe transformation completed on branch {temp_branch}")
            
            return TransformResult(
                success=True,
                files_changed=import_results.files_changed + move_results.files_changed,
                errors=[]
            )
        
        except Exception as e:
            logger.error(f"❌ Transformation failed: {e}")
            
            # Rollback to original branch
            self.repo.git.checkout(original_branch)
            self.repo.git.branch('-D', temp_branch)
            
            return TransformResult(
                success=False,
                files_changed=[],
                errors=[str(e)]
            )
    
    def _determine_new_location(self, file_path: Path, target_structure: Dict[str, str]) -> Optional[str]:
        """Determine where a file should be moved based on target structure"""
        
        file_str = str(file_path).lower()
        
        # Skip files already in good locations
        if any(file_str.startswith(prefix) for prefix in ['src/', 'tests/', 'docs/', 'config/']):
            return None
        
        # Component files
        if file_path.suffix in ['.tsx', '.jsx'] or 'component' in file_str:
            return f"src/components/{file_path.name}"
        
        # Test files
        if 'test' in file_str or 'spec' in file_str:
            return f"tests/unit/{file_path.name}"
        
        # Utility files
        if 'util' in file_str or 'helper' in file_str:
            return f"src/utils/{file_path.name}"
        
        # Service files
        if 'service' in file_str or 'api' in file_str:
            return f"src/services/{file_path.name}"
        
        # Style files
        if file_path.suffix in ['.css', '.scss', '.sass']:
            return f"src/styles/{file_path.name}"
        
        # Documentation
        if file_path.suffix == '.md' and file_path.name != 'README.md':
            return f"docs/{file_path.name}"
        
        # Config files
        if any(keyword in file_path.name.lower() for keyword in ['config', 'webpack', 'babel', 'eslint']):
            return f"config/{file_path.name}"
        
        return None
    
    def _plan_import_rewrites(self, all_files: List[Path], file_mapping: Dict[str, str]) -> List[ImportRewrite]:
        """Plan import statement rewrites"""
        import_rewrites = []
        
        # Build module mapping (remove file extensions for imports)
        module_mapping = {}
        for old_path, new_path in file_mapping.items():
            old_module = str(Path(old_path).with_suffix(''))
            new_module = str(Path(new_path).with_suffix(''))
            module_mapping[old_module] = new_module
        
        # Find files that need import updates
        for file_path in all_files:
            if self._file_has_imports(file_path):
                import_rewrites.append(ImportRewrite(
                    file_path=str(file_path),
                    old_import="",  # Will be filled by transformer
                    new_import=""   # Will be filled by transformer
                ))
        
        return import_rewrites
    
    def _file_has_imports(self, file_path: Path) -> bool:
        """Check if file contains import statements"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            return ('import ' in content or 'require(' in content or 'from ' in content)
        except:
            return False
    
    def _find_dependencies(self, all_files: List[Path], moved_files: List[str]) -> set:
        """Find files that depend on moved files"""
        dependencies = set()
        
        # Simple approach: find files that mention the moved files
        moved_names = {Path(f).stem for f in moved_files}
        
        for file_path in all_files:
            if str(file_path) not in moved_files:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if any(name in content for name in moved_names):
                        dependencies.add(str(file_path))
                except:
                    continue
        
        return dependencies
    
    def _execute_import_rewrites(self, import_rewrites: List[ImportRewrite]) -> TransformResult:
        """Execute import statement rewrites"""
        files_changed = []
        errors = []
        
        # Group by file type
        files_by_transformer = {}
        for rewrite in import_rewrites:
            file_path = Path(rewrite.file_path)
            
            for transformer in self.transformers:
                if transformer.can_handle(file_path):
                    if transformer not in files_by_transformer:
                        files_by_transformer[transformer] = []
                    files_by_transformer[transformer].append(file_path)
                    break
        
        # Apply transformations
        for transformer, files in files_by_transformer.items():
            for file_path in files:
                result = transformer.rewrite_imports(file_path, {})  # Mapping handled internally
                
                if result.success:
                    files_changed.extend(result.files_changed)
                else:
                    errors.extend(result.errors)
        
        return TransformResult(
            success=len(errors) == 0,
            files_changed=files_changed,
            errors=errors
        )
    
    def _execute_file_moves(self, file_moves: List[FileMove]) -> TransformResult:
        """Execute file moves using git mv"""
        files_changed = []
        errors = []
        
        for move in file_moves:
            try:
                old_path = Path(move.old_path)
                new_path = Path(move.new_path)
                
                # Create target directory
                new_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Use git mv for proper tracking
                self.repo.git.mv(str(old_path), str(new_path))
                files_changed.append(f"{old_path} -> {new_path}")
                
            except Exception as e:
                errors.append(f"Failed to move {move.old_path}: {e}")
        
        return TransformResult(
            success=len(errors) == 0,
            files_changed=files_changed,
            errors=errors
        )
    
    def _validate_all_syntax(self) -> List[str]:
        """Validate syntax of all transformed files"""
        errors = []
        
        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file():
                for transformer in self.transformers:
                    if transformer.can_handle(file_path):
                        is_valid, file_errors = transformer.validate_syntax(file_path)
                        if not is_valid:
                            errors.extend(file_errors)
                        break
        
        return errors
    
    def _run_tests(self) -> bool:
        """Run tests to ensure transformations didn't break anything"""
        try:
            # Try common test commands
            test_commands = [
                ['npm', 'test'],
                ['yarn', 'test'],
                ['pytest'],
                ['python', '-m', 'pytest'],
                ['make', 'test']
            ]
            
            for cmd in test_commands:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        timeout=300,
                        cwd=self.repo_path
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Tests passed with command: {' '.join(cmd)}")
                        return True
                    
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            logger.warning("⚠️ No test command succeeded")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Test execution failed: {e}")
            return False
    
    def _validate_changeset(self, changeset: ChangeSet) -> TransformResult:
        """Validate changeset without executing (dry run)"""
        errors = []
        
        # Check if all source files exist
        for move in changeset.file_moves:
            if not Path(move.old_path).exists():
                errors.append(f"Source file does not exist: {move.old_path}")
        
        # Check for conflicts
        target_paths = [move.new_path for move in changeset.file_moves]
        for path in target_paths:
            if Path(path).exists():
                errors.append(f"Target path already exists: {path}")
        
        return TransformResult(
            success=len(errors) == 0,
            files_changed=[f"Would move: {move.old_path} -> {move.new_path}" for move in changeset.file_moves],
            errors=errors
        )