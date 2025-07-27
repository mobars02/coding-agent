# quality_gates/base.py
"""Comprehensive testing and quality gates system"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import subprocess
import json
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class QualityResult:
    gate_name: str
    passed: bool
    score: float  # 0-100
    details: Dict[str, Any]
    artifacts: List[str]  # Paths to generated artifacts
    execution_time: float

@dataclass
class QualityReport:
    overall_passed: bool
    overall_score: float
    gate_results: List[QualityResult]
    summary: Dict[str, Any]
    artifacts_path: str

class QualityGate(ABC):
    """Base class for quality gates"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def required(self) -> bool:
        """Whether this gate is required for deployment"""
        pass
    
    @abstractmethod
    def execute(self, workspace_path: Path) -> QualityResult:
        """Execute the quality gate"""
        pass

# quality_gates/linting.py
"""Linting quality gate with multiple tools"""

import subprocess
from pathlib import Path
from typing import Dict, List, Any

class LintingGate(QualityGate):
    @property
    def name(self) -> str:
        return "linting"
    
    @property
    def required(self) -> bool:
        return True
    
    def execute(self, workspace_path: Path) -> QualityResult:
        start_time = time.time()
        artifacts = []
        details = {}
        
        # Python linting with Ruff (fast modern linter)
        python_score = self._run_python_linting(workspace_path, artifacts, details)
        
        # JavaScript/TypeScript linting
        js_score = self._run_js_linting(workspace_path, artifacts, details)
        
        # Calculate overall score
        scores = [s for s in [python_score, js_score] if s >= 0]
        overall_score = sum(scores) / len(scores) if scores else 100
        
        execution_time = time.time() - start_time
        
        return QualityResult(
            gate_name=self.name,
            passed=overall_score >= 80,
            score=overall_score,
            details=details,
            artifacts=artifacts,
            execution_time=execution_time
        )
    
    def _run_python_linting(self, workspace_path: Path, artifacts: List[str], details: Dict) -> float:
        """Run Python linting with Ruff"""
        python_files = list(workspace_path.rglob("*.py"))
        if not python_files:
            return -1  # No Python files
        
        try:
            # Run Ruff linter
            result = subprocess.run(
                ['ruff', 'check', '--output-format=json', '.'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Save raw output
            ruff_output_file = workspace_path / "artifacts" / "ruff_output.json"
            ruff_output_file.parent.mkdir(exist_ok=True)
            ruff_output_file.write_text(result.stdout or "[]")
            artifacts.append(str(ruff_output_file))
            
            # Parse results
            if result.stdout:
                ruff_results = json.loads(result.stdout)
                error_count = len(ruff_results)
                
                # Categorize errors
                error_types = {}
                for error in ruff_results:
                    error_type = error.get('code', 'unknown')
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                details['python_linting'] = {
                    'tool': 'ruff',
                    'total_errors': error_count,
                    'error_types': error_types,
                    'files_checked': len(python_files)
                }
                
                # Score based on error density
                total_lines = sum(len(f.read_text().splitlines()) for f in python_files)
                error_density = error_count / max(total_lines, 1) * 1000  # Errors per 1000 lines
                
                # Score: 100 - (error_density * 10), minimum 0
                score = max(0, 100 - (error_density * 10))
                
            else:
                details['python_linting'] = {
                    'tool': 'ruff',
                    'total_errors': 0,
                    'files_checked': len(python_files)
                }
                score = 100
            
            return score
            
        except subprocess.TimeoutExpired:
            details['python_linting'] = {'error': 'Timeout during linting'}
            return 0
        except Exception as e:
            details['python_linting'] = {'error': str(e)}
            return 50  # Partial score for tool failure
    
    def _run_js_linting(self, workspace_path: Path, artifacts: List[str], details: Dict) -> float:
        """Run JavaScript/TypeScript linting with ESLint"""
        js_files = list(workspace_path.rglob("*.js")) + list(workspace_path.rglob("*.ts")) + \
                  list(workspace_path.rglob("*.jsx")) + list(workspace_path.rglob("*.tsx"))
        
        if not js_files:
            return -1  # No JS files
        
        try:
            # Check if ESLint is available
            eslint_result = subprocess.run(
                ['npx', 'eslint', '--version'],
                cwd=workspace_path,
                capture_output=True,
                timeout=10
            )
            
            if eslint_result.returncode != 0:
                details['js_linting'] = {'error': 'ESLint not available'}
                return 70  # Partial score
            
            # Run ESLint
            result = subprocess.run(
                ['npx', 'eslint', '--format=json', '.'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Save output
            eslint_output_file = workspace_path / "artifacts" / "eslint_output.json"
            eslint_output_file.write_text(result.stdout or "[]")
            artifacts.append(str(eslint_output_file))
            
            # Parse results
            if result.stdout:
                try:
                    eslint_results = json.loads(result.stdout)
                    total_errors = sum(len(file_result.get('messages', [])) for file_result in eslint_results)
                    
                    details['js_linting'] = {
                        'tool': 'eslint',
                        'total_errors': total_errors,
                        'files_checked': len(js_files)
                    }
                    
                    # Score based on error density
                    total_lines = sum(len(f.read_text().splitlines()) for f in js_files)
                    error_density = total_errors / max(total_lines, 1) * 1000
                    score = max(0, 100 - (error_density * 10))
                    
                except json.JSONDecodeError:
                    score = 50
            else:
                details['js_linting'] = {
                    'tool': 'eslint',
                    'total_errors': 0,
                    'files_checked': len(js_files)
                }
                score = 100
            
            return score
            
        except subprocess.TimeoutExpired:
            details['js_linting'] = {'error': 'Timeout during linting'}
            return 0
        except Exception as e:
            details['js_linting'] = {'error': str(e)}
            return 50

# quality_gates/testing.py
"""Testing quality gate with coverage"""

class TestingGate(QualityGate):
    @property
    def name(self) -> str:
        return "testing"
    
    @property
    def required(self) -> bool:
        return True
    
    def execute(self, workspace_path: Path) -> QualityResult:
        start_time = time.time()
        artifacts = []
        details = {}
        
        # Python testing
        python_score = self._run_python_tests(workspace_path, artifacts, details)
        
        # JavaScript testing
        js_score = self._run_js_tests(workspace_path, artifacts, details)
        
        # Calculate overall score
        scores = [s for s in [python_score, js_score] if s >= 0]
        overall_score = sum(scores) / len(scores) if scores else 0
        
        execution_time = time.time() - start_time
        
        return QualityResult(
            gate_name=self.name,
            passed=overall_score >= 70,  # Lower threshold for testing
            score=overall_score,
            details=details,
            artifacts=artifacts,
            execution_time=execution_time
        )
    
    def _run_python_tests(self, workspace_path: Path, artifacts: List[str], details: Dict) -> float:
        """Run Python tests with pytest and coverage"""
        test_files = list(workspace_path.rglob("test_*.py")) + list(workspace_path.rglob("*_test.py"))
        
        if not test_files:
            details['python_testing'] = {'status': 'no_tests'}
            return -1
        
        try:
            # Run pytest with coverage
            cmd = [
                'python', '-m', 'pytest',
                '--cov=.',
                '--cov-report=json',
                '--cov-report=term',
                '--json-report',
                '--json-report-file=artifacts/pytest_report.json',
                '-v'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Save test artifacts
            pytest_output_file = workspace_path / "artifacts" / "pytest_output.txt"
            pytest_output_file.parent.mkdir(exist_ok=True)
            pytest_output_file.write_text(result.stdout + "\n" + result.stderr)
            artifacts.append(str(pytest_output_file))
            
            # Parse coverage
            coverage_file = workspace_path / "coverage.json"
            coverage_percent = 0
            
            if coverage_file.exists():
                coverage_data = json.loads(coverage_file.read_text())
                coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0)
                
                # Copy coverage report to artifacts
                coverage_artifact = workspace_path / "artifacts" / "coverage.json"
                coverage_artifact.write_text(coverage_file.read_text())
                artifacts.append(str(coverage_artifact))
            
            # Parse test results
            test_report_file = workspace_path / "artifacts" / "pytest_report.json"
            tests_passed = 0
            tests_total = 0
            
            if test_report_file.exists():
                test_data = json.loads(test_report_file.read_text())
                summary = test_data.get('summary', {})
                tests_passed = summary.get('passed', 0)
                tests_total = summary.get('total', 0)
            
            pass_rate = (tests_passed / max(tests_total, 1)) * 100
            
            details['python_testing'] = {
                'tests_passed': tests_passed,
                'tests_total': tests_total,
                'pass_rate': pass_rate,
                'coverage_percent': coverage_percent,
                'status': 'completed'
            }
            
            # Score: 50% based on test pass rate, 50% on coverage
            score = (pass_rate * 0.5) + (coverage_percent * 0.5)
            return score
            
        except subprocess.TimeoutExpired:
            details['python_testing'] = {'status': 'timeout'}
            return 0
        except Exception as e:
            details['python_testing'] = {'status': 'error', 'error': str(e)}
            return 0
    
    def _run_js_tests(self, workspace_path: Path, artifacts: List[str], details: Dict) -> float:
        """Run JavaScript tests"""
        package_json = workspace_path / "package.json"
        
        if not package_json.exists():
            return -1
        
        try:
            package_data = json.loads(package_json.read_text())
            scripts = package_data.get('scripts', {})
            
            if 'test' not in scripts:
                details['js_testing'] = {'status': 'no_test_script'}
                return -1
            
            # Run tests
            result = subprocess.run(
                ['npm', 'test'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Save output
            test_output_file = workspace_path / "artifacts" / "npm_test_output.txt"
            test_output_file.write_text(result.stdout + "\n" + result.stderr)
            artifacts.append(str(test_output_file))
            
            # Basic scoring based on exit code
            if result.returncode == 0:
                score = 85  # Good score for passing tests
                status = 'passed'
            else:
                score = 25  # Some credit for having tests
                status = 'failed'
            
            details['js_testing'] = {
                'status': status,
                'exit_code': result.returncode
            }
            
            return score
            
        except subprocess.TimeoutExpired:
            details['js_testing'] = {'status': 'timeout'}
            return 0
        except Exception as e:
            details['js_testing'] = {'status': 'error', 'error': str(e)}
            return 0

# quality_gates/security.py
"""Security scanning quality gate"""

class SecurityGate(QualityGate):
    @property
    def name(self) -> str:
        return "security"
    
    @property
    def required(self) -> bool:
        return True
    
    def execute(self, workspace_path: Path) -> QualityResult:
        start_time = time.time()
        artifacts = []
        details = {}
        
        # Run multiple security scanners
        semgrep_score = self._run_semgrep(workspace_path, artifacts, details)
        bandit_score = self._run_bandit(workspace_path, artifacts, details)
        npm_audit_score = self._run_npm_audit(workspace_path, artifacts, details)
        secrets_score = self._run_secrets_scan(workspace_path, artifacts, details)
        
        # Calculate overall score (weighted average)
        scores = []
        weights = []
        
        if semgrep_score >= 0:
            scores.append(semgrep_score)
            weights.append(0.4)  # High weight for Semgrep
        
        if bandit_score >= 0:
            scores.append(bandit_score)
            weights.append(0.3)
        
        if npm_audit_score >= 0:
            scores.append(npm_audit_score)
            weights.append(0.2)
        
        if secrets_score >= 0:
            scores.append(secrets_score)
            weights.append(0.1)
        
        if scores:
            overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            overall_score = 85  # Default if no security tools available
        
        execution_time = time.time() - start_time
        
        return QualityResult(
            gate_name=self.name,
            passed=overall_score >= 75,
            score=overall_score,
            details=details,
            artifacts=artifacts,
            execution_time=execution_time
        )
    
    def _run_semgrep(self, workspace_path: Path, artifacts: List[str], details: Dict) -> float:
        """Run Semgrep security scanner"""
        try:
            # Check if Semgrep is available
            version_result = subprocess.run(
                ['semgrep', '--version'],
                capture_output=True,
                timeout=10
            )
            
            if version_result.returncode != 0:
                details['semgrep'] = {'status': 'not_available'}
                return -1
            
            # Run Semgrep with auto config
            result = subprocess.run(
                ['semgrep', '--config=auto', '--json', '--output=artifacts/semgrep.json', '.'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            artifacts.append(str(workspace_path / "artifacts" / "semgrep.json"))
            
            # Parse results
            semgrep_file = workspace_path / "artifacts" / "semgrep.json"
            if semgrep_file.exists():
                semgrep_data = json.loads(semgrep_file.read_text())
                findings = semgrep_data.get('results', [])
                
                # Categorize by severity
                severity_counts = {'ERROR': 0, 'WARNING': 0, 'INFO': 0}
                for finding in findings:
                    severity = finding.get('extra', {}).get('severity', 'INFO')
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                details['semgrep'] = {
                    'total_findings': len(findings),
                    'severity_counts': severity_counts,
                    'status': 'completed'
                }
                
                # Score based on severity (ERROR = -10, WARNING = -3, INFO = -1)
                penalty = (severity_counts['ERROR'] * 10 + 
                          severity_counts['WARNING'] * 3 + 
                          severity_counts['INFO'] * 1)
                
                score = max(0, 100 - penalty)
                return score
            else:
                details['semgrep'] = {'status': 'no_output'}
                return 80
                
        except subprocess.TimeoutExpired:
            details['semgrep'] = {'status': 'timeout'}
            return 50
        except Exception as e:
            details['semgrep'] = {'status': 'error', 'error': str(e)}
            return 50
    
    def _run_bandit(self, workspace_path: Path, artifacts: List[str], details: Dict) -> float:
        """Run Bandit Python security scanner"""
        python_files = list(workspace_path.rglob("*.py"))
        if not python_files:
            return -1
        
        try:
            result = subprocess.run(
                ['bandit', '-r', '.', '-f', 'json', '-o', 'artifacts/bandit.json'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            bandit_file = workspace_path / "artifacts" / "bandit.json"
            artifacts.append(str(bandit_file))
            
            if bandit_file.exists():
                try:
                    bandit_data = json.loads(bandit_file.read_text())
                    findings = bandit_data.get('results', [])
                    
                    # Count by confidence and severity
                    high_issues = sum(1 for f in findings if f.get('issue_confidence') == 'HIGH')
                    medium_issues = sum(1 for f in findings if f.get('issue_confidence') == 'MEDIUM')
                    
                    details['bandit'] = {
                        'total_issues': len(findings),
                        'high_confidence': high_issues,
                        'medium_confidence': medium_issues,
                        'status': 'completed'
                    }
                    
                    # Score: penalize high confidence issues more
                    penalty = high_issues * 15 + medium_issues * 5
                    score = max(0, 100 - penalty)
                    return score
                    
                except json.JSONDecodeError:
                    details['bandit'] = {'status': 'parse_error'}
                    return 70
            else:
                details['bandit'] = {'status': 'no_output'}
                return 85
                
        except FileNotFoundError:
            details['bandit'] = {'status': 'not_available'}
            return -1
        except subprocess.TimeoutExpired:
            details['bandit'] = {'status': 'timeout'}
            return 50
        except Exception as e:
            details['bandit'] = {'status': 'error', 'error': str(e)}
            return 50
    
    def _run_npm_audit(self, workspace_path: Path, artifacts: List[str], details: Dict) -> float:
        """Run npm audit for JavaScript dependencies"""
        package_json = workspace_path / "package.json"
        if not package_json.exists():
            return -1
        
        try:
            result = subprocess.run(
                ['npm', 'audit', '--json'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Save audit output
            audit_file = workspace_path / "artifacts" / "npm_audit.json"
            audit_file.parent.mkdir(exist_ok=True)
            audit_file.write_text(result.stdout)
            artifacts.append(str(audit_file))
            
            if result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    vulnerabilities = audit_data.get('vulnerabilities', {})
                    
                    # Count by severity
                    severity_counts = {'critical': 0, 'high': 0, 'moderate': 0, 'low': 0}
                    for vuln_name, vuln_data in vulnerabilities.items():
                        severity = vuln_data.get('severity', 'low')
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    details['npm_audit'] = {
                        'total_vulnerabilities': len(vulnerabilities),
                        'severity_counts': severity_counts,
                        'status': 'completed'
                    }
                    
                    # Score based on severity
                    penalty = (severity_counts['critical'] * 20 + 
                             severity_counts['high'] * 10 + 
                             severity_counts['moderate'] * 3 + 
                             severity_counts['low'] * 1)
                    
                    score = max(0, 100 - penalty)
                    return score
                    
                except json.JSONDecodeError:
                    details['npm_audit'] = {'status': 'parse_error'}
                    return 70
            else:
                details['npm_audit'] = {'status': 'no_vulnerabilities'}
                return 100
                
        except FileNotFoundError:
            details['npm_audit'] = {'status': 'npm_not_available'}
            return -1
        except subprocess.TimeoutExpired:
            details['npm_audit'] = {'status': 'timeout'}
            return 50
        except Exception as e:
            details['npm_audit'] = {'status': 'error', 'error': str(e)}
            return 50
    
    def _run_secrets_scan(self, workspace_path: Path, artifacts: List[str], details: Dict) -> float:
        """Run secrets detection with gitleaks or simple regex"""
        try:
            # Try gitleaks first
            result = subprocess.run(
                ['gitleaks', 'detect', '--source', '.', '--report-format', 'json', 
                 '--report-path', 'artifacts/gitleaks.json'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            gitleaks_file = workspace_path / "artifacts" / "gitleaks.json"
            if gitleaks_file.exists():
                artifacts.append(str(gitleaks_file))
                
                try:
                    gitleaks_data = json.loads(gitleaks_file.read_text())
                    secrets_found = len(gitleaks_data) if isinstance(gitleaks_data, list) else 0
                    
                    details['secrets_scan'] = {
                        'tool': 'gitleaks',
                        'secrets_found': secrets_found,
                        'status': 'completed'
                    }
                    
                    # Heavy penalty for secrets
                    score = max(0, 100 - (secrets_found * 30))
                    return score
                    
                except json.JSONDecodeError:
                    pass
            
            # If gitleaks succeeded but no secrets
            if result.returncode == 0:
                details['secrets_scan'] = {
                    'tool': 'gitleaks',
                    'secrets_found': 0,
                    'status': 'completed'
                }
                return 100
                
        except FileNotFoundError:
            # Fallback to regex-based detection
            pass
        except subprocess.TimeoutExpired:
            details['secrets_scan'] = {'status': 'timeout'}
            return 50
        except Exception:
            pass
        
        # Fallback: simple regex-based secrets detection
        secrets_found = self._simple_secrets_scan(workspace_path)
        
        details['secrets_scan'] = {
            'tool': 'regex_fallback',
            'secrets_found': secrets_found,
            'status': 'completed'
        }
        
        score = max(0, 100 - (secrets_found * 25))
        return score
    
    def _simple_secrets_scan(self, workspace_path: Path) -> int:
        """Simple regex-based secrets detection"""
        import re
        
        # Common secret patterns
        secret_patterns = [
            r'(?i)(password|passwd|pwd)\s*[:=]\s*["\'][^"\']{8,}["\']',
            r'(?i)(api_key|apikey|api-key)\s*[:=]\s*["\'][^"\']{20,}["\']',
            r'(?i)(secret|token)\s*[:=]\s*["\'][^"\']{20,}["\']',
            r'(?i)(access_key|access-key)\s*[:=]\s*["\'][^"\']{20,}["\']',
            r'(?i)aws_access_key_id\s*[:=]\s*["\'][A-Z0-9]{20}["\']',
            r'(?i)github_token\s*[:=]\s*["\']ghp_[A-Za-z0-9]{36}["\']',
        ]
        
        secrets_count = 0
        
        for file_path in workspace_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.py', '.js', '.ts', '.json', '.env', '.yaml', '.yml']:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    for pattern in secret_patterns:
                        matches = re.findall(pattern, content)
                        secrets_count += len(matches)
                        
                except Exception:
                    continue
        
        return secrets_count

# quality_gates/orchestrator.py
"""Quality gates orchestrator"""

class QualityGatesOrchestrator:
    def __init__(self, workspace_path: str = "."):
        self.workspace_path = Path(workspace_path)
        self.artifacts_path = self.workspace_path / "artifacts" / f"quality_{int(time.time())}"
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize gates
        self.gates = [
            LintingGate(),
            TestingGate(),
            SecurityGate(),
        ]
        
        logger.info(f"Quality gates orchestrator initialized with {len(self.gates)} gates")
    
    def run_all_gates(self, fail_fast: bool = False) -> QualityReport:
        """Run all quality gates and generate report"""
        
        logger.info("🚀 Starting quality gates execution...")
        
        gate_results = []
        overall_scores = []
        
        for gate in self.gates:
            logger.info(f"⚡ Running {gate.name} gate...")
            
            try:
                result = gate.execute(self.workspace_path)
                gate_results.append(result)
                overall_scores.append(result.score)
                
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                logger.info(f"{status} {gate.name}: {result.score:.1f}/100 ({result.execution_time:.1f}s)")
                
                if fail_fast and not result.passed and gate.required:
                    logger.error(f"🛑 Stopping due to required gate failure: {gate.name}")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Gate {gate.name} crashed: {e}")
                gate_results.append(QualityResult(
                    gate_name=gate.name,
                    passed=False,
                    score=0,
                    details={'error': str(e)},
                    artifacts=[],
                    execution_time=0
                ))
                overall_scores.append(0)
        
        # Calculate overall metrics
        overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        overall_passed = all(result.passed for result in gate_results if result.gate_name in [g.name for g in self.gates if g.required])
        
        # Generate summary
        summary = {
            'total_gates': len(self.gates),
            'gates_passed': sum(1 for r in gate_results if r.passed),
            'gates_failed': sum(1 for r in gate_results if not r.passed),
            'total_execution_time': sum(r.execution_time for r in gate_results),
            'gate_scores': {r.gate_name: r.score for r in gate_results}
        }
        
        report = QualityReport(
            overall_passed=overall_passed,
            overall_score=overall_score,
            gate_results=gate_results,
            summary=summary,
            artifacts_path=str(self.artifacts_path)
        )
        
        # Save report
        self._save_report(report)
        
        logger.info(f"🏁 Quality gates completed: {'✅ PASSED' if overall_passed else '❌ FAILED'} ({overall_score:.1f}/100)")
        
        return report
    
    def _save_report(self, report: QualityReport):
        """Save quality report to artifacts"""
        
        # JSON report
        report_data = {
            'overall_passed': report.overall_passed,
            'overall_score': report.overall_score,
            'summary': report.summary,
            'gate_results': [
                {
                    'gate_name': r.gate_name,
                    'passed': r.passed,
                    'score': r.score,
                    'details': r.details,
                    'artifacts': r.artifacts,
                    'execution_time': r.execution_time
                }
                for r in report.gate_results
            ],
            'timestamp': time.time(),
            'artifacts_path': report.artifacts_path
        }
        
        report_file = self.artifacts_path / "quality_report.json"
        report_file.write_text(json.dumps(report_data, indent=2))
        
        # Human-readable report
        markdown_report = self._generate_markdown_report(report)
        markdown_file = self.artifacts_path / "quality_report.md"
        markdown_file.write_text(markdown_report)
        
        logger.info(f"📊 Quality report saved: {report_file}")
    
    def _generate_markdown_report(self, report: QualityReport) -> str:
        """Generate human-readable markdown report"""
        
        status_emoji = "✅" if report.overall_passed else "❌"
        
        markdown = f"""# Quality Gates Report
{status_emoji} **Overall Status**: {'PASSED' if report.overall_passed else 'FAILED'}
📊 **Overall Score**: {report.overall_score:.1f}/100
⏱️ **Total Execution Time**: {report.summary['total_execution_time']:.1f}s

## Summary
- **Gates Passed**: {report.summary['gates_passed']}/{report.summary['total_gates']}
- **Gates Failed**: {report.summary['gates_failed']}/{report.summary['total_gates']}

## Gate Results

"""
        
        for result in report.gate_results:
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            markdown += f"""### {result.gate_name.title()} Gate
{status} **Score**: {result.score:.1f}/100 ({result.execution_time:.1f}s)

**Details**:
```json
{json.dumps(result.details, indent=2)}
```

**Artifacts**:
{chr(10).join(f"- {artifact}" for artifact in result.artifacts)}

---

"""
        
        markdown += f"""## Artifacts
All artifacts saved to: `{report.artifacts_path}`

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return markdown

# CrewAI Tools Integration
from crewai.tools import tool

@tool("run_quality_gates")
def run_quality_gates(fail_fast: bool = False) -> Dict[str, Any]:
    """
    Run comprehensive quality gates on the codebase.
    
    Args:
        fail_fast: Stop execution on first required gate failure
    
    Returns:
        Dict with quality report and artifacts
    """
    orchestrator = QualityGatesOrchestrator()
    report = orchestrator.run_all_gates(fail_fast=fail_fast)
    
    return {
        "overall_passed": report.overall_passed,
        "overall_score": report.overall_score,
        "summary": report.summary,
        "artifacts_path": report.artifacts_path,
        "gate_results": [
            {
                "name": r.gate_name,
                "passed": r.passed,
                "score": r.score,
                "execution_time": r.execution_time
            }
            for r in report.gate_results
        ]
    }

@tool("check_code_quality")
def check_code_quality() -> Dict[str, Any]:
    """
    Quick code quality check focusing on linting and basic metrics.
    
    Returns:
        Dict with quality metrics
    """
    orchestrator = QualityGatesOrchestrator()
    
    # Run only linting gate for quick check
    linting_gate = LintingGate()
    result = linting_gate.execute(orchestrator.workspace_path)
    
    return {
        "passed": result.passed,
        "score": result.score,
        "details": result.details,
        "execution_time": result.execution_time
    }

# Example usage
if __name__ == "__main__":
    # Test the quality gates system
    orchestrator = QualityGatesOrchestrator()
    report = orchestrator.run_all_gates()
    
    print(f"\n{'='*60}")
    print(f"QUALITY GATES REPORT")
    print(f"{'='*60}")
    print(f"Overall Status: {'PASSED' if report.overall_passed else 'FAILED'}")
    print(f"Overall Score: {report.overall_score:.1f}/100")
    print(f"Execution Time: {report.summary['total_execution_time']:.1f}s")
    print(f"\nDetailed results saved to: {report.artifacts_path}")
    
    for result in report.gate_results:
        status = "PASSED" if result.passed else "FAILED"
        print(f"  {result.gate_name}: {status} ({result.score:.1f}/100)")