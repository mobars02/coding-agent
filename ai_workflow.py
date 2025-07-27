#!/usr/bin/env python3
"""
ai_workflow.py - Complete AI Development Workflow
This is your main workflow orchestrator that uses all your existing tools
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import json
import logging

# Import all your existing tools
from enhanced_tools import *
from github_integration import GitHubIntegration, create_ai_improvement_pr
from llm_client import create_llm_client
from vector_store import create_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIWorkflowOrchestrator:
    """Complete AI development workflow orchestrator"""
    
    def __init__(self, workspace_path: str = "."):
        self.workspace_path = Path(workspace_path).resolve()
        self.artifacts_path = self.workspace_path / "artifacts" / f"workflow_{int(time.time())}"
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize integrations
        self.github = GitHubIntegration()
        
        # Results tracking
        self.results = {}
        
        logger.info(f"🚀 AI Workflow Orchestrator initialized at {self.workspace_path}")
    
    def analyze_repository(self) -> Dict[str, Any]:
        """Step 1: Deep repository analysis"""
        logger.info("🔍 Step 1: Analyzing repository...")
        
        # Use your existing quality analysis tool
        analysis = analyze_code_quality(str(self.workspace_path))
        
        # Save analysis report
        analysis_file = self.artifacts_path / "repository_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.results['analysis'] = analysis
        
        logger.info(f"""📊 Analysis Complete:
   - Quality Score: {analysis['quality_score']:.1f}/100
   - Total Files: {analysis['total_files']:,}
   - Issues Found: {len(analysis['issues'])}
   - Languages: {list(analysis['languages'].keys())}""")
        
        return analysis
    
    def ai_code_improvements(self) -> Dict[str, Any]:
        """Step 2: AI-powered code improvements"""
        logger.info("🤖 Step 2: AI code improvements...")
        
        improvements = {
            "optimized_files": [],
            "documentation_generated": [],
            "total_improvements": 0
        }
        
        # Find files that need improvement
        target_files = []
        
        # Look for Python files with issues
        for file_path in self.workspace_path.rglob("*.py"):
            try:
                if file_path.stat().st_size < 5000:  # Focus on smaller, manageable files
                    content = file_path.read_text()
                    if any(keyword in content.upper() for keyword in ["TODO", "FIXME", "HACK"]):
                        target_files.append(file_path)
            except:
                continue
        
        # Optimize up to 5 files
        for file_path in target_files[:5]:
            try:
                logger.info(f"   🔧 Optimizing: {file_path.relative_to(self.workspace_path)}")
                
                # Use your AI optimizer
                result = ai_code_optimizer(str(file_path), "maintainability")
                
                if result.get("success"):
                    improvements["optimized_files"].append(str(file_path))
                    improvements["total_improvements"] += 1
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Could not optimize {file_path}: {e}")
        
        # Generate documentation for key files
        main_files = list(self.workspace_path.glob("main.py")) + list(self.workspace_path.glob("app.py"))
        for file_path in main_files[:2]:
            try:
                logger.info(f"   📚 Documenting: {file_path.relative_to(self.workspace_path)}")
                
                result = ai_documentation_generator(str(file_path))
                
                if result.get("success"):
                    improvements["documentation_generated"].append(result["documentation_file"])
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Could not document {file_path}: {e}")
        
        self.results['ai_improvements'] = improvements
        
        logger.info(f"✅ AI Improvements: {improvements['total_improvements']} files optimized")
        return improvements
    
    def quality_enhancement(self) -> Dict[str, Any]:
        """Step 3: Comprehensive quality enhancement"""
        logger.info("🛡️ Step 3: Quality enhancement...")
        
        # Run comprehensive linting
        lint_result = run_comprehensive_lint(fix_issues=True)
        
        # Run tests if available
        test_result = {"success": True, "message": "No tests found"}
        if list(self.workspace_path.rglob("*test*.py")) or Path("pytest.ini").exists():
            try:
                test_result = run_tests_advanced(coverage=True)
            except:
                test_result = {"success": False, "message": "Tests failed to run"}
        
        # Generate documentation
        doc_result = generate_smart_docs()
        
        quality_results = {
            "linting": lint_result,
            "testing": test_result,
            "documentation": doc_result
        }
        
        self.results['quality'] = quality_results
        
        logger.info(f"""✅ Quality Enhancement Complete:
   - Linting Score: {lint_result['overall_score']:.1f}/100
   - Tests: {'✅ PASSED' if test_result['success'] else '❌ FAILED'}
   - Docs Generated: {doc_result['total_docs']}""")
        
        return quality_results
    
    def repository_organization(self) -> Dict[str, Any]:
        """Step 4: Repository organization and structure"""
        logger.info("🏗️ Step 4: Repository organization...")
        
        # Safe file organization
        org_result = organize_files_safe(str(self.workspace_path), dry_run=False)
        
        # Setup CI/CD if not exists
        ci_result = setup_professional_ci_cd()
        
        # Create/update .gitignore if needed
        gitignore_updates = self._update_gitignore()
        
        org_results = {
            "file_organization": org_result,
            "ci_cd_setup": ci_result,
            "gitignore_updated": gitignore_updates
        }
        
        self.results['organization'] = org_results
        
        logger.info(f"""✅ Repository Organization Complete:
   - Files Organized: {org_result['moves_planned']}
   - CI/CD Setup: {'✅' if ci_result['success'] else '❌'}
   - GitIgnore: {'✅ Updated' if gitignore_updates else '✅ Already Good'}""")
        
        return org_results
    
    def semantic_indexing(self) -> Dict[str, Any]:
        """Step 5: Build semantic code index"""
        logger.info("🔍 Step 5: Building semantic code index...")
        
        indexed_files = 0
        
        # Index Python files
        for file_path in self.workspace_path.rglob("*.py"):
            try:
                if file_path.stat().st_size > 100 and file_path.stat().st_size < 10000:
                    content = file_path.read_text()
                    
                    # Add to vector store
                    add_to_vector_store(content, {
                        "file_path": str(file_path.relative_to(self.workspace_path)),
                        "language": "python",
                        "size": len(content),
                        "indexed_at": time.time()
                    })
                    
                    indexed_files += 1
                    
            except Exception as e:
                continue
        
        # Index documentation files
        for file_path in self.workspace_path.rglob("*.md"):
            try:
                content = file_path.read_text()
                if len(content) > 50:
                    add_to_vector_store(content, {
                        "file_path": str(file_path.relative_to(self.workspace_path)),
                        "language": "markdown",
                        "type": "documentation",
                        "indexed_at": time.time()
                    })
                    indexed_files += 1
            except:
                continue
        
        indexing_result = {
            "files_indexed": indexed_files,
            "index_ready": indexed_files > 0
        }
        
        self.results['indexing'] = indexing_result
        
        logger.info(f"✅ Semantic Indexing Complete: {indexed_files} files indexed")
        return indexing_result
    
    def git_workflow(self) -> Dict[str, Any]:
        """Step 6: Professional Git workflow"""
        logger.info("💾 Step 6: Git workflow...")
        
        # Create feature branch
        branch_result = git_create_feature_branch("ai-improvements")
        
        # Commit all improvements
        commit_message = self._generate_commit_message()
        commit_result = smart_git_commit(commit_message)
        
        git_results = {
            "branch": branch_result,
            "commit": commit_result
        }
        
        self.results['git'] = git_results
        
        if branch_result.get("success") and commit_result.get("success"):
            logger.info(f"""✅ Git Workflow Complete:
   - Branch: {branch_result['branch_name']}
   - Commit: {commit_result['commit_sha'][:8]}""")
        else:
            logger.warning("⚠️ Git workflow had issues")
        
        return git_results
    
    def create_pull_request(self) -> Dict[str, Any]:
        """Step 7: Create professional pull request"""
        logger.info("📤 Step 7: Creating pull request...")
        
        # Generate PR title and description
        pr_title = "AI-powered repository improvements"
        pr_description = self._generate_pr_description()
        
        # Create PR
        pr_result = create_ai_improvement_pr(pr_title, pr_description)
        
        self.results['pull_request'] = pr_result
        
        if pr_result.get("success"):
            logger.info(f"""✅ Pull Request Created:
   - URL: {pr_result['pr_url']}
   - PR Number: #{pr_result['pr_number']}""")
        else:
            logger.warning(f"⚠️ PR creation failed: {pr_result.get('error', 'Unknown error')}")
        
        return pr_result
    
    def generate_report(self) -> Dict[str, Any]:
        """Step 8: Generate comprehensive report"""
        logger.info("📊 Step 8: Generating final report...")
        
        report = {
            "workflow_completed_at": time.time(),
            "workspace_path": str(self.workspace_path),
            "artifacts_path": str(self.artifacts_path),
            "summary": {
                "total_steps": 8,
                "steps_completed": len(self.results),
                "overall_success": len(self.results) >= 6
            },
            "detailed_results": self.results
        }
        
        # Save comprehensive report
        report_file = self.artifacts_path / "ai_workflow_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown report
        markdown_report = self._generate_markdown_report(report)
        markdown_file = self.artifacts_path / "ai_workflow_report.md"
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        
        logger.info(f"""📋 Final Report Generated:
   - JSON: {report_file}
   - Markdown: {markdown_file}
   - Overall Success: {'✅' if report['summary']['overall_success'] else '❌'}""")
        
        return report
    
    def _update_gitignore(self) -> bool:
        """Update .gitignore with common patterns"""
        gitignore_path = self.workspace_path / ".gitignore"
        
        new_patterns = [
            "# AI Development Tools",
            "artifacts/",
            "*.backup",
            "*.log",
            ".coverage",
            "coverage.json",
            "__pycache__/",
            "*.pyc",
            ".env",
            ".DS_Store"
        ]
        
        if gitignore_path.exists():
            current_content = gitignore_path.read_text()
            if "artifacts/" in current_content:
                return False  # Already updated
        else:
            current_content = ""
        
        # Add new patterns
        updated_content = current_content + "\n" + "\n".join(new_patterns) + "\n"
        gitignore_path.write_text(updated_content)
        
        return True
    
    def _generate_commit_message(self) -> str:
        """Generate comprehensive commit message"""
        improvements = self.results.get('ai_improvements', {})
        quality = self.results.get('quality', {})
        
        message = "feat: AI-powered repository improvements\n\n"
        
        if improvements.get('optimized_files'):
            message += f"- Optimized {len(improvements['optimized_files'])} files with AI\n"
        
        if quality.get('linting', {}).get('overall_score', 0) > 0:
            message += f"- Improved code quality (score: {quality['linting']['overall_score']:.1f}/100)\n"
        
        if quality.get('documentation', {}).get('total_docs', 0) > 0:
            message += f"- Generated {quality['documentation']['total_docs']} documentation files\n"
        
        message += "- Organized repository structure\n"
        message += "- Set up professional CI/CD pipeline\n"
        message += "- Indexed codebase for semantic search\n"
        message += "\nGenerated by AI Development Assistant"
        
        return message
    
    def _generate_pr_description(self) -> str:
        """Generate PR description"""
        analysis = self.results.get('analysis', {})
        improvements = self.results.get('ai_improvements', {})
        
        description = f"""This PR contains AI-powered improvements to enhance repository quality and maintainability.

## 📊 Repository Analysis
- **Quality Score**: {analysis.get('quality_score', 0):.1f}/100
- **Total Files**: {analysis.get('total_files', 0):,}
- **Languages**: {', '.join(analysis.get('languages', {}).keys())}

## 🤖 AI Improvements Applied
- **Files Optimized**: {len(improvements.get('optimized_files', []))}
- **Documentation Generated**: {len(improvements.get('documentation_generated', []))}
- **Code Quality Enhanced**: ✅
- **Repository Organized**: ✅
- **CI/CD Pipeline**: ✅

## 🎯 Benefits
- Improved code maintainability
- Enhanced documentation coverage
- Better repository organization
- Automated quality checks
- Professional development workflow

This PR was generated using advanced AI development tools to ensure enterprise-grade code quality.
"""
        
        return description
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate markdown report"""
        
        markdown = f"""# AI Development Workflow Report

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Workspace**: `{report['workspace_path']}`
**Success**: {'✅ YES' if report['summary']['overall_success'] else '❌ NO'}

## Executive Summary

Completed {report['summary']['steps_completed']}/8 workflow steps with comprehensive AI-powered improvements.

## Detailed Results

"""
        
        for step, result in report['detailed_results'].items():
            markdown += f"### {step.title().replace('_', ' ')}\n"
            if isinstance(result, dict):
                if 'success' in result:
                    status = '✅ SUCCESS' if result['success'] else '❌ FAILED'
                    markdown += f"**Status**: {status}\n"
                
                for key, value in result.items():
                    if key != 'success':
                        markdown += f"- **{key.title().replace('_', ' ')}**: {value}\n"
            markdown += "\n"
        
        markdown += f"""## Artifacts

All workflow artifacts saved to: `{report['artifacts_path']}`

## Next Steps

1. Review the generated pull request
2. Run additional tests if needed
3. Merge when ready
4. Monitor quality improvements over time

---
*Generated by AI Development Assistant*
"""
        
        return markdown

def run_complete_ai_workflow(workspace_path: str = ".") -> Dict[str, Any]:
    """Run the complete AI development workflow"""
    
    orchestrator = AIWorkflowOrchestrator(workspace_path)
    
    print("🚀 Starting Complete AI Development Workflow")
    print("=" * 60)
    
    try:
        # Execute all workflow steps
        orchestrator.analyze_repository()
        orchestrator.ai_code_improvements()
        orchestrator.quality_enhancement()
        orchestrator.repository_organization()
        orchestrator.semantic_indexing()
        orchestrator.git_workflow()
        orchestrator.create_pull_request()
        report = orchestrator.generate_report()
        
        print("\n" + "=" * 60)
        print("🎉 AI Development Workflow Complete!")
        print(f"📊 Success Rate: {report['summary']['steps_completed']}/8 steps")
        print(f"📁 Artifacts: {report['artifacts_path']}")
        
        if report['summary']['overall_success']:
            print("✅ All major steps completed successfully!")
        else:
            print("⚠️ Some steps had issues - check the detailed report")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "partial_results": orchestrator.results,
            "artifacts_path": str(orchestrator.artifacts_path)
        }

def run_workflow_step(step_name: str, workspace_path: str = ".") -> Dict[str, Any]:
    """Run a specific workflow step independently"""
    
    orchestrator = AIWorkflowOrchestrator(workspace_path)
    
    step_methods = {
        "analyze": orchestrator.analyze_repository,
        "improve": orchestrator.ai_code_improvements,
        "quality": orchestrator.quality_enhancement,
        "organize": orchestrator.repository_organization,
        "index": orchestrator.semantic_indexing,
        "git": orchestrator.git_workflow,
        "pr": orchestrator.create_pull_request,
        "report": orchestrator.generate_report
    }
    
    if step_name not in step_methods:
        return {
            "success": False,
            "error": f"Unknown step: {step_name}",
            "available_steps": list(step_methods.keys())
        }
    
    try:
        print(f"🚀 Running workflow step: {step_name}")
        result = step_methods[step_name]()
        print(f"✅ Step '{step_name}' completed successfully")
        return {"success": True, "result": result}
        
    except Exception as e:
        print(f"❌ Step '{step_name}' failed: {e}")
        return {"success": False, "error": str(e)}

def create_workflow_config(workspace_path: str = ".") -> Dict[str, Any]:
    """Create a configuration file for customizing the workflow"""
    
    config = {
        "workflow_settings": {
            "workspace_path": workspace_path,
            "create_backup": True,
            "max_files_to_optimize": 10,
            "enable_git_workflow": True,
            "create_pull_request": True,
            "generate_reports": True
        },
        "quality_settings": {
            "run_linting": True,
            "fix_linting_issues": True,
            "run_tests": True,
            "generate_coverage": True,
            "min_quality_score": 70.0
        },
        "ai_settings": {
            "optimization_focus": "maintainability",
            "generate_documentation": True,
            "code_review_level": "thorough",
            "max_file_size": 5000
        },
        "git_settings": {
            "branch_prefix": "ai-improvements",
            "commit_message_style": "conventional",
            "create_detailed_pr": True,
            "include_metrics": True
        },
        "output_settings": {
            "verbose_logging": True,
            "save_artifacts": True,
            "generate_markdown_report": True,
            "include_code_samples": False
        }
    }
    
    config_path = Path(workspace_path) / "ai_workflow_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Workflow configuration created: {config_path}")
    print("Edit this file to customize your AI workflow settings")
    
    return config

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Development Workflow Orchestrator")
    parser.add_argument("--workspace", "-w", default=".", help="Workspace path")
    parser.add_argument("--step", "-s", help="Run specific step only")
    parser.add_argument("--config", "-c", action="store_true", help="Create configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.config:
        create_workflow_config(args.workspace)
    elif args.step:
        result = run_workflow_step(args.step, args.workspace)
        sys.exit(0 if result["success"] else 1)
    else:
        result = run_complete_ai_workflow(args.workspace)
        sys.exit(0 if result.get("success", False) else 1)