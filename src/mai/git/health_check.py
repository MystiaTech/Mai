"""
Health check and validation system for Mai's self-improvement code.

Provides comprehensive testing, validation, and regression detection
to ensure code changes are safe before merging.
"""

import os
import sys
import time
import importlib
import subprocess
import logging
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Callable
from pathlib import Path

from ..core import MaiError, ConfigurationError


class HealthCheckError(MaiError):
    """Raised when health check operations fail."""

    pass


class HealthChecker:
    """
    Comprehensive health validation for Mai's code improvements.

    Provides syntax checking, functionality testing, performance
    validation, and Mai-specific behavior validation.
    """

    def __init__(self, project_path: str = ".", timeout: int = 60):
        """
        Initialize health checker.

        Args:
            project_path: Path to project directory
            timeout: Timeout for health check operations
        """
        self.project_path = Path(project_path).resolve()
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Health check categories
        self.categories = {
            "basic": self._basic_health_checks,
            "extended": self._extended_health_checks,
            "mai-specific": self._mai_specific_tests,
            "performance": self._performance_tests,
        }

        # Configure retry and timeout policies
        self.max_retries = 3
        self.retry_delay = 2

        # Performance baseline tracking
        self.performance_baseline = {}
        self._load_performance_baseline()

        self.logger.info(f"Health checker initialized for {self.project_path}")

    def run_basic_health_checks(self) -> Dict[str, Any]:
        """
        Execute essential system validation tests.

        Returns:
            Detailed results with suggestions for any issues found
        """
        results = {
            "category": "basic",
            "timestamp": datetime.now().isoformat(),
            "checks": [],
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "overall_status": "unknown",
        }

        checks = [
            ("Python Syntax", self._check_python_syntax),
            ("Import Validation", self._check_imports),
            ("Configuration Files", self._check_configuration),
            ("Core Functionality", self._check_core_functionality),
            ("Dependencies", self._check_dependencies),
        ]

        for check_name, check_func in checks:
            self.logger.info(f"Running basic check: {check_name}")

            try:
                check_result = check_func()
                results["checks"].append(
                    {
                        "name": check_name,
                        "status": check_result["status"],
                        "message": check_result["message"],
                        "details": check_result.get("details", {}),
                        "suggestions": check_result.get("suggestions", []),
                    }
                )

                if check_result["status"] == "pass":
                    results["passed"] += 1
                elif check_result["status"] == "warning":
                    results["warnings"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                self.logger.error(f"Basic check '{check_name}' failed: {e}")
                results["checks"].append(
                    {
                        "name": check_name,
                        "status": "error",
                        "message": f"Check failed with error: {e}",
                        "details": {"traceback": traceback.format_exc()},
                        "suggestions": ["Check system configuration and permissions"],
                    }
                )
                results["failed"] += 1

        # Determine overall status
        if results["failed"] == 0:
            results["overall_status"] = "pass" if results["warnings"] == 0 else "warning"
        else:
            results["overall_status"] = "fail"

        return results

    def run_mai_specific_tests(self) -> Dict[str, Any]:
        """
        Run Mai-specific validation tests.

        Returns:
            Test results for Mai-specific functionality
        """
        results = {
            "category": "mai-specific",
            "timestamp": datetime.now().isoformat(),
            "checks": [],
            "passed": 0,
            "failed": 0,
            "overall_status": "unknown",
        }

        checks = [
            ("Model Interface", self._check_model_interface),
            ("Resource Monitoring", self._check_resource_monitoring),
            ("Git Workflows", self._check_git_workflows),
            ("Context Compression", self._check_context_compression),
            ("Core Components", self._check_core_components),
        ]

        for check_name, check_func in checks:
            self.logger.info(f"Running Mai-specific check: {check_name}")

            try:
                check_result = check_func()
                results["checks"].append(
                    {
                        "name": check_name,
                        "status": check_result["status"],
                        "message": check_result["message"],
                        "details": check_result.get("details", {}),
                        "suggestions": check_result.get("suggestions", []),
                    }
                )

                if check_result["status"] == "pass":
                    results["passed"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                self.logger.error(f"Mai-specific check '{check_name}' failed: {e}")
                results["checks"].append(
                    {
                        "name": check_name,
                        "status": "error",
                        "message": f"Check failed with error: {e}",
                        "details": {"traceback": traceback.format_exc()},
                        "suggestions": ["Check Mai component configuration"],
                    }
                )
                results["failed"] += 1

        results["overall_status"] = "pass" if results["failed"] == 0 else "fail"
        return results

    def run_performance_tests(self, duration: int = 30) -> Dict[str, Any]:
        """
        Execute performance benchmarks.

        Args:
            duration: Duration of performance test in seconds

        Returns:
            Performance metrics with trend analysis
        """
        results = {
            "category": "performance",
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "metrics": {},
            "baseline_comparison": {},
            "trend_analysis": {},
            "overall_status": "unknown",
        }

        self.logger.info(f"Running performance tests for {duration} seconds")

        try:
            # Test different performance aspects
            performance_checks = [
                ("Import Speed", self._test_import_speed),
                ("Memory Usage", self._test_memory_usage),
                ("Model Client", self._test_model_client_performance),
                ("Git Operations", self._test_git_performance),
            ]

            for check_name, check_func in performance_checks:
                start_time = time.time()

                try:
                    metrics = check_func(duration)
                    end_time = time.time()

                    results["metrics"][check_name] = {
                        "data": metrics,
                        "test_duration": end_time - start_time,
                        "status": "success",
                    }

                except Exception as e:
                    results["metrics"][check_name] = {"error": str(e), "status": "failed"}

            # Compare with baseline
            results["baseline_comparison"] = self._compare_with_baseline(results["metrics"])

            # Analyze trends
            results["trend_analysis"] = self._analyze_performance_trends(results["metrics"])

            # Determine overall status
            failed_checks = sum(
                1 for m in results["metrics"].values() if m.get("status") == "failed"
            )
            results["overall_status"] = "pass" if failed_checks == 0 else "fail"

            # Update baseline if tests passed
            if results["overall_status"] == "pass":
                self._update_performance_baseline(results["metrics"])

        except Exception as e:
            self.logger.error(f"Performance tests failed: {e}")
            results["overall_status"] = "error"
            results["error"] = str(e)

        return results

    def validate_improvement(self, branch_name: str, base_commit: str) -> Dict[str, Any]:
        """
        Compare branch against baseline with full validation.

        Args:
            branch_name: Name of improvement branch
            base_commit: Base commit to compare against

        Returns:
            Validation report with recommendations
        """
        results = {
            "branch_name": branch_name,
            "base_commit": base_commit,
            "timestamp": datetime.now().isoformat(),
            "validation_results": {},
            "performance_comparison": {},
            "recommendations": [],
            "can_merge": False,
            "overall_status": "unknown",
        }

        self.logger.info(f"Validating improvement branch {branch_name} against {base_commit[:8]}")

        try:
            # Import git functionality for branch comparison
            from .workflow import StagingWorkflow

            workflow = StagingWorkflow(str(self.project_path))

            # Switch to improvement branch temporarily
            original_branch = workflow.repo.active_branch.name
            switch_result = workflow.switch_to_branch(branch_name)

            if not switch_result["success"]:
                return {
                    **results,
                    "overall_status": "error",
                    "error": f"Cannot switch to branch {branch_name}: {switch_result['message']}",
                }

            # Run tests on improvement branch
            improvement_tests = self.run_all_tests("basic")

            # Switch back to base branch
            workflow.switch_to_branch(base_commit)

            # Run tests on base branch
            base_tests = self.run_all_tests("basic")

            # Compare results
            comparison = self._compare_test_results(improvement_tests, base_tests)
            results["validation_results"] = comparison

            # Run performance comparison
            improvement_perf = self.run_performance_tests(duration=15)
            workflow.switch_to_branch(branch_name)
            branch_perf = improvement_perf

            workflow.switch_to_branch(base_commit)
            base_perf = self.run_performance_tests(duration=15)

            results["performance_comparison"] = self._compare_performance(branch_perf, base_perf)

            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(
                comparison, results["performance_comparison"]
            )

            # Determine if safe to merge
            results["can_merge"] = self._can_merge_safely(results)
            results["overall_status"] = "pass" if results["can_merge"] else "fail"

            # Switch back to original branch
            workflow.switch_to_branch(original_branch)

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            results["overall_status"] = "error"
            results["error"] = str(e)

        return results

    def create_test_suite(self, test_type: str = "basic") -> Dict[str, Any]:
        """
        Generate test suite for specific scenarios.

        Args:
            test_type: Type of test suite to generate

        Returns:
            Test suite configuration
        """
        test_suites = {
            "basic": {
                "description": "Essential validation tests",
                "tests": [
                    "Python syntax validation",
                    "Import checking",
                    "Configuration validation",
                    "Basic functionality tests",
                ],
                "duration_estimate": "2-5 minutes",
                "requirements": ["Python 3.10+", "Project dependencies"],
            },
            "extended": {
                "description": "Comprehensive validation including integration",
                "tests": [
                    "All basic tests",
                    "Integration tests",
                    "Error handling validation",
                    "Edge case testing",
                ],
                "duration_estimate": "5-15 minutes",
                "requirements": ["All basic requirements", "Test environment setup"],
            },
            "mai-specific": {
                "description": "Mai-specific behavior and functionality",
                "tests": [
                    "Model interface testing",
                    "Resource monitoring validation",
                    "Git workflow testing",
                    "Context compression testing",
                ],
                "duration_estimate": "3-8 minutes",
                "requirements": ["Ollama running (optional)", "Git repository"],
            },
            "performance": {
                "description": "Performance benchmarking and regression detection",
                "tests": [
                    "Import speed testing",
                    "Memory usage analysis",
                    "Model client performance",
                    "Git operation benchmarks",
                ],
                "duration_estimate": "1-5 minutes",
                "requirements": ["Stable system load", "Consistent environment"],
            },
        }

        if test_type not in test_suites:
            return {
                "error": f"Unknown test type: {test_type}",
                "available_types": list(test_suites.keys()),
            }

        suite = test_suites[test_type]
        suite["test_type"] = test_type
        suite["generated_at"] = datetime.now().isoformat()

        return suite

    def run_all_tests(self, category: str = "basic") -> Dict[str, Any]:
        """
        Run all tests in a specific category.

        Args:
            category: Test category to run

        Returns:
            Aggregated test results
        """
        if category not in self.categories:
            return {
                "error": f"Unknown test category: {category}",
                "available_categories": list(self.categories.keys()),
            }

        return self.categories[category]()

    # Private implementation methods

    def _basic_health_checks(self) -> Dict[str, Any]:
        """Placeholder - implemented as run_basic_health_checks."""
        pass

    def _extended_health_checks(self) -> Dict[str, Any]:
        """Extended health checks with integration testing."""
        # TODO: Implement extended tests
        return {"status": "not_implemented", "message": "Extended checks not yet implemented"}

    def _mai_specific_tests(self) -> Dict[str, Any]:
        """Placeholder - implemented as run_mai_specific_tests."""
        pass

    def _performance_tests(self) -> Dict[str, Any]:
        """Placeholder - implemented as run_performance_tests."""
        pass

    def _check_python_syntax(self) -> Dict[str, Any]:
        """Check Python syntax in all Python files."""
        python_files = list(self.project_path.rglob("*.py"))
        syntax_errors = []

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    compile(f.read(), str(file_path), "exec")
            except SyntaxError as e:
                syntax_errors.append(
                    {
                        "file": str(file_path.relative_to(self.project_path)),
                        "line": e.lineno,
                        "error": str(e),
                    }
                )

        if syntax_errors:
            return {
                "status": "fail",
                "message": f"Syntax errors found in {len(syntax_errors)} files",
                "details": {"errors": syntax_errors},
                "suggestions": ["Fix syntax errors before proceeding"],
            }

        return {
            "status": "pass",
            "message": f"All {len(python_files)} Python files have valid syntax",
        }

    def _check_imports(self) -> Dict[str, Any]:
        """Check that imports work correctly."""
        import_errors = []

        # Try importing main modules
        modules_to_test = ["src.mai.model", "src.mai.git", "src.mai.core"]

        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                import_errors.append({"module": module_name, "error": str(e)})

        if import_errors:
            return {
                "status": "fail",
                "message": f"Import errors in {len(import_errors)} modules",
                "details": {"errors": import_errors},
                "suggestions": ["Check dependencies and module structure"],
            }

        return {"status": "pass", "message": "All core modules import successfully"}

    def _check_configuration(self) -> Dict[str, Any]:
        """Validate configuration files."""
        config_files = ["pyproject.toml", ".env", "README.md"]
        issues = []

        for config_file in config_files:
            file_path = self.project_path / config_file
            if file_path.exists():
                try:
                    if config_file.endswith(".toml"):
                        import toml

                        with open(file_path, "r") as f:
                            toml.load(f)
                except Exception as e:
                    issues.append({"file": config_file, "error": str(e)})

        if issues:
            return {
                "status": "warning",
                "message": f"Configuration issues in {len(issues)} files",
                "details": {"issues": issues},
                "suggestions": ["Review and fix configuration errors"],
            }

        return {"status": "pass", "message": "Configuration files are valid"}

    def _check_core_functionality(self) -> Dict[str, Any]:
        """Check that core functionality works."""
        try:
            # Test basic imports
            from src.mai.core import MaiError, ConfigurationError
            from src.mai.model.ollama_client import OllamaClient
            from src.mai.git.workflow import StagingWorkflow
            from src.mai.git.committer import AutoCommitter

            # Test basic functionality
            client = OllamaClient()
            workflow = StagingWorkflow(str(self.project_path))
            committer = AutoCommitter(str(self.project_path))

            return {
                "status": "pass",
                "message": "Core functionality initializes correctly",
                "details": {
                    "ollama_client": "ok",
                    "staging_workflow": "ok",
                    "auto_committer": "ok",
                },
            }

        except Exception as e:
            return {
                "status": "fail",
                "message": f"Core functionality test failed: {e}",
                "suggestions": ["Check module imports and dependencies"],
            }

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check that required dependencies are available."""
        required_packages = ["ollama", "psutil", "GitPython", "tiktoken"]
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.lower().replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            return {
                "status": "fail",
                "message": f"Missing dependencies: {', '.join(missing_packages)}",
                "details": {"missing": missing_packages},
                "suggestions": [f"Install with: pip install {' '.join(missing_packages)}"],
            }

        return {"status": "pass", "message": "All dependencies are available"}

    def _check_model_interface(self) -> Dict[str, Any]:
        """Check model interface functionality."""
        try:
            from src.mai.model.ollama_client import OllamaClient

            client = OllamaClient()

            # Test basic functionality
            models = client.list_models()

            return {
                "status": "pass",
                "message": f"Model interface working, found {len(models)} models",
                "details": {"model_count": len(models)},
            }

        except Exception as e:
            return {
                "status": "warning",
                "message": f"Model interface test failed: {e}",
                "suggestions": ["Ensure Ollama is running if model detection is needed"],
            }

    def _check_resource_monitoring(self) -> Dict[str, Any]:
        """Check resource monitoring functionality."""
        try:
            import psutil

            # Test basic monitoring
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            return {
                "status": "pass",
                "message": "Resource monitoring working correctly",
                "details": {
                    "cpu_usage": cpu_percent,
                    "memory_available": memory.available,
                    "memory_total": memory.total,
                },
            }

        except Exception as e:
            return {
                "status": "fail",
                "message": f"Resource monitoring test failed: {e}",
                "suggestions": ["Check psutil installation"],
            }

    def _check_git_workflows(self) -> Dict[str, Any]:
        """Check git workflow functionality."""
        try:
            from src.mai.git.workflow import StagingWorkflow
            from src.mai.git.committer import AutoCommitter

            workflow = StagingWorkflow(str(self.project_path))
            committer = AutoCommitter(str(self.project_path))

            # Test basic operations
            branches = workflow.get_active_staging_branches()
            history = committer.get_commit_history(limit=1)

            return {
                "status": "pass",
                "message": "Git workflows working correctly",
                "details": {"staging_branches": len(branches), "commit_history": len(history)},
            }

        except Exception as e:
            return {
                "status": "fail",
                "message": f"Git workflow test failed: {e}",
                "suggestions": ["Check Git repository state"],
            }

    def _check_context_compression(self) -> Dict[str, Any]:
        """Check context compression functionality."""
        try:
            from src.mai.model.compression import ContextCompressor

            compressor = ContextCompressor()

            # Test basic functionality
            test_context = "This is a test context for compression."
            compressed = compressor.compress_context(test_context)

            return {
                "status": "pass",
                "message": "Context compression working correctly",
                "details": {
                    "original_length": len(test_context),
                    "compressed_length": len(compressed),
                },
            }

        except Exception as e:
            return {
                "status": "fail",
                "message": f"Context compression test failed: {e}",
                "suggestions": ["Check compression module implementation"],
            }

    def _check_core_components(self) -> Dict[str, Any]:
        """Check core component availability."""
        core_components = [
            "src.mai.core.exceptions",
            "src.mai.core.config",
            "src.mai.model.ollama_client",
            "src.mai.model.compression",
        ]

        working_components = []
        failed_components = []

        for component in core_components:
            try:
                importlib.import_module(component)
                working_components.append(component)
            except Exception:
                failed_components.append(component)

        if failed_components:
            return {
                "status": "fail",
                "message": f"Core components failed: {len(failed_components)}",
                "details": {"working": working_components, "failed": failed_components},
            }

        return {
            "status": "pass",
            "message": f"All {len(working_components)} core components working",
        }

    def _test_import_speed(self, duration: int) -> Dict[str, float]:
        """Test module import speed."""
        start_time = time.time()
        end_time = start_time + duration
        import_count = 0

        modules_to_test = ["src.mai.core", "src.mai.model", "src.mai.git"]

        while time.time() < end_time:
            for module in modules_to_test:
                try:
                    importlib.import_module(module)
                    import_count += 1
                except ImportError:
                    pass
            time.sleep(0.1)  # Small delay

        actual_duration = time.time() - start_time
        imports_per_second = import_count / actual_duration

        return {
            "imports_per_second": imports_per_second,
            "total_imports": import_count,
            "duration": actual_duration,
        }

    def _test_memory_usage(self, duration: int) -> Dict[str, float]:
        """Test memory usage patterns."""
        try:
            import psutil

            memory_samples = []
            start_time = time.time()

            while time.time() - start_time < duration:
                memory = psutil.virtual_memory()
                memory_samples.append(memory.percent)
                time.sleep(1)

            return {
                "average_memory_percent": sum(memory_samples) / len(memory_samples),
                "max_memory_percent": max(memory_samples),
                "min_memory_percent": min(memory_samples),
                "sample_count": len(memory_samples),
            }

        except Exception as e:
            return {"error": str(e)}

    def _test_model_client_performance(self, duration: int) -> Dict[str, Any]:
        """Test model client performance."""
        try:
            from src.mai.model.ollama_client import OllamaClient

            client = OllamaClient()

            start_time = time.time()
            response_times = []

            while time.time() - start_time < duration:
                request_start = time.time()
                models = client.list_models()
                request_end = time.time()

                response_times.append(request_end - request_start)
                time.sleep(2)  # Delay between requests

            return {
                "average_response_time": sum(response_times) / len(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "request_count": len(response_times),
            }

        except Exception as e:
            return {"error": str(e)}

    def _test_git_performance(self, duration: int) -> Dict[str, float]:
        """Test git operation performance."""
        try:
            from src.mai.git.workflow import StagingWorkflow

            workflow = StagingWorkflow(str(self.project_path))

            start_time = time.time()
            operation_times = []

            while time.time() - start_time < duration:
                op_start = time.time()
                branches = workflow.get_active_staging_branches()
                op_end = time.time()

                operation_times.append(op_end - op_start)
                time.sleep(1)

            return {
                "average_operation_time": sum(operation_times) / len(operation_times),
                "min_operation_time": min(operation_times),
                "max_operation_time": max(operation_times),
                "operation_count": len(operation_times),
            }

        except Exception as e:
            return {"error": str(e)}

    def _load_performance_baseline(self) -> None:
        """Load performance baseline from storage."""
        # TODO: Implement baseline persistence
        self.performance_baseline = {}

    def _compare_with_baseline(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        comparison = {}

        for metric_name, metric_data in current_metrics.items():
            if metric_data.get("status") == "success" and metric_name in self.performance_baseline:
                baseline = self.performance_baseline[metric_name]
                current = metric_data["data"]

                # Simple comparison logic - can be enhanced
                if "imports_per_second" in current:
                    improvement = (
                        (current["imports_per_second"] - baseline.get("imports_per_second", 0))
                        / baseline.get("imports_per_second", 1)
                        * 100
                    )
                    comparison[metric_name] = {
                        "improvement_percent": improvement,
                        "baseline": baseline,
                        "current": current,
                        "status": "improved" if improvement > 0 else "degraded",
                    }

        return comparison

    def _analyze_performance_trends(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends from metrics."""
        trends = {}

        for metric_name, metric_data in metrics.items():
            if metric_data.get("status") == "success":
                data = metric_data["data"]

                # Simple trend analysis
                if "response_times" in str(data):
                    trends[metric_name] = {"trend": "stable", "confidence": "medium"}
                else:
                    trends[metric_name] = {"trend": "unknown", "confidence": "low"}

        return trends

    def _update_performance_baseline(self, new_metrics: Dict[str, Any]) -> None:
        """Update performance baseline with new metrics."""
        for metric_name, metric_data in new_metrics.items():
            if metric_data.get("status") == "success":
                self.performance_baseline[metric_name] = metric_data["data"]

        # TODO: Persist baseline to storage

    def _compare_test_results(
        self, improvement: Dict[str, Any], base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare test results between improvement and base."""
        comparison = {
            "improvement_better": 0,
            "base_better": 0,
            "equal": 0,
            "detail_comparison": {},
        }

        # Compare basic checks
        improvement_checks = improvement.get("checks", [])
        base_checks = base.get("checks", [])

        for imp_check in improvement_checks:
            base_check = next((bc for bc in base_checks if bc["name"] == imp_check["name"]), None)

            if base_check:
                imp_status = imp_check["status"]
                base_status = base_check["status"]

                if imp_status == "pass" and base_status != "pass":
                    comparison["improvement_better"] += 1
                    comparison_result = "improvement_better"
                elif base_status == "pass" and imp_status != "pass":
                    comparison["base_better"] += 1
                    comparison_result = "base_better"
                else:
                    comparison["equal"] += 1
                    comparison_result = "equal"

                comparison["detail_comparison"][imp_check["name"]] = {
                    "result": comparison_result,
                    "improvement": imp_status,
                    "base": base_status,
                }

        return comparison

    def _compare_performance(
        self, improvement_perf: Dict[str, Any], base_perf: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare performance between improvement and base."""
        comparison = {"overall": "unknown", "metrics_comparison": {}}

        imp_metrics = improvement_perf.get("metrics", {})
        base_metrics = base_perf.get("metrics", {})

        for metric_name in imp_metrics:
            if metric_name in base_metrics:
                imp_data = imp_metrics[metric_name].get("data", {})
                base_data = base_metrics[metric_name].get("data", {})

                # Simple performance comparison
                if "imports_per_second" in imp_data and "imports_per_second" in base_data:
                    if imp_data["imports_per_second"] > base_data["imports_per_second"]:
                        result = "improvement_better"
                    else:
                        result = "base_better"
                else:
                    result = "cannot_compare"

                comparison["metrics_comparison"][metric_name] = {
                    "result": result,
                    "improvement": imp_data,
                    "base": base_data,
                }

        # Determine overall comparison
        better_count = sum(
            1
            for comp in comparison["metrics_comparison"].values()
            if comp["result"] == "improvement_better"
        )
        worse_count = sum(
            1
            for comp in comparison["metrics_comparison"].values()
            if comp["result"] == "base_better"
        )

        if better_count > worse_count:
            comparison["overall"] = "improvement_better"
        elif worse_count > better_count:
            comparison["overall"] = "base_better"
        else:
            comparison["overall"] = "equal"

        return comparison

    def _generate_recommendations(
        self, validation: Dict[str, Any], performance: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation and performance."""
        recommendations = []

        # Validation-based recommendations
        failed_checks = [
            check
            for check in validation.get("detail_comparison", {}).values()
            if check["result"] == "base_better"
        ]
        if failed_checks:
            recommendations.append("Fix failing tests before merging")

        # Performance-based recommendations
        if performance.get("overall") == "base_better":
            recommendations.append("Consider performance optimizations")
        elif performance.get("overall") == "improvement_better":
            recommendations.append("Performance improvement detected")

        # Default recommendations
        if not recommendations:
            recommendations.append("Improvement looks safe to merge")

        return recommendations

    def _can_merge_safely(self, results: Dict[str, Any]) -> bool:
        """Determine if improvement can be merged safely."""
        # Check for critical failures
        validation_results = results.get("validation_results", {})

        # If any basic test failed, cannot merge
        base_better_count = validation_results.get("base_better", 0)
        if base_better_count > 0:
            return False

        # If performance is severely degraded, warn but allow
        performance_comparison = results.get("performance_comparison", {})
        if performance_comparison.get("overall") == "base_better":
            # Could add more sophisticated logic here
            pass

        return True
