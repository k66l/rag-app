#!/usr/bin/env python3
"""
Test runner script for RAG Application API tests.

This script provides different ways to run the test suite:
- All tests
- Unit tests only  
- Integration tests only
- Specific test files
- With coverage reporting
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest pytest-asyncio")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG Application API tests")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "health",
                 "login", "upload", "ask", "stream"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first failure"
    )

    args = parser.parse_args()

    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]

    # Add verbose flag
    if args.verbose:
        base_cmd.append("-v")

    # Add fail-fast flag
    if args.fail_fast:
        base_cmd.append("-x")

    # Add coverage if requested
    if args.coverage:
        base_cmd.extend([
            "--cov=app",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])

    # Determine which tests to run
    if args.type == "all":
        test_path = "tests/"
        description = "All tests"
    elif args.type == "unit":
        test_path = "tests/test_*.py"
        description = "Unit tests"
    elif args.type == "integration":
        test_path = "tests/test_integration.py"
        description = "Integration tests"
    elif args.type == "health":
        test_path = "tests/test_health_check.py"
        description = "Health check tests"
    elif args.type == "login":
        test_path = "tests/test_login.py"
        description = "Login tests"
    elif args.type == "upload":
        test_path = "tests/test_upload.py"
        description = "Upload tests"
    elif args.type == "ask":
        test_path = "tests/test_ask.py"
        description = "Ask tests"
    elif args.type == "stream":
        test_path = "tests/test_ask_stream.py"
        description = "Streaming ask tests"

    # Build final command
    cmd = base_cmd + [test_path]

    # Check if test files exist
    if not Path("tests").exists():
        print("❌ Tests directory not found!")
        print("Make sure you're running this from the project root directory.")
        return 1

    # Run tests
    success = run_command(cmd, description)

    if success:
        print(f"\n🎉 {description} passed!")
        if args.coverage:
            print("\n📊 Coverage report generated in htmlcov/index.html")
        return 0
    else:
        print(f"\n💥 {description} failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
