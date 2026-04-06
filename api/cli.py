"""Command-line interface for EduQuery.
Allows users to run EduQuery commands from the terminal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from loguru import logger


def main() -> None:
    # Entry point for CLI commands
    
    logger.info(" EduQuery - AI-powered video tutorial assistant")
    logger.info("=" * 50)

    # Check if command is provided
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1]

    # Show version
    if command == "version":
        from eduquery import __version__
        print(f"EduQuery version {__version__}")

    # Show help
    elif command == "help":
        print_usage()

    # Unknown command
    else:
        logger.error(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


def print_usage() -> None:
    # Displays usage instructions

    usage = """
Usage: EduQuery <command> [options]

Commands:
    version     Show version information
    help        Show this help message
"""
    print(usage)


# Run CLI when file is executed directly
if __name__ == "__main__":
    main()