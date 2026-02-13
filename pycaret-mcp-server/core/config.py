"""
การตั้งค่าสำหรับ PyCaret MCP Server
กำหนดค่าความปลอดภัย, logging, และข้อจำกัดการทำงาน

Week 08 Assignment - Data Science Toolbox
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pycaret_mcp.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Execution limits
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "300"))  # 5 minutes
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "2048"))  # 2GB

# Security settings
ALLOWED_IMPORTS = [
    "pycaret",
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "seaborn",
]

BLOCKED_FUNCTIONS = [
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
]

# PyCaret settings
DEFAULT_SESSION_ID = 123
DEFAULT_TRAIN_SIZE = 0.7
SUPPORTED_TASK_TYPES = ["classification", "regression"]
