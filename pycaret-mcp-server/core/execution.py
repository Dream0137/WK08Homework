"""
โมดูลสำหรับการรันโค้ด PyCaret อย่างปลอดภัย
ตรวจสอบและป้องกันโค้ดที่เป็นอันตราย

ส่วนหนึ่งของ PyCaret MCP Server - Week 08 Assignment
"""

import logging
import sys
from io import StringIO
from typing import Dict, Any
import traceback
from .config import BLOCKED_FUNCTIONS, MAX_EXECUTION_TIME

logger = logging.getLogger(__name__)


def validate_code(code: str) -> tuple[bool, str]:
    """
    ตรวจสอบโค้ดเพื่อหาช่องโหว่ด้านความปลอดภัย
    
    Args:
        code: โค้ด Python ที่ต้องการตรวจสอบ
        
    Returns:
        Tuple (ผ่านการตรวจสอบหรือไม่, ข้อความ error)
    """
    # ตรวจสอบฟังก์ชันที่ถูกบล็อก
    for blocked in BLOCKED_FUNCTIONS:
        if blocked in code:
            return False, f"พบฟังก์ชันที่ไม่อนุญาต '{blocked}' ในโค้ด"
    
    return True, ""


def execute_pycaret_code(code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute PyCaret code in a controlled environment
    
    Args:
        code: Python code to execute
        context: Optional context dictionary with variables
        
    Returns:
        Dictionary with execution results
    """
    # Validate code
    is_valid, error_msg = validate_code(code)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
            "output": ""
        }
    
    # Prepare execution context
    exec_context = context.copy() if context else {}
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # Execute code
        exec(code, exec_context)
        
        # Get output
        output = captured_output.getvalue()
        
        return {
            "success": True,
            "output": output,
            "context": exec_context,
            "error": None
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Code execution failed: {error_trace}")
        
        return {
            "success": False,
            "error": str(e),
            "traceback": error_trace,
            "output": captured_output.getvalue()
        }
        
    finally:
        # Restore stdout
        sys.stdout = old_stdout


def setup_experiment(data, target: str, task_type: str = "classification", 
                     session_id: int = 123, **kwargs) -> Dict[str, Any]:
    """
    Setup a PyCaret experiment
    
    Args:
        data: DataFrame or dataset
        target: Target column name
        task_type: 'classification' or 'regression'
        session_id: Random seed for reproducibility
        **kwargs: Additional setup parameters
        
    Returns:
        Dictionary with setup results
    """
    code = f"""
from pycaret.{task_type} import *
import pandas as pd

# Setup experiment
s = setup(data, target='{target}', session_id={session_id}, verbose=False, **{kwargs})
experiment = s
"""
    
    context = {"data": data}
    result = execute_pycaret_code(code, context)
    
    if result["success"]:
        result["experiment"] = result["context"].get("experiment")
    
    return result


def compare_models_safe(experiment_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely run compare_models
    
    Args:
        experiment_context: Context from setup_experiment
        
    Returns:
        Dictionary with comparison results
    """
    code = """
from pycaret.classification import compare_models
import pandas as pd

# Compare models
best_model = compare_models()
model_results = pull()
"""
    
    result = execute_pycaret_code(code, experiment_context)
    
    if result["success"]:
        result["best_model"] = result["context"].get("best_model")
        result["model_results"] = result["context"].get("model_results")
    
    return result
