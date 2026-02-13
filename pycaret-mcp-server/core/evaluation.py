"""
โมดูลสำหรับการประเมินและวิเคราะห์โมเดล
สร้างกราฟและ metrics เพื่อประเมินประสิทธิภาพ

Week 08 Assignment - PyCaret MCP Server
"""

import logging
from typing import Dict, Any, List
from .execution import execute_pycaret_code

logger = logging.getLogger(__name__)


def evaluate_model_safe(model, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely evaluate a PyCaret model
    
    Args:
        model: Trained model object
        context: Execution context
        
    Returns:
        Dictionary with evaluation results
    """
    code = """
from pycaret.classification import evaluate_model

# Evaluate model
evaluate_model(model)
"""
    
    eval_context = context.copy()
    eval_context["model"] = model
    
    return execute_pycaret_code(code, eval_context)


def plot_model_safe(model, plot_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate model plots
    
    Args:
        model: Trained model object
        plot_type: Type of plot (e.g., 'auc', 'confusion_matrix', 'feature')
        context: Execution context
        
    Returns:
        Dictionary with plot results
    """
    code = f"""
from pycaret.classification import plot_model

# Generate plot
plot_model(model, plot='{plot_type}', save=True)
"""
    
    plot_context = context.copy()
    plot_context["model"] = model
    
    return execute_pycaret_code(code, plot_context)


def get_model_metrics(model, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get model performance metrics
    
    Args:
        model: Trained model object
        context: Execution context
        
    Returns:
        Dictionary with metrics
    """
    code = """
from pycaret.classification import pull

# Get metrics
metrics = pull()
metrics_dict = metrics.to_dict() if hasattr(metrics, 'to_dict') else {}
"""
    
    metrics_context = context.copy()
    metrics_context["model"] = model
    
    result = execute_pycaret_code(code, metrics_context)
    
    if result["success"]:
        result["metrics"] = result["context"].get("metrics_dict", {})
    
    return result


def analyze_model(model, context: Dict[str, Any], 
                  plots: List[str] = None) -> Dict[str, Any]:
    """
    Comprehensive model analysis
    
    Args:
        model: Trained model object
        context: Execution context
        plots: List of plot types to generate
        
    Returns:
        Dictionary with analysis results
    """
    if plots is None:
        plots = ['auc', 'confusion_matrix', 'feature']
    
    results = {
        "success": True,
        "metrics": {},
        "plots": {},
        "errors": []
    }
    
    # Get metrics
    metrics_result = get_model_metrics(model, context)
    if metrics_result["success"]:
        results["metrics"] = metrics_result.get("metrics", {})
    else:
        results["errors"].append(f"Metrics error: {metrics_result.get('error')}")
    
    # Generate plots
    for plot_type in plots:
        plot_result = plot_model_safe(model, plot_type, context)
        if plot_result["success"]:
            results["plots"][plot_type] = "Generated successfully"
        else:
            results["errors"].append(f"Plot '{plot_type}' error: {plot_result.get('error')}")
    
    results["success"] = len(results["errors"]) == 0
    
    return results
