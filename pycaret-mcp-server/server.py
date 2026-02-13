"""
PyCaret MCP Server - เซิร์ฟเวอร์ MCP สำหรับ PyCaret

พัฒนาขึ้นเพื่อให้ AI Agent ใช้งาน PyCaret ผ่าน Model Context Protocol
ดัดแปลงจาก pandas-mcp-server สำหรับ Week 08 Assignment
"""

import logging
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

from core.config import LOG_FILE, LOG_LEVEL
from core.execution import execute_pycaret_code, setup_experiment, compare_models_safe
from core.evaluation import evaluate_model_safe, plot_model_safe, analyze_model

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global context to store experiment state
GLOBAL_CONTEXT: Dict[str, Any] = {}

# Create MCP server
app = Server("pycaret-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available PyCaret MCP tools"""
    return [
        Tool(
            name="setup_experiment",
            description="Initialize a PyCaret classification or regression experiment",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_source": {
                        "type": "string",
                        "description": "Path to CSV file or PyCaret dataset name"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target column name"
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["classification", "regression"],
                        "description": "Type of ML task"
                    },
                    "session_id": {
                        "type": "integer",
                        "description": "Random seed for reproducibility",
                        "default": 123
                    }
                },
                "required": ["data_source", "target", "task_type"]
            }
        ),
        Tool(
            name="compare_models",
            description="Compare multiple ML models and return the best one",
            inputSchema={
                "type": "object",
                "properties": {
                    "n_select": {
                        "type": "integer",
                        "description": "Number of top models to select",
                        "default": 1
                    }
                }
            }
        ),
        Tool(
            name="run_pycaret_code",
            description="Execute custom PyCaret code securely",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code using PyCaret"
                    }
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="evaluate_model",
            description="Evaluate model with plots and metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "plot_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of plot types (e.g., 'auc', 'confusion_matrix', 'feature')",
                        "default": ["auc", "confusion_matrix"]
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    global GLOBAL_CONTEXT
    
    try:
        if name == "setup_experiment":
            # Load data
            data_source = arguments.get("data_source")
            target = arguments.get("target")
            task_type = arguments.get("task_type", "classification")
            session_id = arguments.get("session_id", 123)
            
            # Load data from CSV or PyCaret dataset
            if data_source.endswith('.csv'):
                data = pd.read_csv(data_source)
            else:
                # Try loading from PyCaret datasets
                from pycaret.datasets import get_data
                data = get_data(data_source)
            
            # Setup experiment
            result = setup_experiment(
                data=data,
                target=target,
                task_type=task_type,
                session_id=session_id
            )
            
            if result["success"]:
                GLOBAL_CONTEXT.update(result["context"])
                GLOBAL_CONTEXT["task_type"] = task_type
                return [TextContent(
                    type="text",
                    text=f"✅ Experiment setup successful!\n\nTask: {task_type}\nTarget: {target}\nSession ID: {session_id}\n\nOutput:\n{result['output']}"
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"❌ Setup failed: {result['error']}\n\nTraceback:\n{result.get('traceback', '')}"
                )]
        
        elif name == "compare_models":
            if not GLOBAL_CONTEXT:
                return [TextContent(
                    type="text",
                    text="❌ No experiment setup found. Please run setup_experiment first."
                )]
            
            result = compare_models_safe(GLOBAL_CONTEXT)
            
            if result["success"]:
                GLOBAL_CONTEXT.update(result["context"])
                
                # Format model results
                model_results = result.get("model_results")
                if model_results is not None:
                    results_str = model_results.to_string() if hasattr(model_results, 'to_string') else str(model_results)
                else:
                    results_str = "Results not available"
                
                return [TextContent(
                    type="text",
                    text=f"✅ Model comparison complete!\n\nBest Model: {result.get('best_model')}\n\nResults:\n{results_str}\n\nOutput:\n{result['output']}"
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"❌ Comparison failed: {result['error']}\n\nTraceback:\n{result.get('traceback', '')}"
                )]
        
        elif name == "run_pycaret_code":
            code = arguments.get("code")
            result = execute_pycaret_code(code, GLOBAL_CONTEXT)
            
            if result["success"]:
                GLOBAL_CONTEXT.update(result["context"])
                return [TextContent(
                    type="text",
                    text=f"✅ Code executed successfully!\n\nOutput:\n{result['output']}"
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"❌ Execution failed: {result['error']}\n\nTraceback:\n{result.get('traceback', '')}"
                )]
        
        elif name == "evaluate_model":
            if "best_model" not in GLOBAL_CONTEXT:
                return [TextContent(
                    type="text",
                    text="❌ No model found. Please run compare_models first."
                )]
            
            plot_types = arguments.get("plot_types", ["auc", "confusion_matrix"])
            model = GLOBAL_CONTEXT["best_model"]
            
            result = analyze_model(model, GLOBAL_CONTEXT, plot_types)
            
            if result["success"]:
                metrics_str = "\n".join([f"- {k}: {v}" for k, v in result.get("metrics", {}).items()])
                plots_str = "\n".join([f"- {k}: {v}" for k, v in result.get("plots", {}).items()])
                
                return [TextContent(
                    type="text",
                    text=f"✅ Model evaluation complete!\n\nMetrics:\n{metrics_str}\n\nPlots:\n{plots_str}"
                )]
            else:
                errors_str = "\n".join(result.get("errors", []))
                return [TextContent(
                    type="text",
                    text=f"⚠️ Evaluation completed with errors:\n{errors_str}"
                )]
        
        else:
            return [TextContent(
                type="text",
                text=f"❌ Unknown tool: {name}"
            )]
            
    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"❌ Error: {str(e)}"
        )]


async def main():
    """Run the MCP server"""
    logger.info("Starting PyCaret MCP Server...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
