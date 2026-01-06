from typing import Dict, Any, List
from src.tools.python_repl import handle_python_repl_tool
from src.tools.bash_tool import handle_bash_tool
import os
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

tool_list = [
    {
        "toolSpec": {
            "name": "python_repl_tool",
            "description": "Use this to execute python code and do data analysis or calculation. If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The python code to execute to do further analysis or calculation."
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "bash_tool",
            "description": "Use this to execute bash command and do necessary operations.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "The bash command to be executed."
                        }
                    },
                    "required": ["cmd"]
                }
            }
        }
    }
]

coder_tool_config = {
    "tools": tool_list
}

def process_coder_tool(tool) -> str:
    """Process a tool invocation
    
    Args:
        tool_name: Name of the tool to invoke
        tool_input: Input parameters for the tool
        
    Returns:
        Result of the tool invocation as a string
    """
    
    tool_name, tool_input = tool["name"], tool["input"]
    
    if tool_name == "python_repl_tool":
        results = handle_python_repl_tool(code=tool_input["code"])
    elif tool_name == "bash_tool":
        results = handle_bash_tool(cmd=tool_input["cmd"])
    else:
        print(f"Unknown tool: {tool_name}")
        return None
        
    tool_result = {
        "toolUseId": tool['toolUseId'],
        "content": [{"json": {"text": str(results)}}]
    }
    
    return {"role": "user", "content": [{"toolResult": tool_result}]}