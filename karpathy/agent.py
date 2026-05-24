import ssl
# Bypass SSL verification for Ngrok tunnels
ssl._create_default_https_context = ssl._create_unverified_context

from langchain_ollama.chat_models import ChatOllama
from dotenv import load_dotenv
import os
import json
import subprocess
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

# Your project directory where train.py, prepare.py, and program.md live
WORKSPACE = "/home/Digant_Parmar/deepfake/karpathy"

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ──────────────────────────────────────────────────────────────────────────────
#  CUSTOM TOOLS (Replaces MCP Filesystem - No Node.js needed!)
# ──────────────────────────────────────────────────────────────────────────────

@tool
def read_file(filepath: str) -> str:
    """Reads the content of a file. Use this to read program.md, train.py, or results.json."""
    try:
        # If the path is not absolute, assume it's in the workspace
        if not os.path.isabs(filepath):
            filepath = os.path.join(WORKSPACE, filepath)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # Truncate very large files so we don't overload the LLM context
        if len(content) > 8000:
            return content[:8000] + "\n... [File Truncated]"
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(filepath: str, content: str) -> str:
    """Writes content to a file. Use this to modify train.py or update experiment_log.md."""
    try:
        if not os.path.isabs(filepath):
            filepath = os.path.join(WORKSPACE, filepath)
            
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool
def run_bash_command(command: str) -> str:
    """
    Runs a bash/shell command and returns the output. 
    Use this to run python scripts (e.g., 'python train.py').
    The command will run in the workspace directory.
    """
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=WORKSPACE, 
            timeout=900 # 15 minute timeout for 10-min training runs
        )
        output = f"EXIT CODE: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        
        # Truncate output if it's too long for the LLM context window
        if len(output) > 3000:
            output = output[-3000:] + "\n... [Output Truncated]"
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 900 seconds."

# List of all tools the agent can use
all_tools = [read_file, write_file, run_bash_command]

# ──────────────────────────────────────────────────────────────────────────────
#  LANGGRAPH AGENT SETUP
# ──────────────────────────────────────────────────────────────────────────────

async def main():
    llm = ChatOllama(
        base_url="https://e685-35-227-50-23.ngrok-free.app",
        model="glm-4.7-flash:latest",
        think=False
    )
    
    # Bind our custom tools to the LLM
    llm_with_tools = llm.bind_tools(all_tools)

    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""
        messages = state["messages"]
        result = await llm_with_tools.ainvoke(messages)
        return {"messages": [result]}

    # Pass ALL tools to the ToolNode
    tool_node = ToolNode(all_tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    graph.add_edge("chat_node", END)

    checkpointer = InMemorySaver()
    chatbot = graph.compile(checkpointer=checkpointer)

    # ──────────────────────────────────────────────────────────────────────────────
    #  KICK OFF THE AUTONOMOUS RESEARCH LOOP
    # ──────────────────────────────────────────────────────────────────────────────
    
    config1 = {"configurable": {"thread_id": "1"}}
    
    # This is the initial prompt that starts the whole process!
    initial_prompt = """
    You are an autonomous AI deepfake detection researcher. 
    Read the file `program.md` to understand your mission, rules, and how to run experiments.
    Once you understand the rules, kick off your first experiment!
    """
    
    result = await chatbot.ainvoke(
        {"messages": [HumanMessage(content=initial_prompt)]}, 
        config=config1
    )
    
    print("Final result:", result["messages"][-1].content)
    return result

if __name__ == "__main__":
    load_dotenv()
    import asyncio
    asyncio.run(main())