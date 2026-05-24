from langchain_ollama.chat_models import ChatOllama
from dotenv import load_dotenv
import os
import json
import subprocess
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool  # <-- Import the tool decorator

# Your project directory
WORKSPACE = "E:\\SrootAI\\pathy"

SERVERS = {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        WORKSPACE
      ]
    }
}

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ──────────────────────────────────────────────────────────────────────────────
#  THE MISSING SUPERPOWER: Add a Bash Execution Tool
# ──────────────────────────────────────────────────────────────────────────────

@tool
async def run_bash_command(command: str):
    """
    Runs a bash/shell command and returns the output. 
    Use this to run python scripts (e.g., 'python train.py').
    The command will run in the workspace directory.
    """
    try:
        # Run the command in your project directory with a 15-minute timeout
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

async def main():
    llm = ChatOllama(
        base_url="https://e685-35-227-50-23.ngrok-free.app",
        model="glm-4.7-flash:latest",
        think = False
    )
    
    # 1. Get MCP Filesystem tools
    client = MultiServerMCPClient(SERVERS)
    mcp_tools = await client.get_tools()
    
    # 2. Combine MCP tools with our custom Bash tool
    all_tools = mcp_tools + [run_bash_command]

    # 3. Bind ALL tools to the LLM
    llm_with_tools = llm.bind_tools(all_tools)

    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""
        messages = state["messages"]
        result = await llm_with_tools.ainvoke(messages)
        return {"messages": [result]}

    # 4. Pass ALL tools to the ToolNode
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