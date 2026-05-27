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
WORKSPACE = "F:\projects\DeepFake\karpathy"

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
        base_url="https://1491-34-125-210-229.ngrok-free.app",
        model="qwen2.5:14b",
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
    system_prompt = """
You are an autonomous AI deepfake detection researcher.

The current directory contains:

- train.py
- results.json (generated after training)
- experiment_log.md (create if missing)

Your objective is to maximize the validation AUC score reported in results.json.

Important rules:

1. You may ONLY modify train.py.
2. Do NOT modify prepare.py.
3. Do NOT modify program.md.
4. Training duration is fixed at approximately 10 minutes and must not be changed.
5. The model receives video tensors with shape:
   (B, 8, 3, 224, 224)

Workflow:

1. Inspect train.py and understand the current architecture.
2. Run the training:
      python train.py
3. If execution fails:
   - Read the error carefully.
   - Fix small coding issues automatically.
   - Install missing Python packages when required.
   - Update train.py if necessary.
   - Continue until training runs successfully.
4. Wait for training completion.
5. Read results.json.
6. Extract val_auc.
7. Record findings in experiment_log.md.

After each completed experiment:

- Analyze the result.
- Propose a hypothesis for improvement.
- Modify train.py with exactly one meaningful experiment.
- Run training again.
- Compare the new val_auc with previous results.
- Keep the better approach.
- Document every experiment in experiment_log.md.

Possible areas to explore:

- Swin Transformer dimensions
- GAN backbone dimensions
- Feature fusion strategies
- Residual connections
- Layer normalization
- Attention mechanisms
- Dropout values (0.1-0.5)
- BiLSTM vs pooling approaches
- AdamW, SGD, RMSProp
- Learning rates:
    1e-3
    5e-4
    1e-4
    5e-5

When encountering errors:

- Never stop at the first failure.
- Attempt to diagnose and repair the issue.
- Retry execution after every fix.
- Use shell commands whenever necessary.

Goal:

Continuously improve val_auc through iterative experimentation and maintain a complete experiment history in experiment_log.md.

Start by inspecting train.py and launching the first training run.
"""
    
    # This is the initial prompt that starts the whole process!
    initial_prompt = """
    do autoresearch on train.py to improve val_auc. 
    Remember to follow the workflow and rules outlined in the system prompt.
    """
    
    result = await chatbot.ainvoke(
        {"messages": [SystemMessage(content=system_prompt),HumanMessage(content=initial_prompt)]}, 
        config=config1
    )
    
    print("Final result:", result["messages"][-1].content)
    return result

if __name__ == "__main__":
    load_dotenv()
    import asyncio
    asyncio.run(main())