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
from langchain_core.tools import tool

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
    },
    # "desktop-commander": {
    #   "transport": "stdio",
    #   "command": "npx",
    #   "args": ["-y", "@wonderwhy-er/desktop-commander@latest"]
    # },
    # "fetch": {
    #   "command": "python",
    #   "args": ["-m", "mcp_server_fetch"]
    # }
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
    print(f"[TOOL] run_bash_command called with command: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=WORKSPACE, 
            timeout=900
        )
        output = f"EXIT CODE: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        if len(output) > 3000:
            output = output[-3000:] + "\n... [Output Truncated]"
        print(f"[TOOL] run_bash_command finished. Exit code: {result.returncode}")
        print(f"[TOOL] Output (first 500 chars): {output[:500]}")
        return output
    except subprocess.TimeoutExpired:
        print("[TOOL] run_bash_command TIMED OUT after 900 seconds")
        return "ERROR: Command timed out after 900 seconds."

async def main():
    print("=" * 80)
    print("[INIT] Starting main()")
    print("=" * 80)

    # ── LLM Initialization ──
    print("[INIT] Creating ChatOllama instance...")
    llm = ChatOllama(
        base_url="https://8754-35-197-128-152.ngrok-free.app",
        model="qwen2.5:14b",
        think=False
    )
    print("[INIT] ChatOllama instance created successfully")

    # ── MCP Tools ──
    print("[INIT] Connecting to MCP servers...")
    print(f"[INIT] Server config: {json.dumps(SERVERS, indent=2)}")
    client = MultiServerMCPClient(SERVERS)
    print("[INIT] MultiServerMCPClient created, fetching tools...")
    mcp_tools = await client.get_tools()
    print(f"[INIT] MCP tools fetched. Count: {len(mcp_tools)}")
    for i, t in enumerate(mcp_tools):
        print(f"[INIT]   MCP Tool {i}: {t.name} - {t.description[:100] if t.description else 'No description'}")

    # ── Combine Tools ──
    print("[INIT] Adding custom run_bash_command tool...")
    all_tools = mcp_tools + [run_bash_command]
    print(f"[INIT] Total tools available: {len(all_tools)}")
    for i, t in enumerate(all_tools):
        print(f"[INIT]   Tool {i}: {t.name}")

    # ── Bind Tools to LLM ──
    print("[INIT] Binding tools to LLM...")
    llm_with_tools = llm.bind_tools(all_tools)
    print("[INIT] Tools bound to LLM successfully")

    # ── Chat Node ──
    async def chat_node(state: ChatState):
        print("-" * 80)
        print("[CHAT_NODE] Entered chat_node")
        messages = state["messages"]
        print(f"[CHAT_NODE] Number of messages in state: {len(messages)}")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content_preview = str(msg.content)[:200] if msg.content else "(empty)"
            print(f"[CHAT_NODE]   Message {i}: type={msg_type}, content_preview={content_preview}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"[CHAT_NODE]     Tool calls: {msg.tool_calls}")
        
        print("[CHAT_NODE] Invoking LLM with tools...")
        result = await llm_with_tools.ainvoke(messages)
        
        result_type = type(result).__name__
        result_content_preview = str(result.content)[:300] if result.content else "(empty)"
        print(f"[CHAT_NODE] LLM response received. Type: {result_type}")
        print(f"[CHAT_NODE] LLM content preview: {result_content_preview}")
        if hasattr(result, 'tool_calls') and result.tool_calls:
            print(f"[CHAT_NODE] LLM requested tool calls: {result.tool_calls}")
        else:
            print("[CHAT_NODE] LLM did NOT request any tool calls")
        
        print("[CHAT_NODE] Returning from chat_node")
        print("-" * 80)
        return {"messages": [result]}

    # ── Tool Node ──
    print("[INIT] Creating ToolNode with all tools...")
    tool_node = ToolNode(all_tools)
    print("[INIT] ToolNode created")

    # ── Build Graph ──
    print("[INIT] Building StateGraph...")
    graph = StateGraph(ChatState)
    print("[INIT] Adding nodes...")
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    print("[INIT] Adding edges...")
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    graph.add_edge("chat_node", END)
    print("[INIT] Graph structure built")

    # ── Compile Graph ──
    print("[INIT] Compiling graph with InMemorySaver checkpointer...")
    checkpointer = InMemorySaver()
    chatbot = graph.compile(checkpointer=checkpointer)
    print("[INIT] Graph compiled successfully")

    # ── Kick Off ──
    print("=" * 80)
    print("[KICKOFF] Starting autonomous research loop")
    print("=" * 80)
    
    config1 = {"configurable": {"thread_id": "1"}}
    print(f"[KICKOFF] Config: {config1}")
    
    initial_prompt = """
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
    
    print(f"[KICKOFF] Initial prompt:\n{initial_prompt.strip()}")
    print("[KICKOFF] Invoking chatbot.ainvoke...")
    
    result = await chatbot.ainvoke(
        {"messages": [HumanMessage(content=initial_prompt)]}, 
        config=config1
    )
    
    print("=" * 80)
    print("[DONE] chatbot.ainvoke completed")
    print("=" * 80)
    print(f"[DONE] Total messages in final state: {len(result['messages'])}")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        content_preview = str(msg.content)[:300] if msg.content else "(empty)"
        print(f"[DONE]   Message {i}: type={msg_type}")
        print(f"[DONE]     Content preview: {content_preview}")
    
    final_content = result["messages"][-1].content
    print(f"\n[RESULT] Final result:\n{final_content}")
    return result

if __name__ == "__main__":
    print("[BOOT] Loading .env...")
    load_dotenv()
    print("[BOOT] .env loaded")
    print("[BOOT] Running asyncio.run(main())...")
    import asyncio
    asyncio.run(main())
    print("[BOOT] Program finished.")