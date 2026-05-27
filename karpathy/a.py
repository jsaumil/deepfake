from asyncio import subprocess
from langchain_ollama.chat_models import ChatOllama
from dotenv import load_dotenv
import os
import json
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

WORKSPACE = "E:\\SrootAI\\pathy"  # <--- CHANGE THIS to a real folder path!

SERVERS = {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",  # Uses npx to run the node package
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "E:\\SrootAI\\pathy"  # <--- CHANGE THIS to a real folder path!
      ]
    }
}

@tool
def run_command(command: str) -> str:
    """
    Run any shell command in the workspace directory.
    Use for: running python scripts, installing packages, reading file output.
    Returns stdout + stderr combined. Times out after 15 minutes.
    """
    print(f"\n  [CMD] >>> {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(WORKSPACE),
            timeout=900,
        )
        output = f"EXIT_CODE={result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        # Trim to keep context manageable — last 4000 chars is enough
        if len(output) > 4000:
            output = "...[truncated]...\n" + output[-4000:]
        print(f"  [CMD] exit={result.returncode}, output_len={len(output)}")
        return output
    except subprocess.TimeoutExpired:
        return "EXIT_CODE=1\nERROR: Command timed out after 900 seconds."
    except Exception as e:
        return f"EXIT_CODE=1\nERROR: {e}"


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def fix_tool_call_args(tool_calls):
    """Stringify any dict/list values in tool call arguments."""
    if not tool_calls:
        return tool_calls, False

    needs_fix = any(
        isinstance(value, (dict, list))
        for tc in tool_calls
        for value in tc["args"].values()
    )

    if not needs_fix:
        return tool_calls, False

    fixed = []
    for tc in tool_calls:
        fixed_args = {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            for key, value in tc["args"].items()
        }
        fixed.append({
            "name": tc["name"],
            "args": fixed_args,
            "id": tc["id"],
        })
    return fixed, True

SYSTEM_PROMPT = """You are an autonomous deepfake-detection researcher.

use tools and mcp tools to run experiments, analyze results, and iteratively improve the model in train.py to achieve the best val_auc you can.

WORKSPACE FILES:
- train.py          ← the ONLY file you may modify
- prepare.py        ← DO NOT TOUCH
- prepare1.py       ← DO NOT TOUCH
- results.json      ← written by train.py after training
- experiment_log.md ← your research journal

RULES:
1. Modify ONLY train.py.
2. Make EXACTLY ONE meaningful change per experiment.
3. After editing train.py, run: python train.py
4. Training takes ~10 minutes. Wait for it to finish.
5. If training fails (EXIT_CODE≠0), fix the bug and retry — never give up.
6. After success, read results.json and report val_auc.
7. You MUST end your response with the line:
   EXPERIMENT_DONE: val_auc=<number>
   Example: EXPERIMENT_DONE: val_auc=0.847

IMPORTANT — train.py currently uses:
  from prepare1 import build_dataloaders
  train_loader, val_loader, test_loader = build_dataloaders(
      dataset_root="./dataset_split",
      batch_size=4,
      seq_len=8,
  )
The transform argument must NOT be passed (prepare1 handles it internally).
"""


async def main(prompt):
    # llm = ChatOllama(
    #     base_url="https://squeamish-remix-cahoots.ngrok-free.dev",
    #     model="glm-4.7-flash:latest",
    #     think = False
    # )
    llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
)
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    tools = tools + [run_command]  # Add our custom bash command tool to the list of MCP tools

    # tools_map = {tool.name: tool for tool in tools}
    # print("Available tools:", list(tools_map.keys()))

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""
        messages = state["messages"]
        result = await llm_with_tools.ainvoke(messages)

        # ── Fix: stringify any dict/list arguments in tool calls ──
        if hasattr(result, 'tool_calls') and result.tool_calls:
            fixed_calls, did_fix = fix_tool_call_args(result.tool_calls)
            if did_fix:
                result = AIMessage(
                    content=result.content,
                    tool_calls=fixed_calls,
                    id=result.id,
                    usage_metadata=getattr(result, 'usage_metadata', None),
                    response_metadata=getattr(result, 'response_metadata', {}),
                )
        return {"messages": [result]}

    tool_node = ToolNode(tools) if tools else None

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    graph.add_edge("chat_node", END)

    checkpointer = InMemorySaver()

    chatbot = graph.compile(checkpointer=checkpointer)

    config1 = {"configurable":{"thread_id":"1"}}
        
    result = await chatbot.ainvoke({"messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
            ]},
            config=config1
            )
    print("Final result:", result["messages"][-1].content)
    return result

if __name__ == "__main__":
    load_dotenv()
    import asyncio
    asyncio.run(main())
