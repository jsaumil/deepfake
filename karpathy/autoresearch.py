"""
autoresearch.py
===============
Autonomous research loop for deepfake detection.

Architecture:
- Outer Python while-loop drives the experiment cycle (run forever until Ctrl+C)
- Each iteration: LLM agent gets ONE experiment task → runs it → reads results → decides keep/discard
- State (best_auc, experiment history) is managed in Python, not inside LLM context
- Context window is trimmed each iteration so it never explodes
- Clean stop: press Ctrl+C at any time
"""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama.chat_models import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ── Config ────────────────────────────────────────────────────────────────────

WORKSPACE   = Path(r"F:\projects\DeepFake\karpathy")
TRAIN_FILE  = WORKSPACE / "train.py"
BACKUP_FILE = WORKSPACE / "train.py.backup"
RESULTS_FILE = WORKSPACE / "results.json"
LOG_FILE    = WORKSPACE / "experiment_log.md"

OLLAMA_BASE_URL = "https://8754-35-197-128-152.ngrok-free.app"  # update if ngrok changes
MODEL_NAME      = "qwen2.5:14b"

SERVERS = {
    "filesystem": {
        "transport": "stdio",
        "command":   "npx",
        "args":      ["-y", "@modelcontextprotocol/server-filesystem", str(WORKSPACE)],
    }
}

# ── Graceful stop flag ────────────────────────────────────────────────────────

_stop_requested = False

def _handle_sigint(sig, frame):
    global _stop_requested
    print("\n\n[STOP] Ctrl+C received — will stop after current experiment finishes.")
    _stop_requested = True

signal.signal(signal.SIGINT, _handle_sigint)

# ── Tools ─────────────────────────────────────────────────────────────────────

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


@tool
def read_file(path: str) -> str:
    """Read a file relative to the workspace. Returns its text content."""
    full = WORKSPACE / path
    try:
        return full.read_text(encoding="utf-8")
    except Exception as e:
        return f"ERROR reading {path}: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """
    Write content to a file relative to the workspace.
    Always overwrites. Use this to modify train.py or create experiment_log.md.
    """
    full = WORKSPACE / path
    try:
        full.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR writing {path}: {e}"


ALL_TOOLS = [run_command, read_file, write_file]

# ── Agent graph (single-experiment runner) ────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(llm_with_tools):
    async def llm_node(state: AgentState):
        result = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [result]}

    tool_node = ToolNode(ALL_TOOLS)

    g = StateGraph(AgentState)
    g.add_node("llm",   llm_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", tools_condition)
    g.add_edge("tools", "llm")
    return g.compile()


# ── Helpers ───────────────────────────────────────────────────────────────────

def backup_train():
    """Save current train.py as backup before each experiment."""
    shutil.copy2(TRAIN_FILE, BACKUP_FILE)


def restore_train():
    """Restore train.py from backup (discard failed experiment)."""
    if BACKUP_FILE.exists():
        shutil.copy2(BACKUP_FILE, TRAIN_FILE)
        print("  [RESTORE] train.py restored from backup.")


def read_results() -> float | None:
    """Return val_auc from results.json, or None if missing/invalid."""
    try:
        data = json.loads(RESULTS_FILE.read_text())
        return float(data["val_auc"])
    except Exception:
        return None


def append_log(text: str):
    """Append text to experiment_log.md."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── System prompt (injected fresh each experiment) ────────────────────────────

SYSTEM_PROMPT = """You are an autonomous deepfake-detection researcher.

TOOLS AVAILABLE:
- run_command(command)  : run any shell command in the workspace
- read_file(path)       : read a file (relative to workspace)
- write_file(path, content) : write/overwrite a file

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


def make_experiment_prompt(
    experiment_number: int,
    best_auc: float | None,
    history: list[dict],
    current_train_py: str,
) -> str:
    hist_text = ""
    if history:
        lines = []
        for h in history[-5:]:  # only last 5 to keep context short
            kept = "✓ KEPT" if h["kept"] else "✗ DISCARDED"
            lines.append(f"  Exp {h['exp']:02d} | {h['change']} | val_auc={h['auc']:.4f} | {kept}")
        hist_text = "RECENT EXPERIMENTS:\n" + "\n".join(lines) + "\n\n"

    best_text = f"CURRENT BEST val_auc: {best_auc:.4f}\n\n" if best_auc is not None else "CURRENT BEST val_auc: none yet (first run)\n\n"

    return f"""EXPERIMENT #{experiment_number:02d}
{timestamp()}

{best_text}{hist_text}CURRENT train.py:
```python
{current_train_py}
```

Your task:
1. Decide ONE improvement to try (based on the history above — don't repeat failed ones).
2. Edit train.py using write_file.
3. Run: python train.py  (takes ~10 min, be patient)
4. If it fails, fix and retry.
5. Read results.json.
6. End with: EXPERIMENT_DONE: val_auc=<number>
"""


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main():
    global _stop_requested
    load_dotenv()

    print("=" * 70)
    print("  AutoResearch — DeepFake Detector")
    print("  Press Ctrl+C at any time to stop gracefully.")
    print("=" * 70)

    # Init log
    if not LOG_FILE.exists():
        LOG_FILE.write_text(f"# AutoResearch Log\nStarted: {timestamp()}\n\n", encoding="utf-8")

    # LLM
    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL_NAME, think=False)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    graph = build_graph(llm_with_tools)

    # State
    best_auc: float | None = None
    experiment_number = 1
    history: list[dict] = []

    # Try to load existing best from log
    if RESULTS_FILE.exists():
        existing = read_results()
        if existing is not None:
            best_auc = existing
            print(f"  [INIT] Loaded existing results.json → best_auc = {best_auc:.4f}")

    while not _stop_requested:
        print("\n" + "=" * 70)
        print(f"  EXPERIMENT #{experiment_number:02d}  |  Best so far: {best_auc}")
        print("=" * 70)

        # Backup current train.py before any changes
        backup_train()

        # Read current train.py to show the agent
        current_train = TRAIN_FILE.read_text(encoding="utf-8")

        # Build the prompt for this experiment
        user_prompt = make_experiment_prompt(
            experiment_number, best_auc, history, current_train
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # Run the agent for this experiment (it will loop until it calls no more tools)
        print(f"\n  [AGENT] Starting experiment #{experiment_number}...")
        final_state = await graph.ainvoke({"messages": messages})

        # Extract the last AI message
        last_msg = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                last_msg = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        print(f"\n  [AGENT] Final message:\n{last_msg[:500]}...")

        # Parse the EXPERIMENT_DONE line
        reported_auc: float | None = None
        for line in last_msg.splitlines():
            line = line.strip()
            if line.startswith("EXPERIMENT_DONE:"):
                try:
                    reported_auc = float(line.split("val_auc=")[1].strip())
                except Exception:
                    pass
                break

        # Fallback: try reading results.json directly
        if reported_auc is None:
            reported_auc = read_results()
            if reported_auc is not None:
                print(f"  [FALLBACK] Read val_auc={reported_auc:.4f} directly from results.json")

        # Evaluate: keep or discard?
        if reported_auc is None:
            print("  [RESULT] Training did not produce a valid result. DISCARDING.")
            restore_train()
            kept = False
            auc_display = 0.0
        else:
            auc_display = reported_auc
            if best_auc is None or reported_auc > best_auc:
                print(f"  [RESULT] ✓ IMPROVEMENT! {best_auc} → {reported_auc:.4f}. KEEPING.")
                best_auc = reported_auc
                kept = True
                # Save a "best" snapshot
                shutil.copy2(TRAIN_FILE, WORKSPACE / "train_best.py")
            else:
                print(f"  [RESULT] ✗ No improvement ({reported_auc:.4f} ≤ {best_auc:.4f}). DISCARDING.")
                restore_train()
                kept = False

        # Figure out what change was made (diff the backup)
        try:
            diff = subprocess.run(
                f'git diff --no-index train.py.backup train.py',
                shell=True, capture_output=True, text=True, cwd=str(WORKSPACE)
            )
            change_summary = diff.stdout[:300].strip() if diff.stdout else "no diff available"
        except Exception:
            change_summary = "unknown"

        # Record in history
        history.append({
            "exp":    experiment_number,
            "auc":    auc_display,
            "kept":   kept,
            "change": change_summary[:80],
        })

        # Append to log
        status = "KEPT ✓" if kept else "DISCARDED ✗"
        append_log(
            f"\n## Experiment {experiment_number:02d} — {timestamp()}\n"
            f"- **Status**: {status}\n"
            f"- **val_auc**: {auc_display:.4f}\n"
            f"- **Best so far**: {best_auc}\n"
            f"- **Agent summary** (last 500 chars):\n```\n{last_msg[-500:]}\n```\n"
        )

        print(f"\n  [LOG] Experiment #{experiment_number} logged.")
        experiment_number += 1

        if _stop_requested:
            break

        # Small pause before next round
        print("\n  [WAIT] Pausing 5 seconds before next experiment...")
        await asyncio.sleep(5)

    print("\n" + "=" * 70)
    print(f"  AutoResearch stopped after {experiment_number - 1} experiments.")
    print(f"  Best val_auc achieved: {best_auc}")
    print(f"  Best model saved to: train_best.py")
    print(f"  Full log: {LOG_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
