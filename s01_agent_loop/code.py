#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while True:
        response = LLM(messages, tools)
        if response contains no tool_use:
            break
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Later chapters add policy,
hooks, and lifecycle controls around it.

Usage:
    pip install httpx2 python-dotenv
    ANTHROPIC_API_KEY=... python s01_agent_loop/code.py
"""

import os
import json
import subprocess
from types import SimpleNamespace

import httpx2 as httpx
from dotenv import load_dotenv

load_dotenv(override=True)

# The relay speaks OpenAI Chat Completions, not the Anthropic protocol.
BASE_URL = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
API_URL = BASE_URL + ("/chat/completions" if BASE_URL.endswith("/v1") else "/v1/chat/completions")
MODEL = os.environ["MODEL_ID"]

ENVIRONMENT_PROMPT = (
    "Windows: the bash tool runs through cmd.exe; use cmd.exe syntax, not Unix "
    "Bash or PowerShell syntax"
    if os.name == "nt"
    else "Unix-like: the bash tool runs the system shell"
)
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use bash to solve tasks. Act, don't explain."
)

# -- Tool definition: just bash --
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


# -- Tool execution --
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# -- The core pattern: a while loop that calls tools until the model stops --
def llm_response(messages: list):
    """One Chat Completions call, adapted to this lesson's content blocks."""
    wire = [{"role": "system", "content": SYSTEM}]
    for message in messages:
        if "_openai_message" in message:          # assistant history (raw)
            wire.append(message["_openai_message"])
        elif isinstance(message["content"], str): # user text
            wire.append({"role": "user", "content": message["content"]})
        else:                                     # tool results
            for result in message["content"]:
                wire.append({"role": "tool", "tool_call_id": result["tool_use_id"],
                             "content": result["content"]})

    response = httpx.post(API_URL, headers={
        "Authorization": "Bearer " + os.environ["ANTHROPIC_API_KEY"],
    }, json={
        "model": MODEL, "messages": wire, "max_tokens": 8000,
        "tools": [{"type": "function", "function": {
            "name": tool["name"], "description": tool["description"],
            "parameters": tool["input_schema"],
        }} for tool in TOOLS],
    }, timeout=120.0)
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]
    blocks = []
    if raw.get("content"):
        blocks.append(SimpleNamespace(type="text", text=raw["content"]))
    for tool in raw.get("tool_calls") or []:
        arguments = json.loads(tool["function"]["arguments"])
        blocks.append(SimpleNamespace(type="tool_use", id=tool["id"],
                                      name="bash", input=arguments))
    # Keep the raw assistant message (incl. reasoning_content) for the next turn.
    assistant = {"role": "assistant"}
    for key in ("content", "tool_calls", "reasoning_content"):
        if key in raw:
            assistant[key] = raw[key]
    return SimpleNamespace(content=blocks, raw_message=assistant)


def agent_loop(messages: list):
    while True:
        response = llm_response(messages)

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content,
                         "_openai_message": response.raw_message})

        # If the model didn't call a tool, we're done
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            return

        # Execute each tool call, collect results
        results = []
        for block in tool_calls:
            print(f"\033[33m$ {block.input['command']}\033[0m")
            output = run_bash(block.input["command"])
            print(output[:200])
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

        # Feed tool results back, loop continues
        messages.append({"role": "user", "content": results})


# -- Entry point --
if __name__ == "__main__":
    print("s01: Agent Loop")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            query = input("s01 >> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not query.strip():
            continue
        if query.strip().lower() in ("q", "exit"):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)

        # Print the model's final text response
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
