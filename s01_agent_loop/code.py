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
    pip install anthropic python-dotenv
    ANTHROPIC_API_KEY=... python s01_agent_loop/code.py
"""

import os
import json
import re
import subprocess
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

import anthropic
try:
    import httpx2 as httpx
except ImportError:
    import httpx

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(
    base_url=os.getenv("ANTHROPIC_BASE_URL"), timeout=120.0, max_retries=0,
)
MODEL = os.environ["MODEL_ID"]
API_PROTOCOL = os.getenv("API_PROTOCOL", "anthropic").strip().lower()
if API_PROTOCOL not in ("anthropic", "openai"):
    raise SystemExit("API_PROTOCOL must be anthropic or openai.")

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
def openai_response(messages):
    """Adapt Chat Completions to the content blocks used by this lesson."""
    base = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
    url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    wire = [{"role": "system", "content": SYSTEM}]
    for message in messages:
        if "_openai_message" in message:
            wire.append(message["_openai_message"])
        elif isinstance(message["content"], str):
            wire.append(message)
        elif message["role"] == "user":
            for result in message["content"]:
                wire.append({"role": "tool", "tool_call_id": result["tool_use_id"],
                             "content": result["content"]})
        else:
            raise ValueError("Missing Chat Completions assistant history.")
    request = httpx.Request("POST", url)
    try:
        with httpx.Client(timeout=120.0) as transport:
            response = transport.post(url, headers={
                "Authorization": "Bearer " + os.environ["ANTHROPIC_API_KEY"],
            }, json={
                "model": MODEL, "messages": wire, "max_tokens": 8000,
                "tools": [{"type": "function", "function": {
                    "name": tool["name"], "description": tool["description"],
                    "parameters": tool["input_schema"],
                }} for tool in TOOLS],
            })
    except httpx.TimeoutException as error:
        raise anthropic.APITimeoutError(request=request) from error
    except httpx.RequestError as error:
        raise anthropic.APIConnectionError(request=request) from error
    try:
        body = response.json()
    except ValueError:
        body = response.text
    if response.is_error:
        raise anthropic.APIStatusError("Chat Completions request failed",
                                       response=response, body=body)
    try:
        choice = body["choices"][0]
        raw = choice["message"]
        if choice.get("finish_reason") in ("length", "content_filter"):
            raise ValueError("Incomplete model response; no tools executed.")
        # Preserve reasoning_content for providers that require it on tool turns.
        assistant = {key: value for key, value in raw.items()
                     if key in ("content", "tool_calls", "reasoning_content")}
        assistant["role"] = "assistant"
        blocks = []
        if raw.get("content"):
            blocks.append(SimpleNamespace(type="text", text=raw["content"]))
        for tool in raw.get("tool_calls") or []:
            function = tool["function"]
            arguments = json.loads(function["arguments"])
            if function["name"] != "bash" or not isinstance(arguments.get("command"), str):
                raise ValueError("Invalid tool call; no tools executed.")
            blocks.append(SimpleNamespace(type="tool_use", id=tool["id"],
                                          name=function["name"], input=arguments))
        if not blocks:
            raise ValueError("The provider returned neither text nor tool calls.")
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
        raise anthropic.APIError(
            "Invalid or incomplete Chat Completions response. " + str(error),
            request=response.request, body=None,
        ) from error
    return SimpleNamespace(content=blocks, raw_message=assistant)


def agent_loop(messages: list):
    while True:
        if API_PROTOCOL == "openai":
            response = openai_response(messages)
        else:
            response = client.messages.create(
                model=MODEL, system=SYSTEM, messages=messages,
                tools=TOOLS, max_tokens=8000,
            )

        # Append assistant turn
        assistant_turn = {"role": "assistant", "content": response.content}
        if API_PROTOCOL == "openai":
            assistant_turn["_openai_message"] = response.raw_message
        messages.append(assistant_turn)

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


def safe_diagnostic(value):
    """Redact configured credentials and terminal control characters."""
    text = str(value)
    for name, secret in os.environ.items():
        if any(word in name.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET")):
            if len(secret) >= 4:
                text = text.replace(secret, "[redacted]")
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", text)
    return " ".join("".join(c if c.isprintable() else " " for c in text).split())[:500]


def public_url(value):
    """Keep the endpoint but remove URL credentials, queries and fragments."""
    url = urlsplit(str(value))
    host = url.netloc.rsplit("@", 1)[-1]
    return safe_diagnostic(urlunsplit((url.scheme, host, url.path, "", "")))


def api_error_message(error):
    """Show the server's error message, without dumping its response body."""
    if isinstance(error, anthropic.APITimeoutError):
        return "API request timed out. Check the connection or try again."
    if isinstance(error, anthropic.APIConnectionError):
        return (
            "Cannot connect to the API. Check ANTHROPIC_BASE_URL, network access "
            "and HTTP_PROXY / HTTPS_PROXY settings."
        )
    status = getattr(error, "status_code", None)
    if status is None:
        return safe_diagnostic(error.message)
    hints = {
        400: "Check MODEL_ID and whether the endpoint supports the configured API protocol and tool use.",
        401: "Check ANTHROPIC_API_KEY for this provider.",
        403: "Check API permissions and whether this model is enabled for your account.",
        404: "Check MODEL_ID, API_PROTOCOL and the provider's base URL.",
        429: "The provider rate limit or quota was reached. Check your account and try later.",
    }
    message = f"API request failed (HTTP {status}). " + hints.get(
        status, "Check the provider's service status and API configuration."
    )
    response = getattr(error, "response", None)
    if response is not None:
        request = response.request
        message += f"\nRequest: {request.method} {public_url(request.url)}"
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        detail = body.get("error", body)
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("code") or detail.get("type")
    elif isinstance(body, str) and not body.lstrip().startswith("<"):
        detail = body
    else:
        detail = None
    if detail:
        message += "\nServer: " + safe_diagnostic(detail)
    return message


# -- Entry point --
if __name__ == "__main__":
    print("s01: Agent Loop")
    print(f"API protocol: {API_PROTOCOL}")
    print(f"Model: {safe_diagnostic(MODEL)}")
    print("API base URL: " + public_url(os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"))
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
        turn_start = len(history)
        history.append({"role": "user", "content": query})
        try:
            agent_loop(history)
        except anthropic.APIError as error:
            print(api_error_message(error))
            # Drop an unanswered prompt, but preserve completed tool exchanges.
            if len(history) == turn_start + 1:
                del history[turn_start:]
            continue
        # Print the model's final text response
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
