"""relay_client.py - Anthropic-shaped client backed by Chat Completions.

Our relay speaks the OpenAI protocol, while the lessons are written against
the Anthropic SDK. This shim exposes the same interface:

    client = RelayClient(base_url=os.getenv("ANTHROPIC_BASE_URL"))
    response = client.messages.create(model=..., system=..., messages=..., tools=..., max_tokens=...)

and translates Anthropic-flavored requests/responses to Chat Completions
and back. Nothing else in the lessons needs to change.
"""

import json
import os
from types import SimpleNamespace

import httpx2 as httpx


def _block_get(block, key, default=None):
    """Read a key from either a dict block or a SimpleNamespace block."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _to_wire(system, messages):
    """Anthropic message history -> Chat Completions message list."""
    wire = []
    if system:
        wire.append({"role": "system", "content": system})
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            wire.append({"role": message["role"], "content": content})
        elif message["role"] == "assistant":
            # Blocks -> one assistant message with optional tool_calls
            texts, tool_calls = [], []
            for block in content:
                if _block_get(block, "type") == "text":
                    texts.append(_block_get(block, "text", ""))
                elif _block_get(block, "type") == "tool_use":
                    tool_calls.append({
                        "id": _block_get(block, "id"),
                        "type": "function",
                        "function": {
                            "name": _block_get(block, "name"),
                            "arguments": json.dumps(_block_get(block, "input", {}),
                                                    ensure_ascii=False),
                        },
                    })
            entry = {"role": "assistant", "content": "".join(texts) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            wire.append(entry)
        else:
            # user message carrying tool_result blocks -> role: tool entries
            for block in content:
                if _block_get(block, "type") == "tool_result":
                    wire.append({
                        "role": "tool",
                        "tool_call_id": _block_get(block, "tool_use_id"),
                        "content": _block_get(block, "content", ""),
                    })
    return wire


def _to_blocks(body):
    """Chat Completions response -> Anthropic-style content blocks."""
    raw = body["choices"][0]["message"]
    blocks = []
    if raw.get("content"):
        blocks.append(SimpleNamespace(type="text", text=raw["content"]))
    for call in raw.get("tool_calls") or []:
        blocks.append(SimpleNamespace(
            type="tool_use",
            id=call["id"],
            name=call["function"]["name"],
            input=json.loads(call["function"]["arguments"]),
        ))
    return SimpleNamespace(content=blocks)


class _Messages:
    def __init__(self, url, api_key, timeout):
        self._url = url
        self._api_key = api_key
        self._timeout = timeout

    def create(self, *, model, messages, max_tokens, system=None, tools=None):
        payload = {
            "model": model,
            "messages": _to_wire(system, messages),
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [{
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            } for tool in tools]
        response = httpx.post(
            self._url,
            headers={"Authorization": "Bearer " + self._api_key},
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return _to_blocks(response.json())


class RelayClient:
    """Drop-in replacement for anthropic.Anthropic in these lessons."""

    def __init__(self, base_url=None, api_key=None, timeout=120.0, **_ignored):
        base = (base_url or os.environ["ANTHROPIC_BASE_URL"]).rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1")
                      else "/v1/chat/completions")
        self.messages = _Messages(
            url,
            api_key or os.environ["ANTHROPIC_API_KEY"],
            timeout,
        )
