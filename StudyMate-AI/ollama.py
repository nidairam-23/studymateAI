"""Offline assistant using a local Ollama model: free, no API key, works without internet.

Setup (once):
    1. Install Ollama from https://ollama.com (Windows installer; it runs in the background)
    2. ollama pull qwen2.5:3b          (or llama3.2:3b for a smaller download; qwen2.5:7b is smarter
                                         but needs about 8 GB of free RAM)
    3. pip install openai              (only used as a client for the local server; no key needed)

Self-contained: chat_assistant.py hands in its tool list and a function that runs a tool.
"""
from __future__ import annotations

import json
import os
import urllib.request

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
PREFERRED = ("qwen2.5", "llama3.1", "llama3.2", "qwen3", "mistral")  # families that handle tools well


def installed_models() -> list[str]:
    """Names of models downloaded in Ollama ([] when Ollama isn't running)."""
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=1.5) as resp:
            return [m["name"] for m in json.load(resp).get("models", [])]
    except Exception:
        return []


def available() -> bool:
    return bool(installed_models())


def pick_model() -> str | None:
    wanted = os.getenv("STUDYMATE_OLLAMA_MODEL")
    if wanted:
        return wanted
    models = installed_models()
    for family in PREFERRED:
        for name in models:
            if name.startswith(family):
                return name
    return models[0] if models else None


def chat_turn(messages: list[dict], tool_specs: list[dict], system_prompt: str, call_tool, max_steps: int = 6) -> str:
    from openai import OpenAI

    model = pick_model()
    if not model:
        raise RuntimeError("No Ollama model found. Install Ollama, then run:  ollama pull qwen2.5:3b")
    client = OpenAI(base_url=f"{HOST}/v1", api_key="ollama", timeout=180)
    tools = [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                                "parameters": s["input_schema"]}} for s in tool_specs]
    msgs = [{"role": "system", "content": system_prompt}] + list(messages)

    for _ in range(max_steps):
        try:
            msg = client.chat.completions.create(model=model, messages=msgs, tools=tools).choices[0].message
        except Exception as exc:
            if "tool" in str(exc).lower():  # this model can't use tools: answer in plain chat instead
                plain = client.chat.completions.create(model=model, messages=msgs).choices[0].message.content or ""
                return plain + "\n\n_(This model can't operate the app. Try `ollama pull qwen2.5:3b` for full features.)_"
            raise
        if not msg.tool_calls:
            return msg.content or ""
        msgs.append(msg)
        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = call_tool(call.function.name, args)
            msgs.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})
    return "That took too many steps. Could you try a simpler request?"