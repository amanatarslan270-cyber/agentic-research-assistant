"""A minimal, dependency-light tool-calling agent (ReAct-style loop).

Works with any OpenAI-compatible Chat Completions endpoint: OpenAI, Groq,
OpenRouter, Together, or a local Ollama / LM Studio server.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import requests

from .tools import TOOL_SCHEMAS, Toolbox

SYSTEM_PROMPT = """You are a careful research assistant.
Use the tools to find and read real papers before answering. Never invent papers,
authors or arXiv ids: cite only ids returned by the tools, as [arXiv:ID].
Save the key findings with save_note while you work. When you have enough
evidence, reply WITHOUT a tool call with a concise Markdown literature brief:
## Overview, ## Key papers (bullets with [arXiv:ID]), ## Open problems."""


class LLMClient:
    """Thin client for /chat/completions with tool calling."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    def __call__(self, messages: List[dict], tools: List[dict]) -> dict:
        r = requests.post(f"{self.base_url}/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model, "messages": messages, "tools": tools, "temperature": 0.2})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]


@dataclass
class AgentResult:
    answer: str
    steps: int
    tool_calls: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    unverified_citations: List[str] = field(default_factory=list)


class ResearchAgent:
    def __init__(self, llm: Callable[[List[dict], List[dict]], dict] = None, toolbox: Toolbox = None,
                 max_steps: int = 8, verbose: bool = False):
        self.llm = llm or LLMClient()
        self.tools = toolbox or Toolbox()
        self.max_steps = max_steps
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def run(self, question: str) -> AgentResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        trace: List[dict] = []
        for step in range(1, self.max_steps + 1):
            msg = self.llm(messages, TOOL_SCHEMAS)
            calls = msg.get("tool_calls") or []
            assistant_msg = {"role": "assistant", "content": msg.get("content")}
            if calls:
                assistant_msg["tool_calls"] = calls
            messages.append(assistant_msg)
            if not calls:
                return self._finish(msg.get("content") or "", step, trace)
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                self._log(f"🔧 step {step}: {name}({args})")
                out = self.tools.call(name, args)
                trace.append({"step": step, "tool": name, "args": args})
                messages.append({"role": "tool", "tool_call_id": c.get("id", name),
                                 "content": json.dumps(out, ensure_ascii=False)[:6000]})
        messages.append({"role": "user", "content": "Step limit reached. Write the final brief now without tools."})
        msg = self.llm(messages, [])
        return self._finish(msg.get("content") or "", self.max_steps, trace)

    def _finish(self, answer: str, steps: int, trace: List[dict]) -> AgentResult:
        # Guardrail: flag any cited arXiv id the agent never actually retrieved.
        cited = set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", answer))
        unverified = sorted(cited - set(self.tools.seen))
        return AgentResult(answer, steps, trace, list(self.tools.notes), unverified)
