"""Tools the agent can call. Each tool has a JSON schema (OpenAI function-calling format)."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}


def _http_get(url: str, params: dict) -> str:
    r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "research-agent/0.1"})
    r.raise_for_status()
    return r.text


def _parse_feed(xml: str) -> List[dict]:
    root = ET.fromstring(xml)
    papers = []
    for e in root.findall("a:entry", _NS):
        raw_id = e.findtext("a:id", "", _NS)
        papers.append({
            "arxiv_id": re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1]),
            "title": " ".join(e.findtext("a:title", "", _NS).split()),
            "authors": [a.findtext("a:name", "", _NS) for a in e.findall("a:author", _NS)][:6],
            "published": e.findtext("a:published", "", _NS)[:10],
            "summary": " ".join(e.findtext("a:summary", "", _NS).split()),
        })
    return papers


class Toolbox:
    """Holds tool implementations plus the agent's notes. `http_get` is injectable for tests."""

    def __init__(self, http_get: Callable[[str, dict], str] = _http_get):
        self.http_get = http_get
        self.notes: List[str] = []
        self.seen: Dict[str, dict] = {}

    # ---- tools -------------------------------------------------------------
    def search_arxiv(self, query: str, max_results: int = 5) -> List[dict]:
        xml = self.http_get(ARXIV_API, {"search_query": f"all:{query}", "start": 0,
                                        "max_results": max(1, min(int(max_results), 10)),
                                        "sortBy": "relevance"})
        papers = _parse_feed(xml)
        for p in papers:
            self.seen[p["arxiv_id"]] = p
        return [{k: p[k] for k in ("arxiv_id", "title", "published")} for p in papers]

    def get_paper(self, arxiv_id: str) -> dict:
        if arxiv_id not in self.seen:
            papers = _parse_feed(self.http_get(ARXIV_API, {"id_list": arxiv_id}))
            if not papers:
                return {"error": f"paper {arxiv_id} not found"}
            self.seen[arxiv_id] = papers[0]
        return self.seen[arxiv_id]

    def save_note(self, note: str) -> dict:
        self.notes.append(note)
        return {"saved": True, "total_notes": len(self.notes)}

    # ---- registry ----------------------------------------------------------
    def call(self, name: str, args: dict):
        fn = {"search_arxiv": self.search_arxiv, "get_paper": self.get_paper, "save_note": self.save_note}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except Exception as exc:  # report errors back to the model instead of crashing
            return {"error": f"{type(exc).__name__}: {exc}"}


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_arxiv",
        "description": "Search arXiv for papers. Returns ids, titles and dates.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "keywords"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_paper",
        "description": "Get the full abstract, authors and date of an arXiv paper.",
        "parameters": {"type": "object", "properties": {"arxiv_id": {"type": "string"}},
                       "required": ["arxiv_id"]}}},
    {"type": "function", "function": {
        "name": "save_note",
        "description": "Save a key finding (with its arXiv id) for the final report.",
        "parameters": {"type": "object", "properties": {"note": {"type": "string"}},
                       "required": ["note"]}}},
]
