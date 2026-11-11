"""research-agent "What are recent methods for detecting AI-generated text?" [--out brief.md]"""
from __future__ import annotations

import argparse

from .agent import ResearchAgent


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="research-agent", description="Autonomous arXiv literature-review agent")
    ap.add_argument("question")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--out", help="save the Markdown brief to this file")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    res = ResearchAgent(max_steps=a.max_steps, verbose=not a.quiet).run(a.question)
    print("\n" + res.answer)
    print(f"\n— {res.steps} steps, {len(res.tool_calls)} tool calls, {len(res.notes)} notes")
    if res.unverified_citations:
        print(f"⚠️  Citations not retrieved via tools (possible hallucination): {res.unverified_citations}")
    if a.out:
        open(a.out, "w", encoding="utf-8").write(res.answer + "\n")


if __name__ == "__main__":
    main()
