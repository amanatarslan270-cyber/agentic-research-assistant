import json

from research_agent import ResearchAgent, Toolbox

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.11305v2</id>
    <published>2023-01-26T00:00:00Z</published>
    <title>DetectGPT: Zero-Shot Machine-Generated
      Text Detection</title>
    <summary>We study zero-shot detection of LLM text using probability curvature.</summary>
    <author><name>Eric Mitchell</name></author>
  </entry>
</feed>"""


def fake_http(url, params):
    return FEED


def _call(i, name, args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedLLM:
    """Plays back a fixed sequence of assistant messages."""

    def __init__(self, final_text):
        self.turn = 0
        self.final = final_text

    def __call__(self, messages, tools):
        self.turn += 1
        if self.turn == 1:
            return {"role": "assistant", "content": None,
                    "tool_calls": [_call(1, "search_arxiv", {"query": "AI text detection"})]}
        if self.turn == 2:
            assert messages[-1]["role"] == "tool" and "2301.11305" in messages[-1]["content"]
            return {"role": "assistant", "content": None,
                    "tool_calls": [_call(2, "get_paper", {"arxiv_id": "2301.11305"}),
                                   _call(3, "save_note", {"note": "DetectGPT uses curvature [arXiv:2301.11305]"})]}
        return {"role": "assistant", "content": self.final}


def test_toolbox_parses_arxiv():
    tb = Toolbox(http_get=fake_http)
    res = tb.search_arxiv("x")
    assert res == [{"arxiv_id": "2301.11305", "title": "DetectGPT: Zero-Shot Machine-Generated Text Detection",
                    "published": "2023-01-26"}]
    assert tb.get_paper("2301.11305")["authors"] == ["Eric Mitchell"]
    assert "error" in tb.call("nope", {})


def test_agent_loop_and_guardrail():
    llm = ScriptedLLM("## Key papers\n- DetectGPT [arXiv:2301.11305]\n- Made up [arXiv:9999.99999]")
    res = ResearchAgent(llm=llm, toolbox=Toolbox(http_get=fake_http)).run("detecting AI text?")
    assert res.steps == 3
    assert [t["tool"] for t in res.tool_calls] == ["search_arxiv", "get_paper", "save_note"]
    assert res.notes == ["DetectGPT uses curvature [arXiv:2301.11305]"]
    assert res.unverified_citations == ["9999.99999"]


def test_step_limit():
    class Looping:
        def __call__(self, messages, tools):
            if not tools:
                return {"role": "assistant", "content": "final"}
            return {"role": "assistant", "content": None, "tool_calls": [_call(1, "save_note", {"note": "n"})]}
    res = ResearchAgent(llm=Looping(), toolbox=Toolbox(http_get=fake_http), max_steps=3).run("q")
    assert res.answer == "final" and len(res.tool_calls) == 3
