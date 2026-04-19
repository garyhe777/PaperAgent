from __future__ import annotations

from paperagent.agent.prompts import PromptLoader


def test_paper_profile_prompt_keeps_literal_json_example():
    prompt = PromptLoader().load("paper_profile_system.txt")
    assert '"short_summary"' in prompt
    assert '"keywords"' in prompt
