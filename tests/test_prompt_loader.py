from __future__ import annotations

from paperagent.agent.prompts import PromptLoader


def test_paper_profile_prompt_keeps_literal_json_example():
    prompt = PromptLoader().load("paper_profile_system.txt")
    assert '"short_summary"' in prompt
    assert '"keywords"' in prompt


def test_ppt_generation_skill_renders_without_keyerror():
    prompt = PromptLoader().load(
        "ppt_generation_skill.txt",
        ppt_target_paper_id="paper-1",
        ppt_target_paper_title="Sample Paper",
    )
    assert '"paper_id": "paper-1"' in prompt
    assert '"slides"' in prompt
