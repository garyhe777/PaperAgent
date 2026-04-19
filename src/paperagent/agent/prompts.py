from __future__ import annotations

from pathlib import Path


class PromptLoader:
    def __init__(self) -> None:
        self.prompt_dir = Path(__file__).resolve().parents[1] / "prompts"

    def load(self, name: str, **kwargs: str) -> str:
        template = (self.prompt_dir / name).read_text(encoding="utf-8")
        return template.format(**kwargs)
