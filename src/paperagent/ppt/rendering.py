from __future__ import annotations

import json
import subprocess
from pathlib import Path

from paperagent.config import Settings
from paperagent.schemas.models import RenderResult, SlideContent


class PPTRenderService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def render(
        self,
        paper_id: str,
        deck_title: str,
        slides: list[SlideContent],
        template_path: Path | None = None,
    ) -> RenderResult:
        runtime_info = self._detect_runtime()
        if not runtime_info["available"]:
            raise RuntimeError(
                "PPT skill runtime is unavailable. Missing @oai/artifact-tool or compatible JS builder environment."
            )

        deck_dir = self.settings.deck_dir / paper_id
        deck_dir.mkdir(parents=True, exist_ok=True)
        work_dir = deck_dir / "skill_builder"
        work_dir.mkdir(parents=True, exist_ok=True)

        content_path = work_dir / "deck_content.json"
        render_config_path = work_dir / "render_config.json"
        output_path = deck_dir / "output.pptx"
        content_path.write_text(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "title": deck_title,
                    "slides": [
                        {
                            "type": slide.slide_type,
                            "title": slide.title,
                            "bullets": slide.bullets,
                            "notes": slide.notes,
                            "citations": slide.citations,
                            "layout_hint": slide.layout_hint,
                            "visual_intent": slide.visual_intent,
                        }
                        for slide in slides
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        render_config_path.write_text(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "title": deck_title,
                    "content_path": str(content_path),
                    "output_path": str(output_path),
                    "template_path": str(template_path) if template_path else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        builder_script = Path(__file__).resolve().with_name("skill_builder.mjs")
        self._execute_builder(builder_script=builder_script, render_config_path=render_config_path, work_dir=work_dir)
        if not output_path.exists():
            raise RuntimeError("Skill renderer finished without creating output.pptx.")
        return RenderResult(
            ppt_path=str(output_path),
            slide_count=len(slides),
            renderer="skill",
        )

    def _detect_runtime(self) -> dict[str, bool]:
        command = [
            "node",
            "-e",
            "import('@oai/artifact-tool').then(()=>process.stdout.write('ok')).catch(()=>process.exit(1))",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return {"available": False}
        return {"available": completed.returncode == 0}

    def _execute_builder(self, builder_script: Path, render_config_path: Path, work_dir: Path) -> None:
        command = [
            "node",
            str(builder_script),
            str(render_config_path),
        ]
        completed = subprocess.run(
            command,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Skill renderer failed. "
                + (completed.stderr.strip() or completed.stdout.strip() or "Unknown JS builder error.")
            )
