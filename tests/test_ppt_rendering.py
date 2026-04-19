from __future__ import annotations

from pathlib import Path

from paperagent.schemas.models import SlideContent
from paperagent.services import ServiceContainer


def test_renderer_falls_back_to_python_pptx_when_skill_runtime_is_missing(container: ServiceContainer):
    result = container.ppt_render_service.render(
        paper_id="paper-1",
        deck_title="Test Deck",
        slides=[
            SlideContent(
                slide_type="title",
                title="Test Deck",
                bullets=["Bullet"],
                notes="Notes",
                citations=[],
            )
        ],
    )
    assert Path(result.ppt_path).exists()
    assert result.renderer == "python-pptx"


def test_skill_renderer_creates_output_when_runtime_and_builder_succeed(container: ServiceContainer, monkeypatch):
    def fake_detect_runtime():
        return {"available": True}

    def fake_execute_builder(builder_script: Path, render_config_path: Path, work_dir: Path):
        config = __import__("json").loads(render_config_path.read_text(encoding="utf-8"))
        Path(config["output_path"]).write_bytes(b"fake-pptx")

    monkeypatch.setattr(container.ppt_render_service, "_detect_runtime", fake_detect_runtime)
    monkeypatch.setattr(container.ppt_render_service, "_execute_builder", fake_execute_builder)

    result = container.ppt_render_service.render(
        paper_id="paper-1",
        deck_title="Test Deck",
        slides=[
            SlideContent(
                slide_type="title",
                title="Test Deck",
                bullets=["Bullet"],
                notes="Notes",
                citations=[],
            )
        ],
    )

    assert Path(result.ppt_path).exists()
    assert result.renderer == "skill"
