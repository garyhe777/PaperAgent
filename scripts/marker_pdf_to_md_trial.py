from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trial script: convert a PDF to Markdown using marker-pdf.",
    )
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".paperagent_data/marker_trial"),
        help="Directory to write markdown, images, and metadata into",
    )
    parser.add_argument(
        "--page-range",
        type=str,
        default=None,
        help='Optional page range such as "0", "0-2", or "0,3-5"',
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR for the whole document",
    )
    parser.add_argument(
        "--paginate-output",
        action="store_true",
        help="Insert pagination markers into the markdown output",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    run_name = pdf_path.stem
    output_dir = args.output_dir.resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "output_format": "markdown",
        "paginate_output": args.paginate_output,
        "force_ocr": args.force_ocr,
    }
    if args.page_range:
        config["page_range"] = args.page_range

    # Marker's converter is the core pipeline object. We build it once, then
    # call it on a PDF path to get a structured rendered document back.
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    started_at = time.perf_counter()
    rendered = converter(str(pdf_path))
    elapsed = time.perf_counter() - started_at

    # text_from_rendered flattens marker's internal rendered document into
    # markdown text plus extracted images, which makes it easy to inspect.
    markdown_text, metadata, images = text_from_rendered(rendered)

    markdown_path = output_dir / f"{run_name}.marker.md"
    markdown_path.write_text(markdown_text, encoding="utf-8")

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    written_images: list[str] = []
    for image_name, image in images.items():
        image_path = images_dir / image_name
        image.save(image_path)
        written_images.append(str(image_path))

    summary = {
        "pdf_path": str(pdf_path),
        "markdown_path": str(markdown_path),
        "images_dir": str(images_dir),
        "image_count": len(written_images),
        "page_range": args.page_range,
        "force_ocr": args.force_ocr,
        "paginate_output": args.paginate_output,
        "elapsed_seconds": round(elapsed, 2),
        "metadata": metadata,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
