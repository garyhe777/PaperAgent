from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from datalab_sdk import ConvertOptions, DatalabClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use Datalab SDK to convert one PDF to Markdown and record timing.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("datasets/iccv2025_digital_watermark/Wang_ROAR_ICCV_2025.pdf"),
        help="Input PDF path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/datalab_trial"),
        help="Directory to store Markdown and metadata",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("DATALAB_API_KEY"),
        help="Datalab API key. Defaults to DATALAB_API_KEY env var.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "accurate"],
        help="Datalab processing mode",
    )
    parser.add_argument(
        "--page-range",
        type=str,
        default=None,
        help='Optional page range such as "0-2"',
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional max pages to process",
    )
    parser.add_argument(
        "--paginate",
        action="store_true",
        help="Insert page delimiters into markdown",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Missing API key. Provide --api-key or set DATALAB_API_KEY.")

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        parser.error(f"PDF not found: {pdf_path}")

    output_dir = args.output_dir.resolve() / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    client = DatalabClient(api_key=args.api_key)
    options = ConvertOptions(
        output_format="markdown",
        mode=args.mode,
        paginate=args.paginate,
        page_range=args.page_range,
        max_pages=args.max_pages,
    )

    started_at = time.perf_counter()
    result = client.convert(str(pdf_path), options=options)
    elapsed = time.perf_counter() - started_at

    result.save_output(str(output_dir), save_images=True)

    summary = {
        "pdf": str(pdf_path),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "page_range": args.page_range,
        "max_pages": args.max_pages,
        "paginate": args.paginate,
        "elapsed_seconds": round(elapsed, 2),
        "markdown_chars": len(result.markdown or ""),
    }
    summary_path = output_dir / "timing_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
