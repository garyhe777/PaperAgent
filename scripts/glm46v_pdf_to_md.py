from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import fitz
from openai import OpenAI


DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4.6v"
SYSTEM_PROMPT = """You are a precise PDF-to-Markdown transcription assistant.

Your task:
1. Read the provided PDF page image carefully.
2. Convert only the visible content on this page into clean Markdown.
3. Preserve headings, paragraphs, bullet lists, numbered lists, tables, and code blocks when visible.
4. For formulas, use LaTeX-friendly Markdown when possible.
5. For figures or diagrams, insert a short placeholder like [Figure: brief description].
6. Do not invent missing text.
7. Do not add commentary outside the page transcription.
8. Return Markdown only.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown with GLM-4.6V by rendering pages to images first.",
    )
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path. Defaults to <pdf>.glm46v.md",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GLM_MODEL", DEFAULT_MODEL),
        help=f"Vision model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("GLM_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL for GLM API",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GLM_API_KEY") or os.getenv("ZHIPUAI_API_KEY"),
        help="GLM API key. Defaults to GLM_API_KEY or ZHIPUAI_API_KEY env var",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Render DPI for each PDF page image",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional cap on number of pages to process",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between page requests",
    )
    return parser


def render_page_to_base64(page: fitz.Page, dpi: int) -> str:
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image_bytes = pix.tobytes("png")
    return base64.b64encode(image_bytes).decode("utf-8")


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def transcribe_page(
    client: OpenAI,
    model: str,
    page_index: int,
    total_pages: int,
    image_base64: str,
) -> str:
    user_prompt = (
        f"Convert page {page_index + 1} of {total_pages} into Markdown. "
        "Keep only content from this page."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                ],
            },
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Missing API key. Set --api-key or GLM_API_KEY / ZHIPUAI_API_KEY.")

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        parser.error(f"PDF not found: {pdf_path}")

    output_path = args.output.resolve() if args.output else pdf_path.with_suffix(".glm46v.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = build_client(args.base_url, args.api_key)

    document = fitz.open(pdf_path)
    total_pages = len(document)
    page_limit = min(total_pages, args.max_pages) if args.max_pages else total_pages

    markdown_parts: list[str] = []

    for page_index in range(page_limit):
        page = document.load_page(page_index)
        image_base64 = render_page_to_base64(page, dpi=args.dpi)
        started_at = time.perf_counter()
        page_markdown = transcribe_page(
            client=client,
            model=args.model,
            page_index=page_index,
            total_pages=total_pages,
            image_base64=image_base64,
        ).strip()
        elapsed = time.perf_counter() - started_at

        markdown_parts.append(f"<!-- page: {page_index + 1} -->\n")
        markdown_parts.append(page_markdown)
        markdown_parts.append("\n\n")

        print(
            f"[page {page_index + 1}/{page_limit}] rendered and transcribed in {elapsed:.2f}s",
            file=sys.stderr,
        )
        if args.sleep_seconds > 0 and page_index + 1 < page_limit:
            time.sleep(args.sleep_seconds)

    final_markdown = "".join(markdown_parts).strip() + "\n"
    output_path.write_text(final_markdown, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
# python scripts\glm46v_pdf_to_md.py  "C:\\Users\\garyh\\Desktop\\2503.11945v1.pdf" --api-key ae4d8567489549ebbeda7d58ce287773.zRIbghoHO6RIRAS1