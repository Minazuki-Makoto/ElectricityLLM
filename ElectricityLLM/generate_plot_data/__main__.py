from __future__ import annotations

import argparse
from .config import DEFAULT_DOCUMENT_PATH, DEFAULT_MAX_CHARS, DEFAULT_MODEL, DEFAULT_OUTPUT_DIRECTORY
from .pipeline import generate_plot_data


def main() -> None:
    parser = argparse.ArgumentParser(description="切分DOCX并生成description、embedding和plot_data")
    parser.add_argument("--document", default=str(DEFAULT_DOCUMENT_PATH))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有JSON继续；默认从第一个chunk完整重建",
    )
    args = parser.parse_args()
    generate_plot_data(
        document_path=args.document,
        output_directory=args.output_directory,
        model=args.model,
        max_chars=args.max_chars,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
