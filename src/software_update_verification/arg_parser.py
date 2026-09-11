import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify operating-system update screenshots with OpenAI.",
    )

    parser.add_argument(
        "image_directory",
        type=Path,
        help="Directory containing anonymized PNG or JPEG images.",
    )

    return parser
