from pathlib import Path

from software_update_verification.arg_parser import create_parser


def test_create_parser_parses_image_directory_as_path() -> None:
    expected_path = Path("images")
    parser = create_parser()
    args = parser.parse_args([str(expected_path)])

    assert args.image_directory == expected_path
