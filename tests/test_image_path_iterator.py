from pathlib import Path

from software_update_verification.sample_io import iter_image_paths


def test_iter_image_paths_yields_supported_files_in_sorted_order(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()

    nested = images / "nested.png"
    nested.mkdir()
    (nested / "D003.jpeg").touch()

    for filename in ("D002.JPG", "D001.png", "notes.txt"):
        (images / filename).touch()

    expected_result = [images / "D001.png", images / "D002.JPG"]

    actual_result = list(iter_image_paths(images))

    assert actual_result == expected_result
