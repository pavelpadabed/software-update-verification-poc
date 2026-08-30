import hashlib
from pathlib import Path

import pytest

from software_update_verification.models import Sample
from software_update_verification.models import LoadedImage
from software_update_verification.sample_io import build_sample
from software_update_verification.sample_io import load_image

@pytest.mark.parametrize(
    "supported_image_suffix",
    [
        ".png",
        ".jpg",
        ".jpeg",
        ".PNG",
        ".JPG",
        ".JPEG",
    ],
)
def test_build_sample_returns_sample_for_valid_image_file(
    tmp_path,
    supported_image_suffix: str,
) -> None:
    image_path = tmp_path / f"D001{supported_image_suffix}"
    image_content = b"test"
    image_path.write_bytes(image_content)

    hash_object = hashlib.sha256(image_content)
    expected_image_sha256 = hash_object.hexdigest()

    result = build_sample(image_path)

    assert isinstance(result, Sample)
    assert result.image_path == image_path
    assert result.sample_id == "D001"
    assert result.image_sha256 == expected_image_sha256


@pytest.mark.parametrize(
    "invalid_path_type",
    [
        None,
        "some/image.png",
        123,
        13.3,
        True,
    ],
)
def test_build_sample_rejects_non_path_input(invalid_path_type: object) -> None:
    with pytest.raises(
        TypeError,
        match="path must be a Path",
    ):
        # noinspection PyTypeChecker
        build_sample(invalid_path_type)


def test_build_sample_rejects_symbolic_link(tmp_path) -> None:
    target_path = tmp_path / "target.png"
    target_path.write_bytes(b"test")

    symlink_path = tmp_path / "D001.png"
    symlink_path.symlink_to(target_path)

    with pytest.raises(
        ValueError,
        match="path must not be a symbolic link",
    ):
        build_sample(symlink_path)


def test_build_sample_rejects_missing_file(tmp_path) -> None:
    image_path = tmp_path / "missing.png"

    with pytest.raises(
        FileNotFoundError,
        match="image file does not exist",
    ):
        build_sample(image_path)


def test_build_sample_rejects_directory(tmp_path) -> None:
    dir_path = tmp_path / "image"
    dir_path.mkdir()

    with pytest.raises(
        ValueError,
        match="path must point to a regular file",
    ):
        build_sample(dir_path)


@pytest.mark.parametrize(
    "unsupported_image_format",
    [
        ".pdf",
        ".webp",
        ".txt",
        ".gif",
    ],
)
def test_build_sample_rejects_unsupported_image_format(
    tmp_path,
    unsupported_image_format: str,
) -> None:
    image_path = tmp_path / f"image{unsupported_image_format}"
    image_path.write_bytes(b"test")

    with pytest.raises(
        ValueError,
        match="unsupported image format",
    ):
        build_sample(image_path)


def test_build_sample_rejects_empty_image_file(tmp_path) -> None:
    image_path = tmp_path / "D001.png"
    image_path.touch()

    with pytest.raises(
        ValueError,
        match="image file must not be empty",
    ):
        build_sample(image_path)


def test_load_image_returns_loaded_image_for_valid_sample(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "D001.png"
    image_content = b"test"
    image_path.write_bytes(image_content)

    expected_sha256 = hashlib.sha256(image_content).hexdigest()

    sample = Sample(
        sample_id="D001",
        image_path=image_path,
        image_sha256=expected_sha256,
    )

    result = load_image(sample)

    assert isinstance(result, LoadedImage)
    assert result.content == image_content
    assert result.media_type == "image/png"
    assert result.actual_sha256 == expected_sha256


def test_load_image_rejects_content_changed_after_sample_creation(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "D001.jpg"
    original_content = b"test"
    changed_content = b"content"
    image_path.write_bytes(original_content)

    expected_sha256 = hashlib.sha256(original_content).hexdigest()

    sample = Sample(
        sample_id="D001",
        image_path=image_path,
        image_sha256=expected_sha256,
    )
    image_path.write_bytes(changed_content)

    with pytest.raises(
        ValueError,
        match="image content does not match sample hash",
    ):
        load_image(sample)


@pytest.mark.parametrize(
    "invalid_sample",
    [
        None,
        "banana",
        Path("something.png"),
        {},
        [],
        LoadedImage(
            content=b"test",
            media_type="image/png",
            actual_sha256="a" * 64,
        ),
        True,
    ],
)
def test_load_image_rejects_non_sample_input(
    invalid_sample: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="sample must be a Sample",
    ):
        # noinspection PyTypeChecker
        load_image(invalid_sample)


@pytest.mark.parametrize(
    ("suffix", "expected_media_type"),
    [
        (".png", "image/png"),
        (".PNG", "image/png"),
        (".jpg", "image/jpeg"),
        (".JPG", "image/jpeg"),
        (".jpeg", "image/jpeg"),
        (".JPEG", "image/jpeg"),
    ],
)
def test_load_image_maps_supported_suffix_to_media_type(
    tmp_path: Path,
    suffix: str,
    expected_media_type: str,
) -> None:
    image_path = tmp_path / ("D001" + suffix)
    image_content = b"test"
    image_path.write_bytes(image_content)

    image_sha256 = hashlib.sha256(image_content).hexdigest()

    sample = Sample(
        sample_id="D001",
        image_path=image_path,
        image_sha256=image_sha256,
    )

    result = load_image(sample)

    assert isinstance(result, LoadedImage)
    assert result.media_type == expected_media_type


@pytest.mark.parametrize(
    "unsupported_suffix",
    [
        ".gif",
        ".pdf",
        ".webp",
        ".txt",
    ],
)
def test_load_image_rejects_unsupported_image_suffix(
    tmp_path: Path,
    unsupported_suffix: str,
) -> None:
    image_path = tmp_path / ("D001" + unsupported_suffix)
    image_content = b"test"
    image_path.write_bytes(image_content)

    image_sha256 = hashlib.sha256(image_content).hexdigest()

    sample = Sample(
        sample_id="D001",
        image_path=image_path,
        image_sha256=image_sha256,
    )
    with pytest.raises(
        ValueError,
        match="unsupported image suffix",
    ):
        load_image(sample)
