import hashlib
from collections.abc import Iterator
from pathlib import Path

from software_update_verification.models import LoadedImage, Sample

SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {
        ".png",
        ".jpeg",
        ".jpg",
    },
)
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def build_sample(path: Path) -> Sample:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if path.is_symlink():
        raise ValueError("path must not be a symbolic link")
    if not path.exists():
        raise FileNotFoundError("image file does not exist")
    if not path.is_file():
        raise ValueError("path must point to a regular file")
    sample_id = path.stem
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("unsupported image format")
    content = path.read_bytes()
    if not content:
        raise ValueError("image file must not be empty")
    image_sha256 = hashlib.sha256(content).hexdigest()

    return Sample(
        sample_id=sample_id,
        image_path=path,
        image_sha256=image_sha256,
    )


def load_image(sample: Sample) -> LoadedImage:
    if not isinstance(sample, Sample):
        raise TypeError("sample must be a Sample")
    content = sample.image_path.read_bytes()
    suffix = sample.image_path.suffix.lower()
    if suffix not in IMAGE_MEDIA_TYPES:
        raise ValueError("unsupported image suffix")
    media_type = IMAGE_MEDIA_TYPES[suffix]
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != sample.image_sha256:
        raise ValueError("image content does not match sample hash")

    return LoadedImage(
        content=content,
        media_type=media_type,
        actual_sha256=actual_sha256,
    )


def iter_image_paths(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        yield path
