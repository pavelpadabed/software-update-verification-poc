import hashlib
import json
from pathlib import Path

from software_update_verification.models import ExperimentConfig


def build_experiment_config(
    model: str,
    prompt_path: Path,
    prompt_version: str,
    schema_path: Path,
    schema_version: str,
    image_detail: str,
) -> ExperimentConfig:
    if not isinstance(prompt_path, Path):
        raise TypeError("prompt_path must be a Path")
    if not isinstance(schema_path, Path):
        raise TypeError("schema_path must be a Path")
    if not prompt_path.exists():
        raise FileNotFoundError("prompt file does not exist")
    if not schema_path.exists():
        raise FileNotFoundError("schema file does not exist")
    if not prompt_path.is_file():
        raise ValueError("prompt_path must point to a regular file")
    if not schema_path.is_file():
        raise ValueError("schema_path must point to a regular file")
    prompt_bytes = prompt_path.read_bytes()
    if not prompt_bytes:
        raise ValueError("prompt file must not be empty")
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    prompt = prompt_bytes.decode("utf-8")

    schema_bytes = schema_path.read_bytes()
    if not schema_bytes:
        raise ValueError("schema file must not be empty")
    schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    schema_text = schema_bytes.decode("utf-8")
    response_schema = json.loads(schema_text)

    return ExperimentConfig(
        model=model,
        prompt=prompt,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        response_schema=response_schema,
        schema_version=schema_version,
        schema_sha256=schema_sha256,
        image_detail=image_detail,
    )
