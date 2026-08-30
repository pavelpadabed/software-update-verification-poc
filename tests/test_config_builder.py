import hashlib
import json
from pathlib import Path

import pytest

from software_update_verification.config_builder import build_experiment_config
from software_update_verification.models import ExperimentConfig

MODEL = "gpt-4o-mini-2024-07-18"
PROMPT_VERSION = "v1"
SCHEMA_VERSION = "v1"
IMAGE_DETAIL = "high"


def _write_valid_config_files(tmp_path: Path) -> tuple[Path, Path]:
    prompt_path = tmp_path / "verification_v1.txt"
    prompt_path.write_text("prompt text", encoding="utf-8")
    schema_path = tmp_path / "verification_response.json"
    schema_mapping = {"type": "object", "properties": {}}
    schema_path.write_text(json.dumps(schema_mapping), encoding="utf-8")

    return prompt_path, schema_path


def _build_experiment_config_from_paths(
    prompt_path: object,
    schema_path: object,
) -> ExperimentConfig:
    return build_experiment_config(
        model=MODEL,
        prompt_path=prompt_path,
        prompt_version=PROMPT_VERSION,
        schema_path=schema_path,
        schema_version=SCHEMA_VERSION,
        image_detail=IMAGE_DETAIL,
    )


def test_build_experiment_config_returns_config_for_valid_files(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "verification_v1.txt"
    expected_prompt = "prompt text"
    prompt_bytes = expected_prompt.encode("utf-8")
    prompt_path.write_bytes(prompt_bytes)
    expected_prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()

    schema_path = tmp_path / "verification_response_v1.json"
    expected_response_schema = {"type": "object", "properties": {}}
    schema_text = json.dumps(expected_response_schema)
    schema_bytes = schema_text.encode("utf-8")
    schema_path.write_bytes(schema_bytes)
    expected_schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()

    expected_model = "gpt-4o-mini-2024-07-18"
    expected_prompt_version = "v1"
    expected_schema_version = "v1"
    expected_image_detail = "high"

    result = build_experiment_config(
        model=expected_model,
        prompt_path=prompt_path,
        prompt_version=expected_prompt_version,
        schema_path=schema_path,
        schema_version=expected_schema_version,
        image_detail=expected_image_detail,
    )

    assert isinstance(result, ExperimentConfig)
    assert result.model == expected_model
    assert result.prompt == expected_prompt
    assert result.prompt_version == expected_prompt_version
    assert result.prompt_sha256 == expected_prompt_sha256
    assert result.response_schema == expected_response_schema
    assert result.schema_version == expected_schema_version
    assert result.schema_sha256 == expected_schema_sha256
    assert result.image_detail == expected_image_detail


@pytest.mark.parametrize(
    "invalid_path_type",
    [None, "something", 123, 13.3, True, []],
)
def test_build_experiment_config_rejects_non_path_prompt_path(
    tmp_path: Path,
    invalid_path_type: object,
) -> None:
    _, schema_path = _write_valid_config_files(tmp_path)
    with pytest.raises(
        TypeError,
        match="prompt_path must be a Path",
    ):
        _build_experiment_config_from_paths(
            prompt_path=invalid_path_type,
            schema_path=schema_path,
        )


@pytest.mark.parametrize(
    "invalid_schema_path_type",
    [None, "something", 123, 13.3, True, []],
)
def test_build_experiment_config_rejects_non_path_schema_path(
    tmp_path: Path,
    invalid_schema_path_type: object,
) -> None:
    prompt_path, _ = _write_valid_config_files(tmp_path)
    with pytest.raises(
        TypeError,
        match="schema_path must be a Path",
    ):
        _build_experiment_config_from_paths(
            prompt_path=prompt_path,
            schema_path=invalid_schema_path_type,
        )


def test_build_experiment_config_rejects_missing_prompt_file(
    tmp_path: Path,
) -> None:
    missing_prompt_path = tmp_path / "missing_prompt.txt"
    _, schema_path = _write_valid_config_files(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="prompt file does not exist",
    ):
        _build_experiment_config_from_paths(
            prompt_path=missing_prompt_path,
            schema_path=schema_path,
        )


def test_build_experiment_config_rejects_missing_schema_file(
    tmp_path: Path,
) -> None:
    prompt_path, _ = _write_valid_config_files(tmp_path)
    missing_schema_path = tmp_path / "missing_schema.json"

    with pytest.raises(
        FileNotFoundError,
        match="schema file does not exist",
    ):
        _build_experiment_config_from_paths(
            prompt_path=prompt_path,
            schema_path=missing_schema_path,
        )


def test_build_experiment_config_rejects_directory_prompt_path(
    tmp_path: Path,
) -> None:
    directory_prompt_path = tmp_path / "not_a_file"
    directory_prompt_path.mkdir()

    _, schema_path = _write_valid_config_files(tmp_path)

    with pytest.raises(
        ValueError,
        match="prompt_path must point to a regular file",
    ):
        _build_experiment_config_from_paths(
            prompt_path=directory_prompt_path,
            schema_path=schema_path,
        )


def test_build_experiment_config_rejects_directory_schema_path(
    tmp_path: Path,
) -> None:
    directory_schema_path = tmp_path / "not_a_regular_file"
    directory_schema_path.mkdir()

    prompt_path, _ = _write_valid_config_files(tmp_path)

    with pytest.raises(
        ValueError,
        match="schema_path must point to a regular file",
    ):
        _build_experiment_config_from_paths(
            prompt_path=prompt_path,
            schema_path=directory_schema_path,
        )


def test_build_experiment_config_rejects_empty_prompt_file(
    tmp_path: Path,
) -> None:
    prompt_path, schema_path = _write_valid_config_files(tmp_path)
    prompt_path.write_bytes(b"")

    with pytest.raises(
        ValueError,
        match="prompt file must not be empty",
    ):
        _build_experiment_config_from_paths(
            prompt_path=prompt_path,
            schema_path=schema_path,
        )


def test_build_experiment_config_rejects_empty_schema_file(
    tmp_path: Path,
) -> None:
    prompt_path, schema_path = _write_valid_config_files(tmp_path)
    schema_path.write_bytes(b"")

    with pytest.raises(
        ValueError,
        match="schema file must not be empty",
    ):
        _build_experiment_config_from_paths(
            prompt_path=prompt_path,
            schema_path=schema_path,
        )


def test_build_experiment_config_rejects_prompt_with_invalid_utf8(
    tmp_path: Path,
) -> None:
    prompt_path, schema_path = _write_valid_config_files(tmp_path)
    prompt_path.write_bytes(b"\xff")

    with pytest.raises(
        UnicodeDecodeError,
    ):
        _build_experiment_config_from_paths(
            prompt_path=prompt_path,
            schema_path=schema_path,
        )


def test_build_experiment_config_rejects_schema_with_invalid_utf8(
    tmp_path: Path,
) -> None:
    prompt_path, schema_path = _write_valid_config_files(tmp_path)
    schema_path.write_bytes(b"\xff")

    with pytest.raises(
        UnicodeDecodeError,
    ):
        _build_experiment_config_from_paths(
            prompt_path=prompt_path,
            schema_path=schema_path,
        )


def test_build_experiment_config_rejects_invalid_json_schema(
    tmp_path: Path,
) -> None:
    prompt_path, schema_path = _write_valid_config_files(tmp_path)
    invalid_schema_text = '{"type": "object"'
    schema_path.write_text(invalid_schema_text, encoding="utf-8")

    with pytest.raises(
        json.JSONDecodeError,
    ):
        _build_experiment_config_from_paths(
            prompt_path=prompt_path,
            schema_path=schema_path,
        )


def test_build_experiment_config_rejects_json_schema_with_non_mapping_root(
    tmp_path: Path,
) -> None:
    prompt_path, schema_path = _write_valid_config_files(tmp_path)
    non_mapping_value = []
    schema_path.write_text(json.dumps(non_mapping_value), encoding="utf-8")

    with pytest.raises(
        TypeError,
        match="response_schema must be a Mapping",
    ):
        _build_experiment_config_from_paths(
            prompt_path=prompt_path,
            schema_path=schema_path,
        )


def test_build_experiment_config_rejects_empty_json_schema_mapping(
    tmp_path: Path,
) -> None:
    prompt_path, schema_path = _write_valid_config_files(tmp_path)
    empty_mapping = {}
    schema_path.write_text(json.dumps(empty_mapping), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="response_schema must not be empty",
    ):
        _build_experiment_config_from_paths(
            prompt_path=prompt_path,
            schema_path=schema_path,
        )
