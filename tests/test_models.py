import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from software_update_verification.models import (
    AnalysisFailure,
    AnalysisResponse,
    ExperimentConfig,
    LoadedImage,
    Sample,
)


VALID_ANALYSIS_OUTPUT_TEXT = json.dumps(
    {
        "decision": "accept",
        "platform": "windows",
        "evidence_codes": ["explicit_up_to_date"],
        "reason": "The system is explicitly shown as up to date.",
    }
)


def test_sample_is_created_with_valid_values() -> None:
    expected_path = Path("some/image.png")
    expected_hash = "a" * 64
    result = Sample(
        sample_id="D001",
        image_path=expected_path,
        image_sha256=expected_hash,
    )
    assert isinstance(result, Sample)
    assert result.sample_id == "D001"
    assert result.image_path == expected_path
    assert len(result.image_sha256) == 64
    assert result.image_sha256 == expected_hash


@pytest.mark.parametrize(
    "blank_sample_id",
    [
        "",
        " ",
        "\t",
        "\n",
        "\t\n",
    ],
)
def test_sample_rejects_blank_sample_id(blank_sample_id: str) -> None:
    expected_path = Path("some/image.png")
    expected_hash = "a" * 64

    with pytest.raises(
        ValueError,
        match="sample_id must not be blank",
    ):
        Sample(
            sample_id=blank_sample_id,
            image_path=expected_path,
            image_sha256=expected_hash,
        )


@pytest.mark.parametrize(
    "invalid_sample_id_type",
    [
        None,
        145,
        12.3,
        True,
        b"D001",
    ],
)
def test_sample_rejects_non_string_sample_id(invalid_sample_id_type: object) -> None:
    expected_path = Path("some/image.png")
    expected_hash = "a" * 64

    with pytest.raises(
        TypeError,
        match="sample_id must be a string",
    ):
        # noinspection PyTypeChecker
        Sample(
            sample_id=invalid_sample_id_type,
            image_path=expected_path,
            image_sha256=expected_hash,
        )


@pytest.mark.parametrize(
    "invalid_image_path",
    [
        None,
        "some/image.png",
        123,
    ],
)
def test_sample_rejects_non_path_image_path(invalid_image_path: object) -> None:
    expected_sample_id = "D001"
    expected_hash = "a" * 64

    with pytest.raises(
        TypeError,
        match="image_path must be a Path",
    ):
        # noinspection PyTypeChecker
        Sample(
            sample_id=expected_sample_id,
            image_path=invalid_image_path,
            image_sha256=expected_hash,
        )


@pytest.mark.parametrize(
    "non_string_image_sha256",
    [
        123,
        None,
        True,
        12.3,
        b"a" * 64,
    ],
)
def test_sample_rejects_non_string_image_sha256(
    non_string_image_sha256: object,
) -> None:
    sample_id = "D001"
    image_path = Path("some/image.png")

    with pytest.raises(
        TypeError,
        match="image_sha256 must be a string",
    ):
        # noinspection PyTypeChecker
        Sample(
            sample_id=sample_id,
            image_path=image_path,
            image_sha256=non_string_image_sha256,
        )


@pytest.mark.parametrize(
    "invalid_length_image_sha256",
    [
        "a" * 34,
        "a" * 63,
        "a" * 65,
    ],
)
def test_sample_rejects_image_sha256_with_invalid_length(
    invalid_length_image_sha256: str,
) -> None:
    sample_id = "D001"
    image_path = Path("some/image.png")

    with pytest.raises(
        ValueError,
        match="image_sha256 must be exactly 64 characters long",
    ):
        Sample(
            sample_id=sample_id,
            image_path=image_path,
            image_sha256=invalid_length_image_sha256,
        )


@pytest.mark.parametrize(
    "non_lowercase_hexadecimal_image_sha256",
    [
        "g" * 64,
        "A" * 64,
        "a" * 63 + "!",
        "a" * 63 + " ",
    ],
)
def test_sample_rejects_non_lowercase_hexadecimal_image_sha256(
    non_lowercase_hexadecimal_image_sha256: str,
) -> None:
    sample_id = "D001"
    image_path = Path("some/image.png")

    with pytest.raises(
        ValueError,
        match="image_sha256 must contain only lowercase hexadecimal characters",
    ):
        Sample(
            sample_id=sample_id,
            image_path=image_path,
            image_sha256=non_lowercase_hexadecimal_image_sha256,
        )


def test_loaded_image_is_created_with_valid_values() -> None:
    expected_content = b"test"
    expected_media_type = "image/png"
    expected_sha256 = hashlib.sha256(expected_content).hexdigest()

    result = LoadedImage(
        content=expected_content,
        media_type=expected_media_type,
        actual_sha256=expected_sha256,
    )

    assert isinstance(result, LoadedImage)
    assert result.content == expected_content
    assert result.media_type == expected_media_type
    assert len(result.actual_sha256) == 64
    assert result.actual_sha256 == expected_sha256


@pytest.mark.parametrize(
    "invalid_content_type",
    [
        None,
        "test",
        bytearray(b"test"),
        123,
        13.3,
        memoryview(b"test"),
        ["t", "e", "s", "t"],
    ],
)
def test_loaded_image_rejects_non_bytes_content(
    invalid_content_type: object,
) -> None:
    media_type = "image/png"
    valid_sha256 = "a" * 64

    with pytest.raises(
        TypeError,
        match="content must be bytes",
    ):
        # noinspection PyTypeChecker
        LoadedImage(
            content=invalid_content_type,
            media_type=media_type,
            actual_sha256=valid_sha256,
        )


def test_loaded_image_rejects_empty_content() -> None:
    content = b""
    media_type = "image/png"
    valid_sha256 = "a" * 64

    with pytest.raises(
        ValueError,
        match="content must not be empty",
    ):
        LoadedImage(
            content=content,
            media_type=media_type,
            actual_sha256=valid_sha256,
        )


@pytest.mark.parametrize(
    "invalid_media_type",
    [
        None,
        123,
        12.3,
        True,
        b"image/png",
    ],
)
def test_loaded_image_rejects_non_string_media_type(
    invalid_media_type: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="media_type must be a string",
    ):
        # noinspection PyTypeChecker
        LoadedImage(
            content=b"test",
            media_type=invalid_media_type,
            actual_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "blank_media_type",
    [
        "",
        " ",
        "\t",
        "\n",
        "\t\n",
    ],
)
def test_loaded_image_rejects_blank_media_type(blank_media_type: str) -> None:
    with pytest.raises(
        ValueError,
        match="media_type must not be blank",
    ):
        LoadedImage(
            content=b"test",
            media_type=blank_media_type,
            actual_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "unsupported_media_type",
    [
        "image/jpg",
        "image/gif",
        "image/webp",
        "application/pdf",
        "IMAGE/PNG",
    ],
)
def test_loaded_image_rejects_unsupported_media_type(
    unsupported_media_type: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="media_type must be one of: image/jpeg, image/png",
    ):
        LoadedImage(
            content=b"test",
            media_type=unsupported_media_type,
            actual_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "invalid_actual_sha256",
    [
        None,
        123,
        12.3,
        True,
        b"a" * 64,
    ],
)
def test_loaded_image_rejects_non_string_actual_sha256(
    invalid_actual_sha256: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="actual_sha256 must be a string",
    ):
        # noinspection PyTypeChecker
        LoadedImage(
            content=b"test",
            media_type="image/png",
            actual_sha256=invalid_actual_sha256,
        )


@pytest.mark.parametrize(
    "invalid_actual_sha256",
    [
        "",
        "a" * 34,
        "a" * 63,
        "a" * 65,
    ],
)
def test_loaded_image_rejects_actual_sha256_with_invalid_length(
    invalid_actual_sha256: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="actual_sha256 must be exactly 64 characters long",
    ):
        LoadedImage(
            content=b"test",
            media_type="image/png",
            actual_sha256=invalid_actual_sha256,
        )


@pytest.mark.parametrize(
    "invalid_actual_sha256",
    [
        "g" * 64,
        "A" * 64,
        "a" * 63 + "!",
        "a" * 63 + " ",
    ],
)
def test_loaded_image_rejects_non_lowercase_hexadecimal_actual_sha256(
    invalid_actual_sha256: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="actual_sha256 must contain only lowercase hexadecimal characters",
    ):
        LoadedImage(
            content=b"test",
            media_type="image/png",
            actual_sha256=invalid_actual_sha256,
        )


def test_experiment_config_is_created_with_valid_values() -> None:
    expected_model = "gpt-4o-mini-2024-07-18"
    expected_prompt = "Verify the software update status shown in the image."
    expected_prompt_version = "v1"
    expected_prompt_sha256 = "a" * 64
    expected_response_schema = {"type": "object", "properties": {}}
    expected_schema_version = "v1"
    expected_schema_sha256 = "b" * 64
    expected_image_detail = "high"

    result = ExperimentConfig(
        model=expected_model,
        prompt=expected_prompt,
        prompt_version=expected_prompt_version,
        prompt_sha256=expected_prompt_sha256,
        response_schema=expected_response_schema,
        schema_version=expected_schema_version,
        schema_sha256=expected_schema_sha256,
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


def _make_experiment_config(**overrides: object) -> ExperimentConfig:
    values = {
        "model": "gpt-4o-mini-2024-07-18",
        "prompt": "Verify the software update status shown in the image.",
        "prompt_version": "v1",
        "prompt_sha256": "a" * 64,
        "response_schema": {"type": "object", "properties": {}},
        "schema_version": "v1",
        "schema_sha256": "b" * 64,
        "image_detail": "high",
    }
    values.update(overrides)
    return ExperimentConfig(**values)


@pytest.mark.parametrize(
    "invalid_model",
    [None, 123, 12.3, True, b"gpt-4o-mini"],
)
def test_experiment_config_rejects_non_string_model(invalid_model: object) -> None:
    with pytest.raises(TypeError, match="model must be a string"):
        _make_experiment_config(model=invalid_model)


@pytest.mark.parametrize("blank_model", ["", " ", "\t", "\n", "\t\n"])
def test_experiment_config_rejects_blank_model(blank_model: str) -> None:
    with pytest.raises(ValueError, match="model must not be blank"):
        _make_experiment_config(model=blank_model)


@pytest.mark.parametrize(
    "invalid_prompt",
    [None, 123, 12.3, True, b"prompt"],
)
def test_experiment_config_rejects_non_string_prompt(invalid_prompt: object) -> None:
    with pytest.raises(TypeError, match="prompt must be a string"):
        _make_experiment_config(prompt=invalid_prompt)


@pytest.mark.parametrize("blank_prompt", ["", " ", "\t", "\n", "\t\n"])
def test_experiment_config_rejects_blank_prompt(blank_prompt: str) -> None:
    with pytest.raises(ValueError, match="prompt must not be blank"):
        _make_experiment_config(prompt=blank_prompt)


@pytest.mark.parametrize(
    "invalid_prompt_version",
    [None, 123, 12.3, True, b"v1"],
)
def test_experiment_config_rejects_non_string_prompt_version(
    invalid_prompt_version: object,
) -> None:
    with pytest.raises(TypeError, match="prompt_version must be a string"):
        _make_experiment_config(prompt_version=invalid_prompt_version)


@pytest.mark.parametrize("blank_prompt_version", ["", " ", "\t", "\n", "\t\n"])
def test_experiment_config_rejects_blank_prompt_version(
    blank_prompt_version: str,
) -> None:
    with pytest.raises(ValueError, match="prompt_version must not be blank"):
        _make_experiment_config(prompt_version=blank_prompt_version)


@pytest.mark.parametrize(
    "invalid_prompt_sha256",
    [None, 123, 12.3, True, b"a" * 64],
)
def test_experiment_config_rejects_non_string_prompt_sha256(
    invalid_prompt_sha256: object,
) -> None:
    with pytest.raises(TypeError, match="prompt_sha256 must be a string"):
        _make_experiment_config(prompt_sha256=invalid_prompt_sha256)


@pytest.mark.parametrize(
    "invalid_prompt_sha256",
    ["", "a" * 34, "a" * 63, "a" * 65],
)
def test_experiment_config_rejects_prompt_sha256_with_invalid_length(
    invalid_prompt_sha256: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="prompt_sha256 must be exactly 64 characters long",
    ):
        _make_experiment_config(prompt_sha256=invalid_prompt_sha256)


@pytest.mark.parametrize(
    "invalid_prompt_sha256",
    ["g" * 64, "A" * 64, "a" * 63 + "!", "a" * 63 + " "],
)
def test_experiment_config_rejects_non_lowercase_hexadecimal_prompt_sha256(
    invalid_prompt_sha256: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="prompt_sha256 must contain only lowercase hexadecimal characters",
    ):
        _make_experiment_config(prompt_sha256=invalid_prompt_sha256)


@pytest.mark.parametrize(
    "invalid_response_schema",
    [None, "schema", 123, 12.3, True, [], ()],
)
def test_experiment_config_rejects_non_mapping_response_schema(
    invalid_response_schema: object,
) -> None:
    with pytest.raises(TypeError, match="response_schema must be a Mapping"):
        _make_experiment_config(response_schema=invalid_response_schema)


def test_experiment_config_rejects_empty_response_schema() -> None:
    with pytest.raises(ValueError, match="response_schema must not be empty"):
        _make_experiment_config(response_schema={})


@pytest.mark.parametrize(
    "invalid_schema_version",
    [None, 123, 12.3, True, b"v1"],
)
def test_experiment_config_rejects_non_string_schema_version(
    invalid_schema_version: object,
) -> None:
    with pytest.raises(TypeError, match="schema_version must be a string"):
        _make_experiment_config(schema_version=invalid_schema_version)


@pytest.mark.parametrize("blank_schema_version", ["", " ", "\t", "\n", "\t\n"])
def test_experiment_config_rejects_blank_schema_version(
    blank_schema_version: str,
) -> None:
    with pytest.raises(ValueError, match="schema_version must not be blank"):
        _make_experiment_config(schema_version=blank_schema_version)


@pytest.mark.parametrize(
    "invalid_schema_sha256",
    [None, 123, 12.3, True, b"a" * 64],
)
def test_experiment_config_rejects_non_string_schema_sha256(
    invalid_schema_sha256: object,
) -> None:
    with pytest.raises(TypeError, match="schema_sha256 must be a string"):
        _make_experiment_config(schema_sha256=invalid_schema_sha256)


@pytest.mark.parametrize(
    "invalid_schema_sha256",
    ["", "a" * 34, "a" * 63, "a" * 65],
)
def test_experiment_config_rejects_schema_sha256_with_invalid_length(
    invalid_schema_sha256: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="schema_sha256 must be exactly 64 characters long",
    ):
        _make_experiment_config(schema_sha256=invalid_schema_sha256)


@pytest.mark.parametrize(
    "invalid_schema_sha256",
    ["g" * 64, "A" * 64, "a" * 63 + "!", "a" * 63 + " "],
)
def test_experiment_config_rejects_non_lowercase_hexadecimal_schema_sha256(
    invalid_schema_sha256: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="schema_sha256 must contain only lowercase hexadecimal characters",
    ):
        _make_experiment_config(schema_sha256=invalid_schema_sha256)


@pytest.mark.parametrize(
    "invalid_image_detail",
    [None, True, 123, 12.3, b"123"],
)
def test_experiment_config_rejects_non_string_image_detail(
    invalid_image_detail: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="image_detail must be a string",
    ):
        _make_experiment_config(image_detail=invalid_image_detail)


@pytest.mark.parametrize(
    "blank_image_detail",
    ["", " ", "\t", "\n", "\t\n"],
)
def test_experiment_config_rejects_blank_image_detail(
    blank_image_detail: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="image_detail must not be blank",
    ):
        _make_experiment_config(image_detail=blank_image_detail)


@pytest.mark.parametrize(
    "unsupported_image_detail",
    ["original", "medium", "BANANA", "HIGH", "detailed"],
)
def test_experiment_config_rejects_unsupported_image_detail(
    unsupported_image_detail: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="image_detail must be one of: auto, high, low",
    ):
        _make_experiment_config(image_detail=unsupported_image_detail)


@pytest.mark.parametrize("supported_image_detail", ["auto", "high", "low"])
def test_experiment_config_accepts_supported_image_detail(
    supported_image_detail: str,
) -> None:
    result = _make_experiment_config(image_detail=supported_image_detail)

    assert result.image_detail == supported_image_detail


def _make_analysis_response(**overrides: object) -> AnalysisResponse:
    values = {
        "response_id": "resp_test_001",
        "response_model": "gpt-4o-mini-2024-07-18",
        "output_text": VALID_ANALYSIS_OUTPUT_TEXT,
        "input_tokens": 1250,
        "output_tokens": 75,
    }
    values.update(overrides)
    return AnalysisResponse(**values)


def test_analysis_response_is_created_with_valid_values() -> None:
    expected_response_id = "resp_test_001"
    expected_response_model = "gpt-4o-mini-2024-07-18"
    expected_output_text = VALID_ANALYSIS_OUTPUT_TEXT
    expected_input_tokens = 1250
    expected_output_tokens = 75

    result = AnalysisResponse(
        response_id=expected_response_id,
        response_model=expected_response_model,
        output_text=expected_output_text,
        input_tokens=expected_input_tokens,
        output_tokens=expected_output_tokens,
    )

    assert isinstance(result, AnalysisResponse)
    assert result.response_id == expected_response_id
    assert result.response_model == expected_response_model
    assert result.output_text == expected_output_text
    assert result.input_tokens == expected_input_tokens
    assert result.output_tokens == expected_output_tokens


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_message"),
    [
        ("response_id", None, "response_id must be a string"),
        ("response_id", 123, "response_id must be a string"),
        ("response_id", b"resp_test_001", "response_id must be a string"),
        ("response_model", None, "response_model must be a string"),
        ("response_model", 123, "response_model must be a string"),
        ("response_model", b"gpt-4o-mini", "response_model must be a string"),
        ("output_text", None, "output_text must be a string"),
        ("output_text", 123, "output_text must be a string"),
        ("output_text", {"decision": "accept"}, "output_text must be a string"),
    ],
)
def test_analysis_response_rejects_non_string_text_fields(
    field_name: str,
    invalid_value: object,
    expected_message: str,
) -> None:
    with pytest.raises(TypeError, match=expected_message):
        _make_analysis_response(**{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "blank_value", "expected_message"),
    [
        ("response_id", "", "response_id must not be blank"),
        ("response_id", " \t\n", "response_id must not be blank"),
        ("response_model", "", "response_model must not be blank"),
        ("response_model", " \t\n", "response_model must not be blank"),
        ("output_text", "", "output_text must not be blank"),
        ("output_text", " \t\n", "output_text must not be blank"),
    ],
)
def test_analysis_response_rejects_blank_text_fields(
    field_name: str,
    blank_value: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        _make_analysis_response(**{field_name: blank_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_message"),
    [
        ("input_tokens", None, "input_tokens must be an integer"),
        ("input_tokens", True, "input_tokens must be an integer"),
        ("input_tokens", 12.5, "input_tokens must be an integer"),
        ("output_tokens", None, "output_tokens must be an integer"),
        ("output_tokens", False, "output_tokens must be an integer"),
        ("output_tokens", 12.5, "output_tokens must be an integer"),
    ],
)
def test_analysis_response_rejects_non_integer_token_counts(
    field_name: str,
    invalid_value: object,
    expected_message: str,
) -> None:
    with pytest.raises(TypeError, match=expected_message):
        _make_analysis_response(**{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "expected_message"),
    [
        ("input_tokens", "input_tokens must not be negative"),
        ("output_tokens", "output_tokens must not be negative"),
    ],
)
def test_analysis_response_rejects_negative_token_counts(
    field_name: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        _make_analysis_response(**{field_name: -1})


@pytest.mark.parametrize("field_name", ["input_tokens", "output_tokens"])
def test_analysis_response_accepts_zero_token_counts(field_name: str) -> None:
    result = _make_analysis_response(**{field_name: 0})

    assert getattr(result, field_name) == 0


def _make_analysis_failure(**overrides: object) -> AnalysisFailure:
    data = {
        "failure_code": "timeout",
        "diagnostic_message": "Request timed out",
        "retryable": True,
        "request_id": None,
        "status_code": None,
        "response_status": None,
        "response_id": None,
    }
    data.update(overrides)
    return AnalysisFailure(**data)


def test_analysis_failure_is_created_with_valid_values() -> None:
    expected_analysis_failure = {
        "failure_code": "rate_limit",
        "diagnostic_message": "Rate limit exceeded",
        "retryable": True,
        "request_id": "req_test_001",
        "status_code": 429,
        "response_status": None,
        "response_id": None,
    }

    result = AnalysisFailure(**expected_analysis_failure)

    assert isinstance(result, AnalysisFailure)
    assert asdict(result) == expected_analysis_failure


@pytest.mark.parametrize(
    "invalid_failure_code",
    [None, True, 123, 13.3, b"hello"],
)
def test_analysis_failure_rejects_non_string_failure_code(
    invalid_failure_code: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="failure_code must be a string",
    ):
        _make_analysis_failure(failure_code=invalid_failure_code)


@pytest.mark.parametrize("blank_failure_code", ["", " ", "\t", "\n", "\t\n"])
def test_analysis_failure_rejects_blank_failure_code(
    blank_failure_code: str,
) -> None:
    with pytest.raises(ValueError, match="failure_code must not be blank"):
        _make_analysis_failure(failure_code=blank_failure_code)


@pytest.mark.parametrize(
    "invalid_diagnostic_message",
    [None, True, 123, 13.3, b"Request timed out"],
)
def test_analysis_failure_rejects_non_string_diagnostic_message(
    invalid_diagnostic_message: object,
) -> None:
    with pytest.raises(TypeError, match="diagnostic_message must be a string"):
        _make_analysis_failure(diagnostic_message=invalid_diagnostic_message)


@pytest.mark.parametrize(
    "blank_diagnostic_message",
    ["", " ", "\t", "\n", "\t\n"],
)
def test_analysis_failure_rejects_blank_diagnostic_message(
    blank_diagnostic_message: str,
) -> None:
    with pytest.raises(ValueError, match="diagnostic_message must not be blank"):
        _make_analysis_failure(diagnostic_message=blank_diagnostic_message)


@pytest.mark.parametrize(
    "invalid_retryable",
    [None, 0, 1, "true", b"true", []],
)
def test_analysis_failure_rejects_non_boolean_retryable(
    invalid_retryable: object,
) -> None:
    with pytest.raises(TypeError, match="retryable must be a boolean"):
        _make_analysis_failure(retryable=invalid_retryable)


@pytest.mark.parametrize("retryable", [True, False])
def test_analysis_failure_accepts_boolean_retryable(retryable: bool) -> None:
    result = _make_analysis_failure(retryable=retryable)

    assert result.retryable is retryable


@pytest.mark.parametrize("field_name", ["request_id", "response_id"])
@pytest.mark.parametrize(
    "invalid_identifier",
    [True, 123, 13.3, b"identifier", []],
)
def test_analysis_failure_rejects_non_string_optional_identifiers(
    field_name: str,
    invalid_identifier: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"{field_name} must be a string or None",
    ):
        _make_analysis_failure(**{field_name: invalid_identifier})


@pytest.mark.parametrize("field_name", ["request_id", "response_id"])
@pytest.mark.parametrize("blank_identifier", ["", " ", "\t", "\n", "\t\n"])
def test_analysis_failure_rejects_blank_optional_identifiers(
    field_name: str,
    blank_identifier: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{field_name} must not be blank"):
        _make_analysis_failure(**{field_name: blank_identifier})


@pytest.mark.parametrize("field_name", ["request_id", "response_id"])
def test_analysis_failure_accepts_none_optional_identifiers(
    field_name: str,
) -> None:
    result = _make_analysis_failure(**{field_name: None})

    assert getattr(result, field_name) is None


@pytest.mark.parametrize(
    "invalid_status_code",
    [True, False, "429", 429.0, b"429", []],
)
def test_analysis_failure_rejects_non_integer_status_code(
    invalid_status_code: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="status_code must be an integer or None",
    ):
        _make_analysis_failure(status_code=invalid_status_code)


@pytest.mark.parametrize("invalid_status_code", [-1, 0, 399, 600, 999])
def test_analysis_failure_rejects_status_code_outside_error_range(
    invalid_status_code: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="status_code must be between 400 and 599",
    ):
        _make_analysis_failure(status_code=invalid_status_code)


@pytest.mark.parametrize("status_code", [None, 400, 429, 500, 599])
def test_analysis_failure_accepts_valid_optional_status_code(
    status_code: int | None,
) -> None:
    result = _make_analysis_failure(status_code=status_code)

    assert result.status_code == status_code


@pytest.mark.parametrize(
    "invalid_response_status",
    [True, 123, 13.3, b"failed", []],
)
def test_analysis_failure_rejects_non_string_response_status(
    invalid_response_status: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="response_status must be a string or None",
    ):
        _make_analysis_failure(response_status=invalid_response_status)


@pytest.mark.parametrize(
    "blank_response_status",
    ["", " ", "\t", "\n", "\t\n"],
)
def test_analysis_failure_rejects_blank_response_status(
    blank_response_status: str,
) -> None:
    with pytest.raises(ValueError, match="response_status must not be blank"):
        _make_analysis_failure(response_status=blank_response_status)

@pytest.mark.parametrize(
    "response_status",
    [
        None,
        "completed",
        "future_status",
    ],
)
def test_analysis_failure_accepts_optional_nonblank_response_status(
    response_status: str | None,
) -> None:
    result = _make_analysis_failure(response_status=response_status)

    assert result.response_status == response_status
