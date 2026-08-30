import hashlib
import json
from pathlib import Path

import pytest

from software_update_verification.models import (
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
