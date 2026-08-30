from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_IMAGE_DETAILS = frozenset({"auto", "high", "low"})


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    image_path: Path
    image_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str):
            raise TypeError("sample_id must be a string")
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be blank")
        if not isinstance(self.image_path, Path):
            raise TypeError("image_path must be a Path")
        if not isinstance(self.image_sha256, str):
            raise TypeError("image_sha256 must be a string")
        if len(self.image_sha256) != 64:
            raise ValueError("image_sha256 must be exactly 64 characters long")
        allowed_characters = "0123456789abcdef"
        if not all(
            character in allowed_characters for character in self.image_sha256
        ):
            raise ValueError(
                "image_sha256 must contain only lowercase hexadecimal characters"
            )


@dataclass(frozen=True, slots=True)
class LoadedImage:
    content: bytes
    media_type: str
    actual_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise TypeError("content must be bytes")
        if not self.content:
            raise ValueError("content must not be empty")
        if not isinstance(self.media_type, str):
            raise TypeError("media_type must be a string")
        if not self.media_type.strip():
            raise ValueError("media_type must not be blank")
        supported_media_types = {"image/jpeg", "image/png"}
        if self.media_type not in supported_media_types:
            raise ValueError("media_type must be one of: image/jpeg, image/png")
        if not isinstance(self.actual_sha256, str):
            raise TypeError("actual_sha256 must be a string")
        if len(self.actual_sha256) != 64:
            raise ValueError("actual_sha256 must be exactly 64 characters long")
        allowed_characters = "0123456789abcdef"
        if not all(
            character in allowed_characters for character in self.actual_sha256
        ):
            raise ValueError(
                "actual_sha256 must contain only lowercase hexadecimal characters"
            )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: str
    prompt: str
    prompt_version: str
    prompt_sha256: str
    response_schema: Mapping[str, object]
    schema_version: str
    schema_sha256: str
    image_detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise TypeError("model must be a string")
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        if not isinstance(self.prompt_version, str):
            raise TypeError("prompt_version must be a string")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version must not be blank")
        if not isinstance(self.prompt_sha256, str):
            raise TypeError("prompt_sha256 must be a string")
        if len(self.prompt_sha256) != 64:
            raise ValueError("prompt_sha256 must be exactly 64 characters long")
        allowed_characters = "0123456789abcdef"
        if not all(
            character in allowed_characters for character in self.prompt_sha256
        ):
            raise ValueError(
                "prompt_sha256 must contain only lowercase hexadecimal characters"
            )
        if not isinstance(self.response_schema, Mapping):
            raise TypeError("response_schema must be a Mapping")
        if not self.response_schema:
            raise ValueError("response_schema must not be empty")
        if not isinstance(self.schema_version, str):
            raise TypeError("schema_version must be a string")
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be blank")
        if not isinstance(self.schema_sha256, str):
            raise TypeError("schema_sha256 must be a string")
        if len(self.schema_sha256) != 64:
            raise ValueError("schema_sha256 must be exactly 64 characters long")
        if not all(
            character in allowed_characters for character in self.schema_sha256
        ):
            raise ValueError(
                "schema_sha256 must contain only lowercase hexadecimal characters"
            )
        if not isinstance(self.image_detail, str):
            raise TypeError("image_detail must be a string")
        if not self.image_detail.strip():
            raise ValueError("image_detail must not be blank")
        if self.image_detail not in SUPPORTED_IMAGE_DETAILS:
            raise ValueError("image_detail must be one of: auto, high, low")


@dataclass(frozen=True, slots=True)
class AnalysisResponse:
    response_id: str
    response_model: str
    output_text: str
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.response_id, str):
            raise TypeError("response_id must be a string")
        if not self.response_id.strip():
            raise ValueError("response_id must not be blank")
        if not isinstance(self.response_model, str):
            raise TypeError("response_model must be a string")
        if not self.response_model.strip():
            raise ValueError("response_model must not be blank")
        if not isinstance(self.output_text, str):
            raise TypeError("output_text must be a string")
        if not self.output_text.strip():
            raise ValueError("output_text must not be blank")
        if type(self.input_tokens) is not int:
            raise TypeError("input_tokens must be an integer")
        if self.input_tokens < 0:
            raise ValueError("input_tokens must not be negative")
        if type(self.output_tokens) is not int:
            raise TypeError("output_tokens must be an integer")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must not be negative")
