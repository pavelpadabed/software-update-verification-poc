import hashlib
import json
from typing import NamedTuple

from software_update_verification.models import (
    AnalysisResponse,
    ExperimentConfig,
    LoadedImage,
)
from software_update_verification.provider import OpenAIImageAnalyzer


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


class FakeUsage(NamedTuple):
    input_tokens: int
    output_tokens: int


class FakeResponse(NamedTuple):
    id: str
    model: str
    output_text: str
    usage: FakeUsage


class FakeResponses:
    def __init__(self, fake_response: FakeResponse) -> None:
        self.fake_response = fake_response
        self.called_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> FakeResponse:
        self.called_kwargs = kwargs
        return self.fake_response


class FakeOpenAIClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


VALID_ANALYSIS_OUTPUT_TEXT = json.dumps(
    {
        "decision": "accept",
        "platform": "windows",
        "evidence_codes": ["explicit_up_to_date"],
        "reason": "The system is explicitly shown as up to date.",
    }
)

MODEL = "gpt-4o-mini-2024-07-18"
RESPONSE_ID = "resp_test_001"


def test_analyze_returns_analysis_response_from_successful_sdk_response() -> None:
    experiment_config = _make_experiment_config()
    content = b"test_bytes"
    actual_sha256 = hashlib.sha256(content).hexdigest()
    loaded_image = LoadedImage(
        content=content,
        media_type="image/png",
        actual_sha256=actual_sha256,
    )
    usage = FakeUsage(
        input_tokens=1250,
        output_tokens=75,
    )
    fake_response = FakeResponse(
        id=RESPONSE_ID,
        model=MODEL,
        output_text=VALID_ANALYSIS_OUTPUT_TEXT,
        usage=usage,
    )

    fake_responses = FakeResponses(fake_response)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    result = analyzer.analyze(loaded_image, experiment_config)

    assert isinstance(result, AnalysisResponse)
    assert result.response_id == fake_response.id
    assert result.response_model == fake_response.model
    assert result.output_text == fake_response.output_text
    assert result.input_tokens == fake_response.usage.input_tokens
    assert result.output_tokens == fake_response.usage.output_tokens


def test_analyze_sends_expected_request_to_openai() -> None:
    experiment_config = _make_experiment_config()
    content = b"test_bytes"
    actual_sha256 = hashlib.sha256(content).hexdigest()
    loaded_image = LoadedImage(
        content=content,
        media_type="image/png",
        actual_sha256=actual_sha256,
    )
    usage = FakeUsage(
        input_tokens=1250,
        output_tokens=75,
    )
    fake_response = FakeResponse(
        id=RESPONSE_ID,
        model=MODEL,
        output_text=VALID_ANALYSIS_OUTPUT_TEXT,
        usage=usage,
    )

    fake_responses = FakeResponses(fake_response)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    analyzer.analyze(loaded_image, experiment_config)
    expected_data_url = "data:image/png;base64,dGVzdF9ieXRlcw=="
    expected_kwargs = {
        "model": experiment_config.model,
        "instructions": experiment_config.prompt,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": expected_data_url,
                        "detail": experiment_config.image_detail,
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "software_update_verification",
                "strict": True,
                "schema": experiment_config.response_schema,
            }
        },
        "store": False,
    }

    assert fake_responses.called_kwargs is not None
    assert fake_responses.called_kwargs == expected_kwargs
