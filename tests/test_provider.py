import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple, Never

import httpx2
import openai
import pytest

from software_update_verification.models import (
    AnalysisFailure,
    AnalysisResponse,
    ExperimentConfig,
    LoadedImage,
    Sample,
)
from software_update_verification.provider import OpenAIImageAnalyzer


REQUEST = httpx2.Request(
    "POST",
    "https://api.openai.com/v1/responses",
)
REQUEST_ID = "req_test_001"


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


def _make_loaded_image(
    content: bytes = b"test",
    **overrides: object,
) -> LoadedImage:
    values = {
        "content": content,
        "media_type": "image/png",
        "actual_sha256": hashlib.sha256(content).hexdigest(),
    }
    values.update(overrides)
    return LoadedImage(**values)


class FakeUsage(NamedTuple):
    input_tokens: int
    output_tokens: int


class FakeError(NamedTuple):
    message: str
    code: str | None = None


class FakeIncompleteDetails(NamedTuple):
    reason: str


class FakeRefusal(NamedTuple):
    type: str
    refusal: str


class FakeOutputMessage(NamedTuple):
    type: str
    content: list[FakeRefusal]


class FakeResponse(NamedTuple):
    id: str
    model: str
    status: str | None = "completed"
    error: FakeError | None = None
    incomplete_details: FakeIncompleteDetails | None = None
    output: list[FakeOutputMessage] | None = None
    output_text: str | None = None
    usage: FakeUsage | None = None


class FakeResponses:
    def __init__(self, fake_response: FakeResponse) -> None:
        self.fake_response = fake_response
        self.called_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> FakeResponse:
        self.called_kwargs = kwargs
        return self.fake_response


class RaisingFakeResponses:
    def __init__(self, exception: Exception) -> None:
        self.exception = exception
        self.called_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> Never:
        self.called_kwargs = kwargs
        raise self.exception


class FakeOpenAIClient:
    def __init__(self, responses: FakeResponses | RaisingFakeResponses) -> None:
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
        output=[],
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
        output=[],
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


@pytest.mark.parametrize(
    "invalid_loaded_image",
    [
        None,
        "something",
        123,
        13.3,
        True,
        Sample(
            sample_id="H002",
            image_path=Path("some_file.png"),
            image_sha256="a" * 64,
        ),
    ],
)
def test_analyze_rejects_non_loaded_image(
    invalid_loaded_image: object,
) -> None:
    experiment_config = _make_experiment_config()
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

    with pytest.raises(
        TypeError,
        match="loaded_image must be a LoadedImage",
    ):
        # noinspection PyTypeChecker
        analyzer.analyze(invalid_loaded_image, experiment_config)

    assert fake_responses.called_kwargs is None


@pytest.mark.parametrize(
    "invalid_experiment_config",
    [
        None,
        "something",
        123,
        13.3,
        True,
        Sample(
            sample_id="H002",
            image_path=Path("some_file.png"),
            image_sha256="a" * 64,
        ),
    ],
)
def test_analyze_rejects_non_experiment_config(
    invalid_experiment_config: object,
) -> None:
    content = b"test_bytes"
    loaded_image = LoadedImage(
        content=content,
        media_type="image/png",
        actual_sha256=hashlib.sha256(content).hexdigest(),
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

    with pytest.raises(
        TypeError,
        match="experiment_config must be an ExperimentConfig",
    ):
        # noinspection PyTypeChecker
        analyzer.analyze(loaded_image, invalid_experiment_config)

    assert fake_responses.called_kwargs is None


def test_analyze_returns_retryable_failure_when_sdk_times_out() -> None:
    experiment_config = _make_experiment_config()
    loaded_image = _make_loaded_image()

    request = httpx2.Request(
        "POST",
        "https://api.openai.com/v1/responses",
    )
    timeout_error = openai.APITimeoutError(request=request)
    fake_responses = RaisingFakeResponses(timeout_error)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    result = analyzer.analyze(loaded_image, experiment_config)

    expected_analysis_failure = {
        "failure_code": "timeout",
        "diagnostic_message": str(timeout_error),
        "retryable": True,
        "request_id": None,
        "status_code": None,
        "response_status": None,
        "response_id": None,
    }

    assert isinstance(result, AnalysisFailure)
    assert asdict(result) == expected_analysis_failure

def test_analyze_returns_retryable_failure_when_sdk_connection_fails() -> None:
    experiment_config = _make_experiment_config()
    loaded_image = _make_loaded_image()

    request = httpx2.Request(
        "POST",
        "https://api.openai.com/v1/responses",
    )
    connection_error = openai.APIConnectionError(request=request)
    fake_responses = RaisingFakeResponses(connection_error)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    result = analyzer.analyze(loaded_image, experiment_config)

    expected_analysis_failure = {
        "failure_code": "connection_error",
        "diagnostic_message": str(connection_error),
        "retryable": True,
        "request_id": None,
        "status_code": None,
        "response_status": None,
        "response_id": None,
    }

    assert isinstance(result, AnalysisFailure)
    assert asdict(result) == expected_analysis_failure


@pytest.mark.parametrize(
    ("exception_type", "failure_code", "status_code", "message", "retryable"),
    [
        (openai.BadRequestError, "bad_request", 400, "Invalid request", False),
        (
            openai.AuthenticationError,
            "authentication_error",
            401,
            "Invalid API key",
            False,
        ),
        (
            openai.PermissionDeniedError,
            "permission_denied",
            403,
            "Permission denied",
            False,
        ),
        (openai.NotFoundError, "not_found", 404, "Resource not found", False),
        (openai.APIStatusError, "request_timeout", 408, "Request timed out", True),
        (openai.ConflictError, "conflict", 409, "Request conflict", True),
        (
            openai.UnprocessableEntityError,
            "unprocessable_entity",
            422,
            "Request could not be processed",
            False,
        ),
        (openai.RateLimitError, "rate_limit", 429, "Rate limit exceeded", True),
    ],
)
def test_analyze_returns_expected_failure_for_mapped_http_error(
    exception_type: type[openai.APIStatusError],
    failure_code: str,
    status_code: int,
    message: str,
    retryable: bool,
) -> None:
    experiment_config = _make_experiment_config()
    loaded_image = _make_loaded_image()

    response = httpx2.Response(
        status_code,
        request=REQUEST,
        headers={"x-request-id": REQUEST_ID},
    )
    sdk_error = exception_type(
        response=response,
        body=None,
        message=message,
    )
    fake_responses = RaisingFakeResponses(sdk_error)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    result = analyzer.analyze(loaded_image, experiment_config)

    expected_analysis_failure = {
        "failure_code": failure_code,
        "diagnostic_message": str(sdk_error),
        "retryable": retryable,
        "request_id": REQUEST_ID,
        "status_code": status_code,
        "response_status": None,
        "response_id": None,
    }

    assert isinstance(result, AnalysisFailure)
    assert asdict(result) == expected_analysis_failure


@pytest.mark.parametrize(
    "status_code",
    [500, 502, 503, 599],
)
def test_analyze_returns_retryable_failure_for_server_error(
    status_code: int,
) -> None:
    experiment_config = _make_experiment_config()
    loaded_image = _make_loaded_image()

    response = httpx2.Response(
        status_code,
        request=REQUEST,
        headers={"x-request-id": REQUEST_ID},
    )
    sdk_error = openai.InternalServerError(
        response=response,
        body=None,
        message="Internal server error",
    )
    fake_responses = RaisingFakeResponses(sdk_error)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    result = analyzer.analyze(loaded_image, experiment_config)

    expected_analysis_failure = {
        "failure_code": "internal_server_error",
        "diagnostic_message": str(sdk_error),
        "retryable": True,
        "request_id": REQUEST_ID,
        "status_code": status_code,
        "response_status": None,
        "response_id": None,
    }

    assert isinstance(result, AnalysisFailure)
    assert asdict(result) == expected_analysis_failure


def test_analyze_returns_fallback_failure_for_unknown_http_status() -> None:
    experiment_config = _make_experiment_config()
    loaded_image = _make_loaded_image()

    response = httpx2.Response(
        418,
        request=REQUEST,
        headers={"x-request-id": REQUEST_ID},
    )
    sdk_error = openai.APIStatusError(
        response=response,
        body=None,
        message="API status error",
    )
    fake_responses = RaisingFakeResponses(sdk_error)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    result = analyzer.analyze(loaded_image, experiment_config)

    expected_analysis_failure = {
        "failure_code": "api_status_error",
        "diagnostic_message": str(sdk_error),
        "retryable": False,
        "request_id": REQUEST_ID,
        "status_code": 418,
        "response_status": None,
        "response_id": None,
    }

    assert isinstance(result, AnalysisFailure)
    assert asdict(result) == expected_analysis_failure


def test_analyze_returns_failure_when_response_status_is_failed() -> None:
    experiment_config = _make_experiment_config()
    loaded_image = _make_loaded_image()

    error = FakeError(
        message="invalid_prompt",
    )

    fake_response = FakeResponse(
        id=RESPONSE_ID,
        model=MODEL,
        status="failed",
        error=error,
        output_text=None,
        usage=None,
    )

    fake_responses = FakeResponses(fake_response)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    result = analyzer.analyze(loaded_image, experiment_config)

    expected_analysis_failure = {
        "failure_code": "response_failed",
        "diagnostic_message": error.message,
        "retryable": False,
        "request_id": None,
        "status_code": None,
        "response_status": "failed",
        "response_id": RESPONSE_ID,
    }

    assert isinstance(result, AnalysisFailure)
    assert asdict(result) == expected_analysis_failure


def test_analyze_returns_failure_when_response_is_incomplete() -> None:
    experiment_config = _make_experiment_config()
    loaded_image = _make_loaded_image()

    incomplete_details = FakeIncompleteDetails(
        reason="max_output_tokens",
    )
    fake_response = FakeResponse(
        id=RESPONSE_ID,
        model=MODEL,
        status="incomplete",
        incomplete_details=incomplete_details,
        output_text=None,
        usage=None,
    )

    fake_responses = FakeResponses(fake_response)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    result = analyzer.analyze(loaded_image, experiment_config)

    expected_analysis_failure = {
        "failure_code": "response_incomplete",
        "diagnostic_message": (
            f"Response is incomplete: {incomplete_details.reason}"
        ),
        "retryable": False,
        "request_id": None,
        "status_code": None,
        "response_status": "incomplete",
        "response_id": RESPONSE_ID,
    }

    assert isinstance(result, AnalysisFailure)
    assert asdict(result) == expected_analysis_failure


@pytest.mark.parametrize(
    ("response_status", "failure_code", "message", "retryable"),
    [
        ("queued", "response_queued", "Response is queued", False),
        (
            "in_progress",
            "response_in_progress",
            "Response is still in progress",
            False,
        ),
        ("cancelled", "response_cancelled", "Response was cancelled", False),
    ],
)
def test_analyze_returns_expected_failure_for_static_response_status(
    response_status: str,
    failure_code: str,
    message: str,
    retryable: bool,
) -> None:
    experiment_config = _make_experiment_config()
    loaded_image = _make_loaded_image()

    fake_response = FakeResponse(
        id=RESPONSE_ID,
        model=MODEL,
        status=response_status,
        error=None,
        output_text=None,
        usage=None,
    )
    fake_responses = FakeResponses(fake_response)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    result = analyzer.analyze(loaded_image, experiment_config)

    expected_analysis_failure = {
        "failure_code": failure_code,
        "diagnostic_message": message,
        "retryable": retryable,
        "request_id": None,
        "status_code": None,
        "response_status": response_status,
        "response_id": RESPONSE_ID,
    }

    assert isinstance(result, AnalysisFailure)
    assert asdict(result) == expected_analysis_failure


@pytest.mark.parametrize(
    "unknown_response_status",
    [None, "unknown_status"],
)
def test_analyze_returns_fallback_failure_for_unknown_response_status(
    unknown_response_status: str | None,
) -> None:
    experiment_config = _make_experiment_config()
    loaded_image = _make_loaded_image()

    fake_response = FakeResponse(
        id=RESPONSE_ID,
        model=MODEL,
        status=unknown_response_status,
        error=None,
        output_text=None,
        usage=None,
    )
    fake_responses = FakeResponses(fake_response)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    result = analyzer.analyze(loaded_image, experiment_config)

    expected_analysis_failure = {
        "failure_code": "response_unknown_status",
        "diagnostic_message": (
            f"Unexpected response status: {unknown_response_status}"
        ),
        "retryable": False,
        "request_id": None,
        "status_code": None,
        "response_status": unknown_response_status,
        "response_id": RESPONSE_ID,
    }

    assert isinstance(result, AnalysisFailure)
    assert asdict(result) == expected_analysis_failure


def test_analyze_returns_failure_when_model_refuses() -> None:
    experiment_config = _make_experiment_config()
    loaded_image = _make_loaded_image()

    refusal = FakeRefusal(
        type="refusal",
        refusal="I'm sorry, I cannot assist with that request.",
    )
    message = FakeOutputMessage(
        type="message",
        content=[refusal],
    )

    fake_response = FakeResponse(
        id=RESPONSE_ID,
        model=MODEL,
        status="completed",
        error=None,
        output=[message],
        output_text="",
        usage=None,
    )

    fake_responses = FakeResponses(fake_response)
    fake_client = FakeOpenAIClient(fake_responses)
    analyzer = OpenAIImageAnalyzer(fake_client)

    result = analyzer.analyze(loaded_image, experiment_config)

    expected_analysis_failure = {
        "failure_code": "model_refusal",
        "diagnostic_message": refusal.refusal,
        "retryable": False,
        "request_id": None,
        "status_code": None,
        "response_status": "completed",
        "response_id": RESPONSE_ID,
    }

    assert isinstance(result, AnalysisFailure)
    assert asdict(result) == expected_analysis_failure
