import base64

import openai

from software_update_verification.models import (
    AnalysisFailure,
    AnalysisResponse,
    ExperimentConfig,
    LoadedImage,
)

HTTP_FAILURE_POLICIES = {
    400: ("bad_request", False),
    401: ("authentication_error", False),
    403: ("permission_denied", False),
    404: ("not_found", False),
    408: ("request_timeout", True),
    409: ("conflict", True),
    422: ("unprocessable_entity", False),
    429: ("rate_limit", True),
}

STATIC_RESPONSE_FAILURE_POLICIES = {
    "queued": ("response_queued", "Response is queued", False),
    "in_progress": (
        "response_in_progress",
        "Response is still in progress",
        False,
    ),
    "cancelled": ("response_cancelled", "Response was cancelled", False),
}


def _resolve_http_failure_policy(status_code: int) -> tuple[str, bool]:
    if 500 <= status_code <= 599:
        return ("internal_server_error", True)
    return HTTP_FAILURE_POLICIES.get(status_code, ("api_status_error", False))


def _resolve_response_failure_policy(
    response_status: str | None,
) -> tuple[str, str, bool]:
    return STATIC_RESPONSE_FAILURE_POLICIES.get(
        response_status,
        (
            "response_unknown_status",
            f"Unexpected response status: {response_status}",
            False,
        ),
    )


class OpenAIImageAnalyzer:
    def __init__(self, client) -> None:
        self.client = client

    def analyze(
        self,
        loaded_image: LoadedImage,
        experiment_config: ExperimentConfig,
    ) -> AnalysisResponse | AnalysisFailure:
        if not isinstance(loaded_image, LoadedImage):
            raise TypeError("loaded_image must be a LoadedImage")
        if not isinstance(experiment_config, ExperimentConfig):
            raise TypeError("experiment_config must be an ExperimentConfig")
        model = experiment_config.model
        instructions = experiment_config.prompt
        encoded_image_bytes = base64.b64encode(loaded_image.content)
        encoded_image_text = encoded_image_bytes.decode("ascii")
        data_url = f"data:{loaded_image.media_type};base64,{encoded_image_text}"
        request_input = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": experiment_config.image_detail,
                    },
                ],
            }
        ]
        text_config = {
            "format": {
                "type": "json_schema",
                "name": "software_update_verification",
                "strict": True,
                "schema": experiment_config.response_schema,
            },
        }
        try:
            sdk_response = self.client.responses.create(
                model=model,
                instructions=instructions,
                input=request_input,
                text=text_config,
                store=False,
            )
        except openai.APITimeoutError as error:
            return AnalysisFailure(
                failure_code="timeout",
                diagnostic_message=str(error),
                retryable=True,
            )
        except openai.APIConnectionError as error:
            return AnalysisFailure(
                failure_code="connection_error",
                diagnostic_message=str(error),
                retryable=True,
            )
        except openai.APIStatusError as error:
            status_code = error.status_code
            failure_code, retryable = _resolve_http_failure_policy(status_code)
            return AnalysisFailure(
                failure_code=failure_code,
                diagnostic_message=str(error),
                retryable=retryable,
                request_id=error.request_id,
                status_code=status_code,
            )
        if sdk_response.status == "failed":
            diagnostic_message = sdk_response.error.message
            return AnalysisFailure(
                failure_code="response_failed",
                diagnostic_message=diagnostic_message,
                retryable=False,
                response_status=sdk_response.status,
                response_id=sdk_response.id,
            )
        if sdk_response.status == "incomplete":
            incomplete_details = sdk_response.incomplete_details.reason
            return AnalysisFailure(
                failure_code="response_incomplete",
                diagnostic_message=(
                    f"Response is incomplete: {incomplete_details}"
                ),
                retryable=False,
                response_status=sdk_response.status,
                response_id=sdk_response.id,
            )
        if sdk_response.status != "completed":
            (
                failure_code,
                diagnostic_message,
                retryable,
            ) = _resolve_response_failure_policy(sdk_response.status)

            return AnalysisFailure(
                failure_code=failure_code,
                diagnostic_message=diagnostic_message,
                retryable=retryable,
                response_status=sdk_response.status,
                response_id=sdk_response.id,
            )
        for output_item in sdk_response.output:
            if output_item.type == "message":
                for content_item in output_item.content:
                    if content_item.type == "refusal":
                        return AnalysisFailure(
                            failure_code="model_refusal",
                            diagnostic_message=content_item.refusal,
                            retryable=False,
                            response_status=sdk_response.status,
                            response_id=sdk_response.id,
                        )

        return AnalysisResponse(
            response_id=sdk_response.id,
            response_model=sdk_response.model,
            output_text=sdk_response.output_text,
            input_tokens=sdk_response.usage.input_tokens,
            output_tokens=sdk_response.usage.output_tokens,
        )
