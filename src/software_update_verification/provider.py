import base64

from software_update_verification.models import (
    AnalysisResponse,
    ExperimentConfig,
    LoadedImage,
)


class OpenAIImageAnalyzer:
    def __init__(self, client) -> None:
        self.client = client

    def analyze(
        self,
        loaded_image: LoadedImage,
        experiment_config: ExperimentConfig,
    ) -> AnalysisResponse:
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
        sdk_response = self.client.responses.create(
            model=model,
            instructions=instructions,
            input=request_input,
            text=text_config,
            store=False,
        )

        return AnalysisResponse(
            response_id=sdk_response.id,
            response_model=sdk_response.model,
            output_text=sdk_response.output_text,
            input_tokens=sdk_response.usage.input_tokens,
            output_tokens=sdk_response.usage.output_tokens,
        )
