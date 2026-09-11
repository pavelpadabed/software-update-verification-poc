from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from software_update_verification.config_builder import build_experiment_config
from software_update_verification.models import AnalysisFailure, VerificationResult
from software_update_verification.parser import parse_verification_output
from software_update_verification.policy import validate_verification_policy
from software_update_verification.provider import OpenAIImageAnalyzer
from software_update_verification.sample_io import (
    build_sample,
    iter_image_paths,
    load_image,
)

MODEL = "gpt-4.1-mini-2025-04-14"
PROMPT_VERSION = "v3"
SCHEMA_VERSION = "v1"
IMAGE_DETAIL = "low"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIRECTORY_PATH = PROJECT_ROOT / "smoke_test" / "images"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    client = OpenAI()

    analyzer = OpenAIImageAnalyzer(client)

    image_directory = IMAGE_DIRECTORY_PATH

    prompt_path = PROJECT_ROOT / "prompts" / "verification_v3.txt"
    schema_path = PROJECT_ROOT / "schemas" / "verification_response_v1.json"
    experiment_config = build_experiment_config(
        model=MODEL,
        prompt_path=prompt_path,
        prompt_version=PROMPT_VERSION,
        schema_path=schema_path,
        schema_version=SCHEMA_VERSION,
        image_detail=IMAGE_DETAIL,
    )

    input_tokens = 0
    output_tokens = 0

    for image_path in iter_image_paths(image_directory):
        try:
            image_path_name = image_path.name
            stage = "sample"
            sample = build_sample(image_path)
            stage = "load"
            loaded_image = load_image(sample)

            stage = "analyze"
            analysis_result = analyzer.analyze(loaded_image, experiment_config)

            if isinstance(analysis_result, AnalysisFailure):
                failure_code = analysis_result.failure_code
                print(f"{image_path_name}; failure_code: {failure_code}")
                continue

            output_text = analysis_result.output_text

            input_tokens += analysis_result.input_tokens
            output_tokens += analysis_result.output_tokens

            stage = "parse"
            output_text_mapping = parse_verification_output(output_text)

            stage = "validate"
            verification_result = VerificationResult.model_validate(output_text_mapping)

            stage = "policy"
            validate_verification_policy(verification_result)

            print(f"{image_path_name}; verification_result: {verification_result}")

        except (OSError, ValueError) as error:
            error_type = type(error).__name__
            print(
                f"{image_path.name}; "
                f"stage={stage}, "
                f"type={error_type}, "
                f"message={error}"
            )

    print(
        f"usage: input_tokens={input_tokens}, "
        f"output_tokens={output_tokens}"
    )


if __name__ == "__main__":
    main()
