import json
from collections.abc import Mapping


def parse_verification_output(output_text: str) -> Mapping[str, object]:
    try:
        output_text_mapping = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise ValueError("output_text must contain valid JSON") from error
    if not isinstance(output_text_mapping, Mapping):
        raise ValueError("output_text must contain a JSON object")

    return output_text_mapping
