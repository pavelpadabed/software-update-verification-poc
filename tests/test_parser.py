import pytest

from software_update_verification.parser import parse_verification_output


def test_parse_verification_output_returns_mapping_for_valid_json_object() -> None:
    output_text = (
        '{"decision":"reject","platform":"windows",'
        '"evidence_codes":["mandatory_update_available"],'
        '"reason":"A required operating system update is visibly available."}'
    )

    expected_result = {
        "decision": "reject",
        "platform": "windows",
        "evidence_codes": ["mandatory_update_available"],
        "reason": "A required operating system update is visibly available.",
    }

    actual_result = parse_verification_output(output_text)

    assert actual_result == expected_result


def test_parse_verification_output_raises_value_error_for_malformed_json() -> None:
    invalid_output_text = '{"decision: "reject,'

    with pytest.raises(
        ValueError,
        match="output_text must contain valid JSON",
    ):
        parse_verification_output(invalid_output_text)


def test_parse_verification_output_raises_value_error_when_json_root_is_not_object() -> None:
    non_mapping_output_text = '["reject", "windows"]'

    with pytest.raises(
        ValueError,
        match="output_text must contain a JSON object",
    ):
        parse_verification_output(non_mapping_output_text)
