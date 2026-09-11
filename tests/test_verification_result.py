from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from software_update_verification.models import VerificationResult


def test_verification_result_is_created_with_valid_values() -> None:
    expected_payload = {
        "decision": "accept",
        "platform": "macos",
        "evidence_codes": ["explicit_up_to_date"],
        "reason": "The interface explicitly states that the Mac is up to date.",
    }

    result = VerificationResult(**expected_payload)

    assert result.model_dump() == expected_payload


def test_verification_result_rejects_extra_fields() -> None:
    payload_with_extra_field = {
        "decision": "accept",
        "platform": "macos",
        "evidence_codes": ["explicit_up_to_date"],
        "reason": "The interface explicitly states that the Mac is up to date.",
        "unexpected_field": "Hello",
    }

    with pytest.raises(ValidationError) as exc_info:
        VerificationResult(**payload_with_extra_field)

    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    ("missing_field", "payload_with_missing_field"),
    [
        (
            "decision",
            {
                "platform": "macos",
                "evidence_codes": ["explicit_up_to_date"],
                "reason": "The interface explicitly states that the Mac is up to date.",
            },
        ),
        (
            "platform",
            {
                "decision": "accept",
                "evidence_codes": ["explicit_up_to_date"],
                "reason": "The interface explicitly states that the Mac is up to date.",
            },
        ),
        (
            "evidence_codes",
            {
                "decision": "accept",
                "platform": "macos",
                "reason": "The interface explicitly states that the Mac is up to date.",
            },
        ),
        (
            "reason",
            {
                "decision": "accept",
                "platform": "macos",
                "evidence_codes": ["explicit_up_to_date"],
            },
        ),
    ],
)
def test_verification_result_rejects_missing_required_fields(
    missing_field: str,
    payload_with_missing_field: Mapping[str, object],
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        VerificationResult(**payload_with_missing_field)

    errors = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"] == (missing_field,)


def test_verification_result_rejects_unsupported_decision() -> None:
    payload_with_unsupported_decision = {
        "decision": "approve",
        "platform": "macos",
        "evidence_codes": ["explicit_up_to_date"],
        "reason": "The interface explicitly states that the Mac is up to date.",
    }

    with pytest.raises(ValidationError) as exc_info:
        VerificationResult(**payload_with_unsupported_decision)

    errors = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "literal_error"
    assert errors[0]["loc"] == ("decision",)


def test_verification_result_rejects_unsupported_platform() -> None:
    payload_with_unsupported_platform = {
        "decision": "accept",
        "platform": "my_super_os",
        "evidence_codes": ["explicit_up_to_date"],
        "reason": "The interface explicitly states that the Mac is up to date.",
    }

    with pytest.raises(ValidationError) as exc_info:
        VerificationResult(**payload_with_unsupported_platform)

    errors = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "literal_error"
    assert errors[0]["loc"] == ("platform",)


def test_verification_result_rejects_unsupported_evidence_code() -> None:
    payload_with_unsupported_evidence_codes = {
        "decision": "accept",
        "platform": "macos",
        "evidence_codes": ["some_code"],
        "reason": "The interface explicitly states that the Mac is up to date.",
    }

    with pytest.raises(ValidationError) as exc_info:
        VerificationResult(**payload_with_unsupported_evidence_codes)

    errors = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "literal_error"
    assert errors[0]["loc"] == ("evidence_codes", 0)


def test_verification_result_rejects_empty_evidence_codes() -> None:
    payload_with_empty_evidence_codes = {
        "decision": "accept",
        "platform": "macos",
        "evidence_codes": [],
        "reason": "The interface explicitly states that the Mac is up to date.",
    }

    with pytest.raises(ValidationError) as exc_info:
        VerificationResult(**payload_with_empty_evidence_codes)

    errors = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "too_short"
    assert errors[0]["loc"] == ("evidence_codes",)


def test_verification_result_rejects_non_string_reason() -> None:
    payload_with_non_string_reason = {
        "decision": "accept",
        "platform": "macos",
        "evidence_codes": ["explicit_up_to_date"],
        "reason": b"The interface explicitly states that the Mac is up to date.",
    }

    with pytest.raises(ValidationError) as exc_info:
        VerificationResult(**payload_with_non_string_reason)

    errors = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "string_type"
    assert errors[0]["loc"] == ("reason",)


@pytest.mark.parametrize(
    "blank_reason",
    ["", " ", "\t", "\n", "\t\n"],
)
def test_verification_result_rejects_blank_reason(
    blank_reason: str,
) -> None:
    payload_with_blank_reason = {
        "decision": "accept",
        "platform": "macos",
        "evidence_codes": ["explicit_up_to_date"],
        "reason": blank_reason,
    }

    with pytest.raises(ValidationError) as exc_info:
        VerificationResult(**payload_with_blank_reason)

    errors = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "value_error"
    assert errors[0]["loc"] == ("reason",)
