import pytest

from software_update_verification.models import VerificationResult
from software_update_verification.policy import validate_verification_policy


def _make_verification_result(**overrides: object) -> VerificationResult:
    data = {
        "decision": "accept",
        "platform": "macos",
        "evidence_codes": ["explicit_up_to_date"],
        "reason": "The interface explicitly states that the Mac is up to date.",
    }
    data.update(overrides)

    return VerificationResult(**data)


def test_validate_verification_policy_accepts_consistent_accept_result() -> None:
    verification_result = _make_verification_result()

    actual_result = validate_verification_policy(verification_result)

    assert actual_result is None


@pytest.mark.parametrize(
    ("evidence_code", "decision"),
    [
        ("explicit_up_to_date", "accept"),
        ("optional_update_only", "accept"),
        ("mandatory_update_available", "reject"),
        ("installation_in_progress", "reject"),
        ("restart_required", "reject"),
        ("installation_error", "reject"),
        ("irrelevant_image", "reject"),
        ("update_check_error", "manual_review"),
        ("system_info_only", "manual_review"),
        ("successful_update_only", "manual_review"),
        ("contradictory_information", "manual_review"),
        ("unreadable", "manual_review"),
        ("insufficient_evidence", "manual_review"),
    ],
)
def test_validate_verification_policy_accepts_each_supported_evidence_code(
    evidence_code: str,
    decision: str,
) -> None:
    verification_result = _make_verification_result(
        decision=decision,
        evidence_codes=[evidence_code],
    )

    actual_result = validate_verification_policy(verification_result)

    assert actual_result is None


def test_validate_verification_policy_rejects_accept_with_manual_review_evidence(
) -> None:
    verification_result = _make_verification_result(
        evidence_codes=[
            "explicit_up_to_date",
            "system_info_only",
        ]
    )

    expected_message = (
        "Actual decision: accept; "
        "expected decision: manual_review; "
        "evidence_codes: ['explicit_up_to_date', 'system_info_only']"
    )

    with pytest.raises(ValueError) as exc_info:
        validate_verification_policy(verification_result)

    assert str(exc_info.value) == expected_message


def test_validate_verification_policy_rejects_accept_for_reject_evidence(
) -> None:
    verification_result = _make_verification_result(
        evidence_codes=["mandatory_update_available"]
    )
    expected_message = (
        "Actual decision: accept; "
        "expected decision: reject; "
        "evidence_codes: ['mandatory_update_available']"
    )
    with pytest.raises(ValueError) as exc_info:
        validate_verification_policy(verification_result)

    assert str(exc_info.value) == expected_message


def test_validate_verification_policy_accepts_manual_review_for_mixed_evidence(
) -> None:
    verification_result = _make_verification_result(
        decision="manual_review",
        evidence_codes=[
            "explicit_up_to_date",
            "system_info_only",
        ]
    )

    actual_result = validate_verification_policy(verification_result)

    assert actual_result is None


def test_validate_verification_policy_accepts_manual_review_for_conflicting_accept_and_reject_evidence(
) -> None:
    verification_result = _make_verification_result(
        decision="manual_review",
        evidence_codes=[
            "explicit_up_to_date",
            "mandatory_update_available",
        ]
    )

    actual_result = validate_verification_policy(verification_result)

    assert actual_result is None
