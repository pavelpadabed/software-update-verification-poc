from software_update_verification.models import VerificationResult

EVIDENCE_DECISION_POLICY = {
    "explicit_up_to_date": "accept",
    "optional_update_only": "accept",
    "mandatory_update_available": "reject",
    "installation_in_progress": "reject",
    "restart_required": "reject",
    "installation_error": "reject",
    "irrelevant_image": "reject",
    "update_check_error": "manual_review",
    "system_info_only": "manual_review",
    "successful_update_only": "manual_review",
    "contradictory_information": "manual_review",
    "unreadable": "manual_review",
    "insufficient_evidence": "manual_review",
}


def validate_verification_policy(result: VerificationResult) -> None:
    policy_decisions = {
        EVIDENCE_DECISION_POLICY[evidence_code]
        for evidence_code in result.evidence_codes
    }

    expected_decision = "manual_review"

    if len(policy_decisions) == 1:
        expected_decision, = policy_decisions

    if expected_decision != result.decision:
        raise ValueError(
            f"Actual decision: {result.decision}; "
            f"expected decision: {expected_decision}; "
            f"evidence_codes: {result.evidence_codes}"
        )
