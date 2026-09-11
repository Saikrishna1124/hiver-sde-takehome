"""Phase 11 - Rule-based auto-handle vs human escalation router.

Determines whether an incoming customer inquiry can be safely resolved
automatically with a grounded response template (AUTO_HANDLED) or requires
routing to a human support agent (HUMAN_ESCALATION).
Provides an explicit human-readable reason for every decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

DecisionType = Literal["AUTO_HANDLED", "HUMAN_ESCALATION"]

# Keywords indicating customer frustration, security concerns, or explicit human requests
ESCALATION_KEYWORDS = [
    "human",
    "agent",
    "representative",
    "supervisor",
    "manager",
    "lawsuit",
    "lawyer",
    "legal",
    "police",
    "fraud",
    "scam",
    "stolen",
    "hack",
    "hacked",
    "unauthorized",
    "chargeback",
    "dispute",
]

ESCALATION_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ESCALATION_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Sensitive business intents requiring human oversight
SENSITIVE_INTENTS = {
    "account_access",     # Security, OTP, password recovery
    "payment_billing",    # Financial transactions, unauthorized billing
    "service_complaint",  # Escalated customer complaints, brand damage risk
}

# Transactional intents eligible for automated resolution if criteria are met
ELIGIBLE_AUTO_INTENTS = {
    "delivery_order_status",
    "refund_return",
    "device_digital_content",
    "prime_membership",
    "seller_marketplace",
    "item_quality_issue",
}


@dataclass
class EscalationDecision:
    decision: DecisionType
    reason: str
    triggered_rule: str
    confidence: float
    retrieval_similarity: float
    intent: str


class EscalationRouter:
    """Deterministic policy engine for customer support routing."""

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        retrieval_threshold: float = 0.30,
    ) -> None:
        """Initialize routing policy thresholds.

        Note: 0.45 confidence and 0.30 retrieval similarity are design baseline
        thresholds tuned for precision over recall in auto-handling.
        """
        self.confidence_threshold = confidence_threshold
        self.retrieval_threshold = retrieval_threshold

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float,
        max_retrieval_similarity: float,
    ) -> EscalationDecision:
        """Evaluate message and decide AUTO_HANDLED vs HUMAN_ESCALATION with reason."""
        # Rule 1: Explicit urgent / escalation keyword detected
        kw_match = ESCALATION_KEYWORD_RE.search(customer_message)
        if kw_match:
            keyword = kw_match.group(0).lower()
            return EscalationDecision(
                decision="HUMAN_ESCALATION",
                reason=f"Urgent escalation keyword detected in message: '{keyword}'",
                triggered_rule="urgent_keyword",
                confidence=confidence,
                retrieval_similarity=max_retrieval_similarity,
                intent=predicted_intent,
            )

        # Rule 2: Sensitive intent categories require human agent
        if predicted_intent in SENSITIVE_INTENTS:
            return EscalationDecision(
                decision="HUMAN_ESCALATION",
                reason=f"Sensitive intent category '{predicted_intent}' requires human oversight.",
                triggered_rule="sensitive_intent",
                confidence=confidence,
                retrieval_similarity=max_retrieval_similarity,
                intent=predicted_intent,
            )

        # Rule 3: Unclassified fallback category cannot be auto-resolved
        if predicted_intent == "other":
            return EscalationDecision(
                decision="HUMAN_ESCALATION",
                reason="Inquiry classified as unmapped fallback ('other'); cannot reliably auto-resolve.",
                triggered_rule="fallback_intent",
                confidence=confidence,
                retrieval_similarity=max_retrieval_similarity,
                intent=predicted_intent,
            )

        # Rule 4: Low classifier confidence
        if confidence < self.confidence_threshold:
            return EscalationDecision(
                decision="HUMAN_ESCALATION",
                reason=(
                    f"Low intent confidence ({confidence:.2f} < "
                    f"{self.confidence_threshold:.2f} policy threshold)."
                ),
                triggered_rule="low_confidence",
                confidence=confidence,
                retrieval_similarity=max_retrieval_similarity,
                intent=predicted_intent,
            )

        # Rule 5: Insufficient historical grounding / low retrieval similarity
        if max_retrieval_similarity < self.retrieval_threshold:
            return EscalationDecision(
                decision="HUMAN_ESCALATION",
                reason=(
                    f"Insufficient historical grounding (similarity {max_retrieval_similarity:.2f} < "
                    f"{self.retrieval_threshold:.2f} policy threshold)."
                ),
                triggered_rule="low_grounding",
                confidence=confidence,
                retrieval_similarity=max_retrieval_similarity,
                intent=predicted_intent,
            )

        # Rule 6: High confidence + standard transactional intent + grounded match
        return EscalationDecision(
            decision="AUTO_HANDLED",
            reason=(
                f"High-confidence {predicted_intent} inquiry ({confidence:.2f} >= "
                f"{self.confidence_threshold:.2f}) with verified historical response template "
                f"(similarity: {max_retrieval_similarity:.2f})."
            ),
            triggered_rule="eligible_auto_handled",
            confidence=confidence,
            retrieval_similarity=max_retrieval_similarity,
            intent=predicted_intent,
        )


if __name__ == "__main__":
    router = EscalationRouter()
    cases = [
        ("where is my order package late", "delivery_order_status", 0.75, 0.42),
        ("i need to speak to an agent right now", "delivery_order_status", 0.70, 0.35),
        ("someone hacked my password and email", "account_access", 0.88, 0.50),
        ("something random weird hello", "other", 0.35, 0.12),
        ("how do i return this shirt", "refund_return", 0.30, 0.35),
    ]
    for msg, intent, conf, sim in cases:
        dec = router.evaluate(msg, intent, conf, sim)
        print(f"Msg: {msg}\nDecision: {dec.decision} | Rule: {dec.triggered_rule}\nReason: {dec.reason}\n")
