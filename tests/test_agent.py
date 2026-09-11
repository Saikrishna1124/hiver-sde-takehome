"""Tests for Historical Reply Retrieval, Escalation Router, and SupportAgent."""

from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd

from src.agent import AgentOutput, SupportAgent, batch_evaluate_agent
from src.escalation import EscalationDecision, EscalationRouter
from src.retrieval import HistoricalReplyRetriever, clean_brand_reply


def test_clean_brand_reply():
    assert clean_brand_reply("@115850 Please reach out to us.") == "Please reach out to us."
    assert clean_brand_reply("@user1 @user2 Hello there!") == "Hello there!"
    assert clean_brand_reply("No mentions here.") == "No mentions here."
    assert clean_brand_reply("") == ""


def test_retriever_initialization_and_search():
    retriever = HistoricalReplyRetriever()
    assert len(retriever.df) > 0

    results = retriever.retrieve("Where is my package and delivery tracking?", top_k=3)
    assert len(results) == 3
    assert results[0].similarity >= results[1].similarity >= results[2].similarity
    assert all(len(r.brand_reply_grounded) > 0 for r in results)
    assert all(0.0 <= r.similarity <= 1.0 for r in results)


def test_retriever_empty_query():
    retriever = HistoricalReplyRetriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retriever_intent_filter():
    retriever = HistoricalReplyRetriever()
    results = retriever.retrieve("order refund", top_k=2, intent_filter="delivery_order_status")
    for r in results:
        # If intent matches exist, proposed_intent should match filter
        assert r.intent == "delivery_order_status"


def test_escalation_router_urgent_keywords():
    router = EscalationRouter()
    decision = router.evaluate(
        customer_message="I suspect fraud on my account, get me an agent",
        predicted_intent="delivery_order_status",
        confidence=0.90,
        max_retrieval_similarity=0.80,
    )
    assert decision.decision == "HUMAN_ESCALATION"
    assert decision.triggered_rule == "urgent_keyword"
    assert "agent" in decision.reason or "fraud" in decision.reason


def test_escalation_router_sensitive_intent():
    router = EscalationRouter()
    for sensitive in ["account_access", "payment_billing", "service_complaint"]:
        decision = router.evaluate(
            customer_message="Routine issue with no bad words",
            predicted_intent=sensitive,
            confidence=0.95,
            max_retrieval_similarity=0.75,
        )
        assert decision.decision == "HUMAN_ESCALATION"
        assert decision.triggered_rule == "sensitive_intent"


def test_escalation_router_fallback_other():
    router = EscalationRouter()
    decision = router.evaluate(
        customer_message="random gibberish text",
        predicted_intent="other",
        confidence=0.85,
        max_retrieval_similarity=0.60,
    )
    assert decision.decision == "HUMAN_ESCALATION"
    assert decision.triggered_rule == "fallback_intent"


def test_escalation_router_low_confidence():
    router = EscalationRouter(confidence_threshold=0.45)
    decision = router.evaluate(
        customer_message="my order was late yesterday",
        predicted_intent="delivery_order_status",
        confidence=0.30,
        max_retrieval_similarity=0.60,
    )
    assert decision.decision == "HUMAN_ESCALATION"
    assert decision.triggered_rule == "low_confidence"


def test_escalation_router_low_grounding():
    router = EscalationRouter(retrieval_threshold=0.30)
    decision = router.evaluate(
        customer_message="my order was late yesterday",
        predicted_intent="delivery_order_status",
        confidence=0.80,
        max_retrieval_similarity=0.15,
    )
    assert decision.decision == "HUMAN_ESCALATION"
    assert decision.triggered_rule == "low_grounding"


def test_escalation_router_auto_handled():
    router = EscalationRouter(confidence_threshold=0.45, retrieval_threshold=0.30)
    decision = router.evaluate(
        customer_message="where is my tracking number for delivery",
        predicted_intent="delivery_order_status",
        confidence=0.82,
        max_retrieval_similarity=0.55,
    )
    assert decision.decision == "AUTO_HANDLED"
    assert decision.triggered_rule == "eligible_auto_handled"
    assert "High-confidence" in decision.reason


def test_support_agent_end_to_end():
    agent = SupportAgent()
    out = agent.process_message("Tracking says delivered but package is missing from doorstep")
    assert isinstance(out, AgentOutput)
    assert out.predicted_intent == "delivery_order_status"
    assert out.intent_confidence > 0.40
    assert len(out.grounded_reply) > 0
    assert out.decision in ("AUTO_HANDLED", "HUMAN_ESCALATION")
    assert len(out.escalation_reason) > 0
    assert len(out.retrieved_contexts) > 0


def test_batch_evaluate_agent(tmp_path):
    agent = SupportAgent()
    # Create small test CSV
    test_df = pd.DataFrame([
        {"customer_message": "Where is my package tracking?"},
        {"customer_message": "Someone hacked my account call police!"},
        {"customer_message": "thank you for everything"},
    ])
    test_csv = tmp_path / "test_inquiries.csv"
    test_df.to_csv(test_csv, index=False)

    res_df, summary = batch_evaluate_agent(agent, test_csv)
    assert len(res_df) == 3
    assert summary["total_evaluated"] == 3
    assert summary["auto_handled_count"] + summary["escalated_count"] == 3
    assert "triggered_rules" in summary
