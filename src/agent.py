"""End-to-End AI Customer Support Agent.

Integrates:
  1. Intent Classification (trained TF-IDF + LogReg model)
  2. Historically Grounded Retrieval (TF-IDF k-NN over AmazonHelp train.csv)
  3. Grounded Reply Formulation (nearest-neighbor template or optional Gemini synthesis)
  4. Auto-Handle vs. Human Escalation (deterministic policy with explicit reasons)

Runs 100% offline by default without requiring an API key.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import ROOT
from src.escalation import EscalationDecision, EscalationRouter
from src.preprocessing import clean_text
from src.retrieval import HistoricalReplyRetriever, RetrievalResult

MODEL_PATH = ROOT / "models" / "baseline_tfidf_logreg.joblib"


@dataclass
class AgentOutput:
    customer_message: str
    cleaned_message: str
    predicted_intent: str
    intent_confidence: float
    grounded_reply: str
    decision: str  # AUTO_HANDLED or HUMAN_ESCALATION
    escalation_reason: str
    triggered_rule: str
    top_historical_similarity: float
    retrieved_contexts: list[dict[str, Any]]
    generation_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SupportAgent:
    """Unified AmazonHelp customer support agent."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        retriever: HistoricalReplyRetriever | None = None,
        router: EscalationRouter | None = None,
    ) -> None:
        self.model_path = Path(model_path or MODEL_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Run `py -m src.train_baseline` first."
            )
        self.model = joblib.load(self.model_path)
        self.retriever = retriever or HistoricalReplyRetriever()
        self.router = router or EscalationRouter()

    def classify_intent(self, text: str) -> tuple[str, float]:
        """Predict intent and return (predicted_intent, confidence_score)."""
        cleaned = clean_text(text)
        if not cleaned:
            return "other", 0.0

        probs = self.model.predict_proba([cleaned])[0]
        classes = self.model.classes_
        top_idx = int(probs.argmax())
        predicted_intent = str(classes[top_idx])
        confidence = float(probs[top_idx])
        return predicted_intent, round(confidence, 4)

    def process_message(
        self,
        message: str,
        top_k_retrieval: int = 3,
        use_llm: bool = False,
    ) -> AgentOutput:
        """Process an incoming customer message end-to-end."""
        cleaned = clean_text(message)
        predicted_intent, confidence = self.classify_intent(cleaned)

        # 2. Retrieve historically grounded responses
        retrieval_results = self.retriever.retrieve(
            query=cleaned,
            top_k=top_k_retrieval,
            intent_filter=predicted_intent if predicted_intent != "other" else None,
        )

        max_sim = retrieval_results[0].similarity if retrieval_results else 0.0

        # 3. Draft reply
        generation_source = "historical_retrieval"
        grounded_reply = ""

        if retrieval_results:
            grounded_reply = retrieval_results[0].brand_reply_grounded
        else:
            grounded_reply = (
                "Please reach out to our customer support team directly with your order details "
                "so we can assist you promptly."
            )

        # 4. Optional Gemini LLM synthesis if requested and key available
        if use_llm and retrieval_results:
            try:
                from src.llm_labeler import gemini_api_key_from_env
                api_key = gemini_api_key_from_env()
                if api_key:
                    # Synthesize reply grounded in retrieved context
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    llm = genai.GenerativeModel("gemini-2.5-flash")
                    ctx_text = "\n".join(
                        f"Example {i+1}: {r.brand_reply_grounded}"
                        for i, r in enumerate(retrieval_results[:2])
                    )
                    prompt = (
                        "You are an AmazonHelp customer support assistant on Twitter. "
                        "Draft a single, helpful, polite tweet (under 200 chars) replying to the customer message, "
                        "grounded strictly on the style and content of these historical brand responses:\n"
                        f"{ctx_text}\n\n"
                        f"Customer Message: {message}\n"
                        "Reply tweet:"
                    )
                    resp = llm.generate_content(prompt)
                    if resp.text:
                        grounded_reply = resp.text.strip()
                        generation_source = "gemini_llm"
            except Exception:
                # Graceful fallback to offline nearest-neighbor template
                pass

        # 5. Escalation decision
        decision: EscalationDecision = self.router.evaluate(
            customer_message=message,
            predicted_intent=predicted_intent,
            confidence=confidence,
            max_retrieval_similarity=max_sim,
        )

        contexts = [
            {
                "similarity": r.similarity,
                "intent": r.intent,
                "brand_reply": r.brand_reply_grounded,
                "historical_inquiry": r.historical_customer_message,
            }
            for r in retrieval_results
        ]

        return AgentOutput(
            customer_message=message,
            cleaned_message=cleaned,
            predicted_intent=predicted_intent,
            intent_confidence=confidence,
            grounded_reply=grounded_reply,
            decision=decision.decision,
            escalation_reason=decision.reason,
            triggered_rule=decision.triggered_rule,
            top_historical_similarity=max_sim,
            retrieved_contexts=contexts,
            generation_source=generation_source,
        )


def batch_evaluate_agent(
    agent: SupportAgent,
    csv_path: Path,
    text_col: str = "customer_message",
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run agent in batch over a dataset and compute operational metrics."""
    df = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
    if limit:
        df = df.head(limit)

    outputs = []
    for _, row in df.iterrows():
        msg = str(row[text_col])
        res = agent.process_message(msg)
        outputs.append(res.to_dict())

    res_df = pd.DataFrame(outputs)

    auto_count = int((res_df["decision"] == "AUTO_HANDLED").sum())
    escalated_count = int((res_df["decision"] == "HUMAN_ESCALATION").sum())
    total = len(res_df)

    rule_breakdown = res_df["triggered_rule"].value_counts().to_dict()

    summary = {
        "total_evaluated": total,
        "auto_handled_count": auto_count,
        "auto_handled_pct": round((auto_count / total) * 100, 2) if total > 0 else 0.0,
        "escalated_count": escalated_count,
        "escalated_pct": round((escalated_count / total) * 100, 2) if total > 0 else 0.0,
        "avg_confidence": round(float(res_df["intent_confidence"].mean()), 4),
        "avg_similarity": round(float(res_df["top_historical_similarity"].mean()), 4),
        "triggered_rules": rule_breakdown,
    }

    return res_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="AmazonHelp AI Support Agent CLI")
    parser.add_argument("--message", type=str, help="Single customer message to process")
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to CSV for batch evaluation (e.g. evaluation/golden_set.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/agent_evaluation.json",
        help="Output path for batch metrics",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit batch evaluation rows")
    parser.add_argument("--use-llm", action="store_true", help="Enable live Gemini reply synthesis")
    args = parser.parse_args()

    agent = SupportAgent()

    if args.message:
        res = agent.process_message(args.message, use_llm=args.use_llm)
        print("\n" + "=" * 60)
        print("AI SUPPORT AGENT INFERENCE RESULT")
        print("=" * 60)
        print(f"Customer Message : {res.customer_message}")
        print(f"Predicted Intent : {res.predicted_intent} (Confidence: {res.intent_confidence:.1%})")
        print(f"Top Hist. Sim    : {res.top_historical_similarity:.2f}")
        print(f"Routing Decision : [{res.decision}]")
        print(f"Routing Reason   : {res.escalation_reason}")
        print(f"Reply Source     : {res.generation_source}")
        print("-" * 60)
        print(f"Grounded Reply   :\n\"{res.grounded_reply}\"")
        print("=" * 60 + "\n")

    elif args.batch:
        batch_path = Path(args.batch)
        if not batch_path.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_path}")
        print(f"Running batch agent evaluation on {batch_path}...")
        res_df, summary = batch_evaluate_agent(agent, batch_path, limit=args.limit)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\n" + "=" * 60)
        print("BATCH AGENT EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Evaluated  : {summary['total_evaluated']}")
        print(f"Auto-Handled     : {summary['auto_handled_count']} ({summary['auto_handled_pct']}%)")
        print(f"Human-Escalated  : {summary['escalated_count']} ({summary['escalated_pct']}%)")
        print(f"Avg Confidence   : {summary['avg_confidence']:.1%}")
        print(f"Avg Grounding Sim: {summary['avg_similarity']:.2f}")
        print("\nTriggered Escalation Rules Breakdown:")
        for rule, count in summary["triggered_rules"].items():
            print(f"  - {rule:<25}: {count}")
        print(f"\nSummary report saved to: {out_path}")
        print("=" * 60 + "\n")
    else:
        # Interactive demo message
        demo_msg = "My package has not arrived yet and the delivery date was yesterday. What is going on?"
        print("No --message or --batch provided. Running default demo inquiry:")
        res = agent.process_message(demo_msg)
        print(f"\nMessage: {demo_msg}")
        print(f"Intent : {res.predicted_intent} ({res.intent_confidence:.1%})")
        print(f"Decision: {res.decision} ({res.escalation_reason})")
        print(f"Reply  : {res.grounded_reply}\n")


if __name__ == "__main__":
    main()
