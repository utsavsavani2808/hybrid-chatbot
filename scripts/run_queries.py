"""Run the sample queries through the hybrid chatbot and save raw outputs.

Run from the repository root after installing requirements and configuring any
required model/API credentials:
    python scripts/run_queries.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.chatbot import hybrid_chatbot

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "test_queries.csv"
OUTPUT = ROOT / "outputs" / "run_results.csv"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with INPUT.open(newline="", encoding="utf-8-sig") as f:
        queries = list(csv.DictReader(f))

    rows = []
    for item in queries:
        result = hybrid_chatbot(item["user_query"])
        rows.append({
            "query_id": item.get("query_id", ""),
            "query_category": item.get("query_category", ""),
            "user_query": item["user_query"],
            "expected_answer": item.get("expected_answer", ""),
            "response": result.get("response", ""),
            "component_used": result.get("component_used", ""),
            "matched_intent_or_faq": result.get("matched_intent_or_faq", ""),
            "confidence_score": result.get("confidence_score", ""),
        })

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["query_id", "user_query", "response"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} raw response rows to {OUTPUT}")


if __name__ == "__main__":
    main()
