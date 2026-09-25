"""Evaluation harness running reproducible benchmarks across Theme 04 scenarios."""

import uuid
from typing import Any

from avy.core.models import ControllerState
from avy.evaluation.dataset import EvalTestCase, get_standard_eval_dataset
from avy.evaluation.metrics import (
    TestResult,
    calculate_retrieval_recall,
    estimate_turn_cost,
)
from avy.streaming.engine import StreamingLiveRAGEngine


class EvaluationHarness:
    """Benchmark runner executing and evaluating AVY Streaming Live RAG."""

    def __init__(self, engine: StreamingLiveRAGEngine | None = None) -> None:
        self.engine = engine or StreamingLiveRAGEngine()

    def run_test_case(self, test: EvalTestCase) -> TestResult:
        """Execute a single test case through the streaming pipeline."""
        session_id = f"eval_{test.test_id}_{uuid.uuid4().hex[:6]}"
        actual_state = ControllerState.WAIT
        decomposed_queries: list[str] = []
        retrieved_docs: list[str] = []
        full_answer = []
        citations_found = []
        ttft_ms = 0.0
        total_latency_ms = 0.0
        last_evidence = []

        if test.is_multi_turn:
            # Execute turn 1 then turn 2
            for turn_idx, turn_text in enumerate(test.inputs):
                is_last_turn = (turn_idx == len(test.inputs) - 1)
                turn_answer = []
                for event in self.engine.process_transcript_stream(
                    transcript_chunk=turn_text,
                    session_id=session_id,
                    is_final_chunk=True,
                ):
                    etype = event.get("type")
                    if etype == "controller_decision":
                        actual_state = ControllerState(event.get("state"))
                    elif etype == "decomposition":
                        decomposed_queries = [
                            sq["subquery_text"]
                            for sq in event.get("decomposition", {}).get("subqueries", [])
                        ]
                    elif etype == "evidence":
                        ev_list = event.get("evidence", [])
                        last_evidence = ev_list
                        retrieved_docs = list({e["document_id"] for e in ev_list})
                    elif etype == "token":
                        turn_answer.append(event.get("token", ""))
                    elif etype == "citations":
                        citations_found = event.get("citations", [])
                    elif etype == "timings":
                        timings = event.get("timings", {})
                        if is_last_turn:
                            ttft_ms = timings.get("ttft_ms", 0.0)
                            total_latency_ms = timings.get("total_ms", 0.0)
                if is_last_turn:
                    full_answer = turn_answer
        else:
            # Sequential streaming chunks
            accumulated = ""
            for i, chunk in enumerate(test.inputs):
                accumulated = f"{accumulated} {chunk}".strip()
                is_final = (i == len(test.inputs) - 1)
                for event in self.engine.process_transcript_stream(
                    transcript_chunk=chunk,
                    session_id=session_id,
                    accumulated_transcript=accumulated,
                    is_final_chunk=is_final,
                ):
                    etype = event.get("type")
                    if etype == "controller_decision":
                        actual_state = ControllerState(event.get("state"))
                    elif etype == "decomposition":
                        decomposed_queries = [
                            sq["subquery_text"]
                            for sq in event.get("decomposition", {}).get("subqueries", [])
                        ]
                    elif etype == "evidence":
                        ev_list = event.get("evidence", [])
                        last_evidence = ev_list
                        retrieved_docs = list({e["document_id"] for e in ev_list})
                    elif etype == "token":
                        full_answer.append(event.get("token", ""))
                    elif etype == "citations":
                        citations_found = event.get("citations", [])
                    elif etype == "timings":
                        timings = event.get("timings", {})
                        ttft_ms = timings.get("ttft_ms", 0.0)
                        total_latency_ms = timings.get("total_ms", 0.0)

        # Validate decision correctness
        decision_correct = (actual_state == test.expected_controller_state)

        # Calculate Recall
        recall = calculate_retrieval_recall(retrieved_docs, test.expected_doc_ids)

        # Calculate Groundedness
        groundedness = 1.0
        answer_str = "".join(full_answer)
        if actual_state == ControllerState.RETRIEVE and last_evidence:
            # Reconstruct temporary evidence objects for score calculation
            from avy.retrieval.models import FusedEvidence

            ev_objs = [
                FusedEvidence(
                    chunk_id=e["chunk_id"],
                    document_id=e["document_id"],
                    source=e["source"],
                    title=e["title"],
                    text=e["text"],
                    rrf_score=e.get("rrf_score", 1.0),
                )
                for e in last_evidence
            ]
            groundedness = self.engine.grounding.calculate_groundedness_score(answer_str, ev_objs)
        elif actual_state == ControllerState.NO_RETRIEVE:
            groundedness = 1.0

        # Cost calculation
        is_local = self.engine.provider.capabilities().local
        cost = estimate_turn_cost(" ".join(test.inputs), answer_str, is_local=is_local)

        return TestResult(
            test_id=test.test_id,
            scenario_name=test.scenario_name,
            decision_correct=decision_correct,
            actual_state=actual_state.value,
            expected_state=test.expected_controller_state.value,
            recall=recall,
            groundedness=groundedness,
            citations_count=len(citations_found),
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            estimated_cost_usd=cost,
            decomposed_queries=decomposed_queries,
            retrieved_docs=retrieved_docs,
        )

    def run_benchmark(
        self,
        dataset: list[EvalTestCase] | None = None,
    ) -> dict[str, Any]:
        """Run complete benchmark suite and return aggregate statistics."""
        suite = dataset or get_standard_eval_dataset()
        results: list[TestResult] = []

        for tc in suite:
            res = self.run_test_case(tc)
            results.append(res)

        total_cases = len(results)
        decisions_correct = sum(1 for r in results if r.decision_correct)
        accuracy = (decisions_correct / total_cases) * 100.0 if total_cases > 0 else 0.0

        # Retrieval cases only for recall
        retrieval_cases = [r for r in results if r.expected_state == ControllerState.RETRIEVE.value]
        mean_recall = (
            sum(r.recall for r in retrieval_cases) / len(retrieval_cases)
            if retrieval_cases
            else 1.0
        )

        # Groundedness across evaluated answers
        evaluated_answers = [r for r in results if r.actual_state != ControllerState.WAIT.value]
        mean_groundedness = (
            sum(r.groundedness for r in evaluated_answers) / len(evaluated_answers)
            if evaluated_answers
            else 1.0
        )

        # Latencies
        active_latencies = [r for r in results if r.total_latency_ms > 0]
        mean_ttft = (
            sum(r.ttft_ms for r in active_latencies) / len(active_latencies)
            if active_latencies
            else 0.0
        )
        mean_total_latency = (
            sum(r.total_latency_ms for r in active_latencies) / len(active_latencies)
            if active_latencies
            else 0.0
        )

        return {
            "total_tests": total_cases,
            "decision_accuracy_pct": round(accuracy, 2),
            "mean_retrieval_recall": round(mean_recall, 4),
            "mean_groundedness": round(mean_groundedness, 4),
            "mean_ttft_ms": round(mean_ttft, 2),
            "mean_total_latency_ms": round(mean_total_latency, 2),
            "test_results": [r.to_dict() for r in results],
        }

    def format_summary_table(self, report: dict[str, Any]) -> str:
        """Format evaluation report as a clean markdown table."""
        lines = [
            "# AVY — Streaming Live RAG Benchmark Evaluation",
            "",
            "## Summary Metrics",
            f"- **Controller Decision Accuracy**: {report['decision_accuracy_pct']}%",
            f"- **Mean Retrieval Recall@K**: {report['mean_retrieval_recall'] * 100:.1f}%",
            f"- **Mean Answer Groundedness**: {report['mean_groundedness'] * 100:.1f}%",
            f"- **Mean Time-to-First-Token (TTFT)**: {report['mean_ttft_ms']:.1f} ms",
            f"- **Mean Total Latency**: {report['mean_total_latency_ms']:.1f} ms",
            "",
            "## Detailed Scenario Results",
            "| ID | Scenario | Expected | Actual | Match | Recall | Grounded | TTFT (ms) | Total (ms) |",
            "|---|---|---|---|:---:|:---:|:---:|:---:|:---:|",
        ]

        for r in report["test_results"]:
            match_icon = "✅" if r["decision_correct"] else "❌"
            rec_str = f"{r['recall'] * 100:.0f}%" if r["expected_state"] == "RETRIEVE" else "N/A"
            grd_str = f"{r['groundedness'] * 100:.0f}%" if r["actual_state"] != "WAIT" else "N/A"
            ttft_str = f"{r['ttft_ms']:.0f}" if r["ttft_ms"] > 0 else "-"
            total_str = f"{r['total_latency_ms']:.0f}" if r["total_latency_ms"] > 0 else "-"

            lines.append(
                f"| {r['test_id']} | {r['scenario']} | {r['expected_state']} | {r['actual_state']} | {match_icon} | {rec_str} | {grd_str} | {ttft_str} | {total_str} |"
            )

        return "\n".join(lines)
