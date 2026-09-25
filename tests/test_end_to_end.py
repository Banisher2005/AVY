"""End-to-end integration tests for all 6 Theme 04 benchmark scenarios."""

from avy.evaluation.dataset import get_standard_eval_dataset
from avy.evaluation.harness import EvaluationHarness
from avy.streaming.engine import StreamingLiveRAGEngine


def test_all_scenarios_end_to_end(test_engine: StreamingLiveRAGEngine) -> None:
    harness = EvaluationHarness(engine=test_engine)
    dataset = get_standard_eval_dataset()

    report = harness.run_benchmark(dataset)

    assert report["total_tests"] == len(dataset)
    assert report["decision_accuracy_pct"] == 100.0
    assert report["mean_retrieval_recall"] > 0.0
    assert report["mean_groundedness"] > 0.0

    # Ensure all test results completed cleanly
    for res in report["test_results"]:
        assert res["decision_correct"] is True
        assert res["actual_state"] in ("WAIT", "RETRIEVE", "NO_RETRIEVE")
