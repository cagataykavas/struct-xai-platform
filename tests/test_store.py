from pathlib import Path

from cloud_service.store import ExperimentStore


def test_experiment_store_lifecycle(tmp_path: Path):
    store = ExperimentStore(tmp_path / "experiments.db")
    created = store.create("exp-1", {"prompt": "hello", "candidates": ["A", "B"]})
    assert created.status == "queued"

    running = store.update("exp-1", status="running")
    assert running.status == "running"

    completed = store.update("exp-1", status="completed", artifact_path="runs/exp-1/report.html")
    assert completed.status == "completed"
    assert completed.artifact_path.endswith("report.html")
    assert store.list()[0].experiment_id == "exp-1"
