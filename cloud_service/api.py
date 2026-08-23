from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from cloud_service.store import ExperimentStore
from structxai.experiment import ExperimentSpec, compare_interventions
from structxai.interventions import delete_literal, replace_literal
from structxai.report import render_report

STORE = ExperimentStore(os.getenv("STRUCT_XAI_DB", "artifacts/experiments.db"))
ARTIFACT_ROOT = Path(os.getenv("STRUCT_XAI_ARTIFACTS", "artifacts/runs"))
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Struct-XAI Experiment Service",
    version="0.2.0",
    description="Queue and persist small-model interpretability experiments.",
)


class InterventionRequest(BaseModel):
    kind: str = Field(pattern="^(delete|replace)$")
    literal: str
    replacement: str = ""
    name: str | None = None


class ExperimentRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    prompt: str = Field(min_length=2)
    candidates: list[str] = Field(min_length=2, max_length=8)
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str | None = None
    interventions: list[InterventionRequest] = []


def _run(experiment_id: str, request: ExperimentRequest) -> None:
    STORE.update(experiment_id, status="running")
    run_dir = ARTIFACT_ROOT / experiment_id
    try:
        interventions = []
        for item in request.interventions:
            if item.kind == "delete":
                interventions.append(delete_literal(request.prompt, item.literal, item.name))
            else:
                interventions.append(
                    replace_literal(
                        request.prompt,
                        item.literal,
                        item.replacement,
                        item.name,
                    )
                )

        spec = ExperimentSpec(
            request.name,
            request.prompt,
            tuple(request.candidates),
            request.model_name,
        )
        payload = compare_interventions(spec, interventions, run_dir, device=request.device)
        report_path = render_report(payload, run_dir / "report.html")
        STORE.update(
            experiment_id,
            status="completed",
            artifact_path=str(report_path),
        )
    except Exception as exc:
        STORE.update(experiment_id, status="failed", error=f"{type(exc).__name__}: {exc}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "artifact_root": str(ARTIFACT_ROOT)}


@app.post("/experiments", status_code=202)
def create_experiment(request: ExperimentRequest, background: BackgroundTasks) -> dict:
    experiment_id = uuid.uuid4().hex
    record = STORE.create(experiment_id, request.model_dump())
    background.add_task(_run, experiment_id, request)
    return asdict(record)


@app.get("/experiments")
def list_experiments(limit: int = 50) -> list[dict]:
    return [asdict(record) for record in STORE.list(min(max(limit, 1), 200))]


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict:
    try:
        return asdict(STORE.get(experiment_id))
    except KeyError as exc:
        raise HTTPException(404, "experiment not found") from exc
