from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from structxai.experiment import (
    ExperimentSpec,
    PerturbationAuditRejected,
    PerturbationAuditSpec,
    compare_interventions,
)
from structxai.interventions import delete_literal


def _trace(input_ids: list[int], offsets: list[tuple[int, int]]) -> dict:
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "offset_mapping": offsets,
        "vocab_size": 128,
        "pad_token_id": 0,
        "mask_token_id": 127,
        "special_token_ids": [1, 2],
    }


def _analysis(prompt: str, trace: dict) -> dict:
    return {
        "model": "tiny-model",
        "prompt": prompt,
        "candidates": [
            {"label": " yes", "token_ids": [7]},
            {"label": " no", "token_ids": [8]},
        ],
        "metric": "first_token_candidate_logit_margin",
        "device": "cpu",
        "dtype": "float32",
        "layers": [
            {
                "layer": 0,
                "winner": " yes",
                "winner_score": 1.0,
                "runner_up": " no",
                "runner_up_score": 0.0,
                "margin": 1.0,
                "candidate_scores": {" yes": 1.0, " no": 0.0},
            }
        ],
        "_token_trace": trace,
    }


def _audit_spec() -> PerturbationAuditSpec:
    return PerturbationAuditSpec(
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
        policy_id="token-splice-v1",
        protected_literal="C",
    )


def test_compare_emits_and_admits_live_token_evidence(tmp_path: Path, monkeypatch) -> None:
    traces = {
        "A B C": _trace([1, 10, 20, 30, 2], [(0, 0), (0, 1), (2, 3), (4, 5), (0, 0)]),
        "A  C": _trace([1, 10, 30, 2], [(0, 0), (0, 1), (3, 4), (0, 0)]),
    }

    def fake_runner(prompt, _candidates, _model, **kwargs):
        assert kwargs["model_revision"] == "a" * 40
        assert kwargs["tokenizer_revision"] == "b" * 40
        assert kwargs["capture_token_trace"] is True
        return _analysis(prompt, traces[prompt])

    monkeypatch.setattr("structxai.experiment.run_layerwise", fake_runner)
    spec = ExperimentSpec("audited-delete", "A B C", (" yes", " no"), "tiny-model")
    payload = compare_interventions(
        spec,
        [delete_literal(spec.prompt, "B")],
        tmp_path,
        perturbation_audit=_audit_spec(),
    )

    integrity = payload["perturbation_integrity"]
    assert integrity["status"] == "accepted"
    assert integrity["variants"] == 1
    assert len(integrity["evidence_sha256"]) == 64
    report = json.loads(Path(integrity["report_path"]).read_text())
    assert report["metrics"]["accepted_variants"] == 1
    persisted = json.loads((tmp_path / "audited-delete.json").read_text())
    assert "_token_trace" not in json.dumps(persisted)


def test_compare_rejects_neighbor_retokenization_before_main_artifact(tmp_path: Path, monkeypatch) -> None:
    traces = {
        "A B C": _trace([1, 10, 20, 30, 2], [(0, 0), (0, 1), (2, 3), (4, 5), (0, 0)]),
        "A  C": _trace([1, 99, 30, 2], [(0, 0), (0, 1), (3, 4), (0, 0)]),
    }

    def fake_runner(prompt, _candidates, _model, **_kwargs):
        return _analysis(prompt, traces[prompt])

    monkeypatch.setattr("structxai.experiment.run_layerwise", fake_runner)
    spec = ExperimentSpec("rejected-delete", "A B C", (" yes", " no"), "tiny-model")
    with pytest.raises(PerturbationAuditRejected):
        compare_interventions(
            spec,
            [delete_literal(spec.prompt, "B")],
            tmp_path,
            perturbation_audit=_audit_spec(),
        )

    assert not (tmp_path / "rejected-delete.json").exists()
    report = json.loads((tmp_path / "rejected-delete.perturbation-audit.json").read_text())
    assert report["status"] == "rejected"
    assert report["metrics"]["finding_code_counts"] == {"TOKEN001": 1}


def test_character_edits_must_be_token_aligned(tmp_path: Path, monkeypatch) -> None:
    trace = _trace([1, 10, 20, 2], [(0, 0), (0, 1), (2, 5), (0, 0)])

    def fake_runner(prompt, _candidates, _model, **_kwargs):
        return _analysis(prompt, trace)

    monkeypatch.setattr("structxai.experiment.run_layerwise", fake_runner)
    spec = ExperimentSpec("partial-token", "A B C", (" yes", " no"), "tiny-model")
    with pytest.raises(ValueError, match="token-aligned"):
        compare_interventions(
            spec,
            [delete_literal(spec.prompt, "B")],
            tmp_path,
            perturbation_audit=_audit_spec(),
        )


def test_hf_runner_captures_the_exact_model_input_and_pins_revisions(monkeypatch) -> None:
    from structxai import hf_runner

    calls: dict[str, dict] = {}

    class Encoding(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        pad_token_id = 0
        mask_token_id = 99
        all_special_ids = (0, 1)

        def __len__(self):
            return 100

        def __call__(self, _prompt, **kwargs):
            value = Encoding(
                input_ids=torch.tensor([[1, 10, 2]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
            )
            if kwargs.get("return_offsets_mapping"):
                value["offset_mapping"] = torch.tensor([[[0, 0], [0, 1], [0, 0]]])
            return value

        def encode(self, label, **_kwargs):
            return [7 if "yes" in label else 8]

    class Model:
        def to(self, _device):
            return self

        def eval(self):
            return None

        def __call__(self, **_kwargs):
            return type("Output", (), {"hidden_states": (torch.zeros(1, 1, 1),)})()

    def load_tokenizer(_name, **kwargs):
        calls["tokenizer"] = kwargs
        return Tokenizer()

    def load_model(_name, **kwargs):
        calls["model"] = kwargs
        return Model()

    monkeypatch.setattr(hf_runner.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(hf_runner.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(hf_runner, "project_hidden_states", lambda _model, _states: [torch.arange(100)])

    result = hf_runner.run_layerwise(
        "A",
        (" yes", " no"),
        "tiny-model",
        device="cpu",
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
        capture_token_trace=True,
    )

    assert calls["model"]["revision"] == "a" * 40
    assert calls["tokenizer"]["revision"] == "b" * 40
    assert result["_token_trace"]["input_ids"] == [1, 10, 2]
    assert result["_token_trace"]["attention_mask"] == [1, 1, 1]
    assert result["_token_trace"]["offset_mapping"] == [(0, 0), (0, 1), (0, 0)]
