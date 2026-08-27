from __future__ import annotations

from structxai.evaluation import evaluate_interventions
from structxai.experiment import ExperimentSpec
from structxai.provenance import build_provenance, validate_artifact


def _analysis(prompt: str, margins: list[float]) -> dict:
    return {
        "model": "tiny-model",
        "prompt": prompt,
        "candidates": [
            {"label": "A", "token_ids": [1]},
            {"label": "B", "token_ids": [2]},
        ],
        "metric": "first_token_candidate_logit_margin",
        "device": "cpu",
        "dtype": "float32",
        "layers": [
            {
                "layer": layer,
                "winner": "A" if margin >= 0 else "B",
                "winner_score": max(margin, 0.0),
                "runner_up": "B" if margin >= 0 else "A",
                "runner_up_score": min(margin, 0.0),
                "margin": abs(margin),
                "candidate_scores": {"A": margin, "B": 0.0},
            }
            for layer, margin in enumerate(margins)
        ],
    }


def _payload() -> dict:
    spec = ExperimentSpec(
        name="synthetic",
        prompt="A compact prompt",
        candidates=("A", "B"),
        model_name="tiny-model",
    )
    base = _analysis(spec.prompt, [1.0, 2.0, 3.0])
    variant = _analysis("A changed prompt", [0.5, -1.0, -2.0])
    variants = {
        "delete_context": {
            "intervention": {
                "name": "delete_context",
                "kind": "delete",
                "literal": "compact",
                "replacement": "",
            },
            "prompt": "A changed prompt",
            "analysis": variant,
        }
    }
    return {
        "experiment": {
            "name": spec.name,
            "prompt": spec.prompt,
            "candidates": list(spec.candidates),
            "model_name": spec.model_name,
        },
        "provenance": build_provenance(spec, base, requested_device="cpu"),
        "base": base,
        "variants": variants,
    }


def test_intervention_evaluation_detects_margin_shift_and_winner_flip() -> None:
    payload = _payload()
    evaluation = evaluate_interventions(
        payload["base"],
        payload["variants"],
        ("A", "B"),
    )

    pair = evaluation["pairs"][0]
    intervention = pair["interventions"]["delete_context"]
    assert pair["baseline"]["final_margin"] == 3.0
    assert intervention["final_margin_delta"] == -5.0
    assert intervention["max_abs_margin_delta"] == 5.0
    assert intervention["peak_effect_layer"] == 2
    assert intervention["pairwise_winner_changed_at_final_layer"] is True
    assert intervention["variant_sign_flip_count"] == 1


def test_artifact_validation_accepts_consistent_experiment() -> None:
    validation = validate_artifact(_payload())

    assert validation["valid"] is True
    assert validation["errors"] == []
    assert validation["layers"] == 3
    assert validation["variants"] == 1


def test_artifact_validation_detects_config_tampering() -> None:
    payload = _payload()
    payload["experiment"]["prompt"] = "tampered prompt"

    validation = validate_artifact(payload)

    assert validation["valid"] is False
    assert "experiment_fingerprint_mismatch" in validation["errors"]
    assert "prompt_hash_mismatch" in validation["errors"]
    assert "base_prompt_mismatch" in validation["errors"]


def test_artifact_validation_detects_layer_misalignment() -> None:
    payload = _payload()
    payload["variants"]["delete_context"]["analysis"]["layers"][2]["layer"] = 99

    validation = validate_artifact(payload)

    assert validation["valid"] is False
    assert "variant_layer_alignment_mismatch:delete_context" in validation["errors"]
