import json
from pathlib import Path

import torch

from structxai.core import Candidate, find_sign_flips, layerwise_decisions
from structxai.interventions import delete_literal, replace_literal
from structxai.report import render_report


def test_candidate_decisions_and_sign_flip():
    candidates = [Candidate("A", (1,)), Candidate("B", (2,))]
    logits = [
        torch.tensor([0.0, 3.0, 1.0]),
        torch.tensor([0.0, 1.0, 4.0]),
    ]
    decisions = layerwise_decisions(logits, candidates)
    assert decisions[0].winner == "A"
    assert decisions[1].winner == "B"
    flips = find_sign_flips(decisions, "A", "B")
    assert [flip.layer for flip in flips] == [1]


def test_literal_interventions():
    prompt = "The capital of Turkey is"
    deletion = delete_literal(prompt, "Turkey")
    replacement = replace_literal(prompt, "Turkey", "Greece")
    assert deletion.apply(prompt) == "The capital of  is"
    assert replacement.apply(prompt) == "The capital of Greece is"


def test_report_is_standalone_html(tmp_path: Path):
    payload = {
        "experiment": {
            "name": "demo",
            "prompt": "Question?",
            "candidates": ["A", "B"],
            "model_name": "tiny",
        },
        "base": {
            "metric": "first_token_candidate_logit_margin",
            "sign_flips": [],
            "layers": [
                {
                    "layer": 0,
                    "winner": "A",
                    "winner_score": 2.0,
                    "runner_up": "B",
                    "runner_up_score": 1.0,
                    "margin": 1.0,
                    "candidate_scores": {"A": 2.0, "B": 1.0},
                }
            ],
        },
        "variants": {},
    }
    path = render_report(payload, tmp_path / "report.html")
    text = path.read_text()
    assert "Struct-XAI" in text
    assert "Layer-wise candidate margin" in text
