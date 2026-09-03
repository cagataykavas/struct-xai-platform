from __future__ import annotations

import argparse
import json
from pathlib import Path

from structxai.evaluation import evaluate_interventions
from structxai.stability import compare_runs

CANDIDATES = ("A", "B", "C")


def analysis(rows: list[tuple[float, float, float]]) -> dict[str, object]:
    return {
        "layers": [
            {
                "layer": index,
                "candidate_scores": {
                    "A": scores[0],
                    "B": scores[1],
                    "C": scores[2],
                },
            }
            for index, scores in enumerate(rows)
        ]
    }


def build_report() -> dict[str, object]:
    base = analysis(
        [
            (0.04, 0.02, -0.01),
            (0.10, 0.06, 0.01),
            (0.19, 0.11, 0.04),
            (0.31, 0.15, 0.07),
        ]
    )
    deletion = analysis(
        [
            (0.03, 0.02, -0.01),
            (0.06, 0.07, 0.01),
            (0.09, 0.13, 0.04),
            (0.12, 0.19, 0.08),
        ]
    )
    replacement = analysis(
        [
            (0.04, 0.02, -0.01),
            (0.09, 0.06, 0.01),
            (0.16, 0.12, 0.05),
            (0.24, 0.18, 0.08),
        ]
    )
    repeat = analysis(
        [
            (0.041, 0.019, -0.011),
            (0.099, 0.061, 0.009),
            (0.188, 0.112, 0.041),
            (0.307, 0.153, 0.071),
        ]
    )

    interventions = evaluate_interventions(
        base,
        {
            "delete_evidence_span": {"analysis": deletion},
            "replace_evidence_span": {"analysis": replacement},
        },
        CANDIDATES,
    )
    stability = compare_runs(base, repeat, CANDIDATES)

    return {
        "scope": "synthetic reproducibility evidence",
        "claim_boundary": (
            "This report validates metric plumbing and repeatability semantics on synthetic "
            "layer-wise candidate scores; it is not a model-quality or causal-faithfulness claim."
        ),
        "candidates": list(CANDIDATES),
        "intervention_evaluation": interventions,
        "cross_run_stability": stability,
    }


def markdown(report: dict[str, object]) -> str:
    stability = report["cross_run_stability"]
    assert isinstance(stability, dict)
    summary = stability["summary"]
    assert isinstance(summary, dict)

    interventions = report["intervention_evaluation"]
    assert isinstance(interventions, dict)
    pairs = interventions["pairs"]
    assert isinstance(pairs, list)

    lines = [
        "# Struct-XAI Synthetic Reproducibility Evidence",
        "",
        "> This is synthetic CI evidence for the evaluation pipeline, not a scientific result.",
        "",
        "## Cross-run stability",
        "",
        f"- Mean Pearson correlation: `{float(summary['mean_pearson_correlation']):.6f}`",
        f"- Mean sign agreement: `{float(summary['mean_sign_agreement_rate']):.6f}`",
        f"- Mean absolute margin difference: `{float(summary['mean_abs_margin_difference']):.6f}`",
        f"- Final pairwise winners all agree: `{bool(summary['all_final_pairwise_winners_agree'])}`",
        "",
        "## Intervention sensitivity",
        "",
        "| Pair | Intervention | Final margin delta | Peak layer | Winner changed |",
        "|---|---|---:|---:|---|",
    ]

    for pair in pairs:
        assert isinstance(pair, dict)
        pair_name = f"{pair['positive']} vs {pair['negative']}"
        rows = pair["interventions"]
        assert isinstance(rows, dict)
        for name, payload in rows.items():
            assert isinstance(payload, dict)
            lines.append(
                "| "
                + " | ".join(
                    [
                        pair_name,
                        str(name),
                        f"{float(payload['final_margin_delta']):.6f}",
                        str(payload["peak_effect_layer"]),
                        str(bool(payload["pairwise_winner_changed_at_final_layer"])),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic Struct-XAI CI evidence")
    parser.add_argument("--json", default="artifacts/reproducibility_evidence.json")
    parser.add_argument("--markdown", default="artifacts/reproducibility_evidence.md")
    args = parser.parse_args()

    report = build_report()
    json_path = Path(args.json)
    markdown_path = Path(args.markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
