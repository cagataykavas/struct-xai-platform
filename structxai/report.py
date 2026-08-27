from __future__ import annotations

import html
import json
from pathlib import Path


def _bar(value: float, scale: float) -> str:
    width = min(abs(value) / max(scale, 1e-9) * 100.0, 100.0)
    direction = "positive" if value >= 0 else "negative"
    return f'<div class="bar-track"><div class="bar {direction}" style="width:{width:.1f}%"></div></div>'


def _evaluation_cards(payload: dict) -> str:
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        return ""

    cards: list[str] = []
    for pair in evaluation.get("pairs", []):
        positive = html.escape(str(pair["positive"]))
        negative = html.escape(str(pair["negative"]))
        baseline = pair["baseline"]
        interventions = pair.get("interventions", {})
        if not interventions:
            cards.append(
                '<article class="card">'
                f"<h3>{positive} vs {negative}</h3>"
                f"<p>Final base margin: <code>{baseline['final_margin']:.4f}</code></p>"
                "</article>"
            )
            continue

        for name, metrics in interventions.items():
            cards.append(
                '<article class="card">'
                f"<h3>{positive} vs {negative} · {html.escape(str(name))}</h3>"
                f"<p>Mean |Δ margin|: <code>{metrics['mean_abs_margin_delta']:.4f}</code><br>"
                f"Final Δ margin: <code>{metrics['final_margin_delta']:.4f}</code><br>"
                f"Peak effect: layer <code>{metrics['peak_effect_layer']}</code> "
                f"(|Δ|={metrics['max_abs_margin_delta']:.4f})<br>"
                "Final pairwise winner changed: "
                f"<code>{metrics['pairwise_winner_changed_at_final_layer']}</code></p>"
                "</article>"
            )
    return "".join(cards)


def _provenance_card(payload: dict) -> str:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return ""
    runtime = provenance.get("runtime", {})
    fingerprint = html.escape(str(provenance.get("experiment_fingerprint", "unknown")))
    resolved_device = html.escape(str(provenance.get("resolved_device", "unknown")))
    dtype = html.escape(str(provenance.get("dtype", "unknown")))
    versions = html.escape(
        f"torch={runtime.get('torch')} · transformers={runtime.get('transformers')} · "
        f"python={runtime.get('python')}"
    )
    return (
        '<article class="card">'
        "<h3>Reproducibility</h3>"
        f"<p>Fingerprint: <code>{fingerprint}</code><br>"
        f"Runtime: <code>{resolved_device} / {dtype}</code><br>"
        f"{versions}</p>"
        "</article>"
    )


def render_report(payload: dict, output: str | Path) -> Path:
    base_layers = payload["base"]["layers"]
    all_scores = [abs(layer["margin"]) for layer in base_layers] or [1.0]
    scale = max(all_scores)

    rows = []
    for layer in base_layers:
        rows.append(
            "<tr>"
            f"<td>{layer['layer']}</td>"
            f"<td>{html.escape(layer['winner'])}</td>"
            f"<td>{layer['margin']:.4f}</td>"
            f"<td>{_bar(layer['margin'], scale)}</td>"
            "</tr>"
        )

    variant_cards = []
    for name, variant in payload.get("variants", {}).items():
        flips = variant["analysis"].get("sign_flips", [])
        variant_cards.append(
            '<article class="card">'
            f"<h3>{html.escape(name)}</h3>"
            f"<p><strong>Prompt:</strong> {html.escape(variant['prompt'])}</p>"
            f"<p><strong>Sign flips:</strong> {html.escape(json.dumps(flips))}</p>"
            "</article>"
        )

    provenance_card = _provenance_card(payload)
    evaluation_cards = _evaluation_cards(payload)
    evaluation_section = ""
    if evaluation_cards:
        evaluation_section = (
            "<h2>Quantitative intervention evaluation</h2>"
            f'<div class="grid">{evaluation_cards}</div>'
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Struct-XAI — {html.escape(payload['experiment']['name'])}</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin:0; background:#090d14; color:#e8edf7; }}
main {{ max-width:1100px; margin:auto; padding:42px 22px 80px; }}
h1 {{ font-size:clamp(2rem,5vw,4rem); margin:.2em 0; }}
.sub {{ color:#93a4bd; max-width:760px; line-height:1.6; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; margin:28px 0; }}
.card {{ background:#111824; border:1px solid #263247; border-radius:16px; padding:18px; overflow-wrap:anywhere; }}
table {{ width:100%; border-collapse:collapse; background:#111824; border-radius:16px; overflow:hidden; }}
th,td {{ padding:11px 14px; border-bottom:1px solid #202c3e; text-align:left; }}
th {{ color:#93a4bd; }}
.bar-track {{ width:100%; height:9px; background:#20293a; border-radius:99px; overflow:hidden; }}
.bar {{ height:100%; border-radius:99px; background:linear-gradient(90deg,#5de4c7,#7aa2f7); }}
.bar.negative {{ background:linear-gradient(90deg,#ff7a90,#ffb86b); }}
code {{ color:#8bd5ca; }}
</style>
</head>
<body><main>
<p class="sub">STRUCT-XAI / LAYER-WISE DECISION TRACE</p>
<h1>{html.escape(payload['experiment']['name'])}</h1>
<p class="sub">Model: <code>{html.escape(payload['experiment']['model_name'])}</code><br>
Metric: <code>{html.escape(payload['base']['metric'])}</code></p>
<div class="grid">
<article class="card"><h3>Prompt</h3><p>{html.escape(payload['experiment']['prompt'])}</p></article>
<article class="card"><h3>Candidates</h3><p>{html.escape(' · '.join(payload['experiment']['candidates']))}</p></article>
<article class="card"><h3>Base sign flips</h3><p>{html.escape(json.dumps(payload['base'].get('sign_flips', [])))}</p></article>
{provenance_card}
</div>
<h2>Layer-wise candidate margin</h2>
<table><thead><tr><th>Layer</th><th>Winner</th><th>Margin</th><th>Magnitude</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Interventions</h2>
<div class="grid">{''.join(variant_cards) or '<article class="card">No intervention variants.</article>'}</div>
{evaluation_section}
</main></body></html>"""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
