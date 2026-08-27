from __future__ import annotations

import argparse
import json
from pathlib import Path

from structxai.evaluation import evaluate_interventions
from structxai.experiment import ExperimentSpec, compare_interventions
from structxai.interventions import delete_literal, replace_literal
from structxai.patching import patch_final_position, serialize_patch
from structxai.provenance import validate_artifact
from structxai.report import render_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="struct-xai",
        description="Compact layer-wise LLM interpretability toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="run base + deletion/replacement experiments")
    compare.add_argument("--prompt", required=True)
    compare.add_argument("--candidates", nargs="+", required=True)
    compare.add_argument("--delete")
    compare.add_argument("--replace")
    compare.add_argument("--with", dest="replacement")
    compare.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    compare.add_argument("--name", default="experiment")
    compare.add_argument("--output", type=Path, default=Path("artifacts"))
    compare.add_argument("--device")

    patch = sub.add_parser("patch", help="patch one layer's final-position activation")
    patch.add_argument("--source-prompt", required=True)
    patch.add_argument("--target-prompt", required=True)
    patch.add_argument("--layer", type=int, required=True)
    patch.add_argument("--candidates", nargs="+", required=True)
    patch.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    patch.add_argument("--device")

    evaluate = sub.add_parser(
        "evaluate",
        help="validate an experiment artifact and recompute intervention metrics offline",
    )
    evaluate.add_argument("artifact", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--fail-invalid", action="store_true")

    report = sub.add_parser("report", help="render an experiment JSON artifact as standalone HTML")
    report.add_argument("artifact", type=Path)
    report.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "compare":
        interventions = []
        if args.delete:
            interventions.append(delete_literal(args.prompt, args.delete))
        if args.replace:
            if args.replacement is None:
                raise SystemExit("--replace requires --with")
            interventions.append(replace_literal(args.prompt, args.replace, args.replacement))
        spec = ExperimentSpec(args.name, args.prompt, tuple(args.candidates), args.model)
        payload = compare_interventions(spec, interventions, args.output, device=args.device)
        html_path = render_report(payload, args.output / f"{args.name}.html")
        print(json.dumps({"artifact": payload["artifact_path"], "report": str(html_path)}, indent=2))
        return 0

    if args.command == "patch":
        result = patch_final_position(
            args.source_prompt,
            args.target_prompt,
            args.layer,
            args.candidates,
            args.model,
            args.device,
        )
        print(json.dumps(serialize_patch(result), indent=2))
        return 0

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    if args.command == "evaluate":
        candidates = tuple(str(item) for item in payload["experiment"]["candidates"])
        result = {
            "validation": validate_artifact(payload),
            "evaluation": evaluate_interventions(payload["base"], payload.get("variants", {}), candidates),
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(args.output)
        else:
            print(text)
        if args.fail_invalid and not result["validation"]["valid"]:
            return 2
        return 0

    output = args.output or args.artifact.with_suffix(".html")
    print(render_report(payload, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
