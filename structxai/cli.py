from __future__ import annotations

import argparse
import json
from pathlib import Path

from structxai.experiment import ExperimentSpec, compare_interventions
from structxai.interventions import delete_literal, replace_literal
from structxai.patching import patch_final_position, serialize_patch
from structxai.report import render_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="struct-xai", description="Compact layer-wise LLM interpretability toolkit")
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

    report = sub.add_parser("report", help="render an experiment JSON artifact as standalone HTML")
    report.add_argument("artifact", type=Path)
    report.add_argument("--output", type=Path)
    return parser


def main() -> None:
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
        return

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
        return

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    output = args.output or args.artifact.with_suffix(".html")
    print(render_report(payload, output))


if __name__ == "__main__":
    main()
