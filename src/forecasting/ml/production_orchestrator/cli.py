from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from src.forecasting.common.sandbox_paths import resolve_sandbox_output_roots
from src.forecasting.ml.production_orchestrator.runner import DEFAULT_OUTPUT_DIR, OrchestratorArgs, run_orchestrator


def parse_args(argv: Optional[Sequence[str]] = None) -> OrchestratorArgs:
    parser = argparse.ArgumentParser(description="Lightweight production orchestrator for mature numeric ML modules.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--profile", type=str, default="production")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--resume-run", type=str, default="")
    parser.add_argument("--no-resume-latest", action="store_true")
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    parser.add_argument("--python-exe", type=str, default="")
    parser.add_argument("--deep-disk-preflight", action="store_true", help="Run expensive recursive gap scans before deciding a module can be skipped.")
    parser.add_argument(
        "--sandbox-output-root",
        type=Path,
        default=None,
        help="Enable sandbox output mode and redirect write-class artifacts under this root.",
    )
    args = parser.parse_args(argv)
    sandbox_roots = resolve_sandbox_output_roots(args)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            sandbox_roots.diagnostics_root / "production_numeric_orchestrator"
            if sandbox_roots.enabled
            else DEFAULT_OUTPUT_DIR
        )
    return OrchestratorArgs(
        project_root=args.project_root,
        output_dir=output_dir,
        profile=str(args.profile or "production"),
        run_id=str(args.run_id or ""),
        resume_run=str(args.resume_run or ""),
        no_resume_latest=bool(args.no_resume_latest),
        sample_seconds=float(args.sample_seconds),
        python_exe=(str(args.python_exe).strip() or __import__("sys").executable),
        deep_disk_preflight=bool(args.deep_disk_preflight),
        sandbox_output_root=args.sandbox_output_root,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_root = run_orchestrator(args)
    print(json.dumps({"run_root": str(run_root), "manifest": str((run_root / "orchestrator_run_manifest.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
