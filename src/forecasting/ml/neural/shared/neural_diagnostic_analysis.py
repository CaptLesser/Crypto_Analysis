from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.forecasting.ml.shared.diagnostic_analysis_common import (
    StageContext,
    analyze_manifest_for_model,
    build_asset_combo_detail,
    build_stage3_survivor_handoff,
    build_stage3_window_candidates,
    build_window_summary,
    load_json,
    percentile,
    safe_float,
    select_stage3_window,
    stage3_window_gate,
    stage_contexts,
    utc_now_iso,
    write_dual,
)


def main_for_model(model_key: str) -> None:
    parser = argparse.ArgumentParser(description="Analyze the latest Neural numeric diagnostic run.")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    outputs = analyze_manifest_for_model(model_key, args.manifest.resolve())
    for value in outputs.values():
        print(value)


if __name__ == "__main__":
    main_for_model("lstm")
