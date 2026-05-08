from __future__ import annotations

from src.forecasting.stats.shared.stats_stage_runner import run_stage_for_model


if __name__ == "__main__":
    run_stage_for_model("egarch", "stage1")
