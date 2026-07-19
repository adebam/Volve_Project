"""
MODEL 1: direct multi-horizon Lasso models, deployed with periodic reset.

Reuses Model 0's horizon library (one independently-trained model per
(phase, horizon) pair -- see Direct_Horizon_Model0_helper_functions.py) but
deploys it as an actual walk-forward simulation instead of Model 0's dense-
anchor diagnostic scoring.

At each reset date, every day out to `block_horizon` is predicted directly
from that one real anchor: day+1 from the horizon=1 model, day+2 from the
horizon=2 model, ..., day+block_horizon from the horizon=block_horizon
model -- each day's feature row is just the anchor's real row, reused
across all H models. No day's prediction ever depends on another day's
prediction; there is no recursive chaining within a block. Only the anchor
itself resets, every block_horizon days, to the next real measurement.

A 7-day deployment uses the horizon 1-7 subset of the trained library; a
30-day deployment uses the full 1-30 set -- both reuse the same underlying
models, no retraining needed per block length.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from Machine_Learning_Multi import (
    plot_train_val_actual_vs_predicted_crossplot,
    plot_train_val_predictions_time,
)

from Direct_Horizon_Model0_helper_functions import (
    FUTURE_KNOWN_VARIABLES,
    MAX_LOOKBACK_DAYS,
    SENSOR_LAG_DAYS,
    SENSOR_VARIABLES,
    TARGET_COLS,
    TARGET_LAG_DAYS,
    build_model0_direct_horizon_table,
)


# ------------------------------------------------------------------#
def _get_block_boundaries(start_date, end_date, horizon):
    """Reset dates starting at start_date, every `horizon` days, while
    still within [start_date, end_date]."""
    boundaries = []
    current = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    while current <= end_date:
        boundaries.append(current)
        current = current + pd.Timedelta(days=horizon)

    return boundaries


# ------------------------------------------------------------------#
def run_model1_direct_walk_forward(
    models_by_horizon,
    df,
    simulation_data,
    block_horizon,
    target_cols=TARGET_COLS,
    sensor_variables=SENSOR_VARIABLES,
    future_known_variables=FUTURE_KNOWN_VARIABLES,
    sensor_lag_days=SENSOR_LAG_DAYS,
    target_lag_days=TARGET_LAG_DAYS,
    clip_negative=True,
    verbose=True,
):
    """
    Walk simulation_data forward in blocks of `block_horizon` days, using
    Model 0's per-horizon models directly (no recursion).

    models_by_horizon : dict[int -> dict[target_col -> fitted estimator]]
        Must contain an entry for every horizon in 1..block_horizon, e.g.
        {h: run_model0_horizon_sweep(..., horizons=range(1, 31))[0][h]["models"]
         for h in range(1, 31)}.

    Returns
    -------
    daily_results_df : pd.DataFrame
        One row per (date, phase): Predicted, Actual (real, for scoring
        only), Days_Since_Reset, Block_Id, Is_Reset_Day.
    end_of_block_results_df : pd.DataFrame
        Subset of daily_results_df where Is_Reset_Day is True.
    """
    missing_horizons = [
        h for h in range(1, block_horizon + 1) if h not in models_by_horizon
    ]

    if missing_horizons:
        raise ValueError(
            f"models_by_horizon is missing horizons {missing_horizons}; "
            f"need every horizon from 1 to block_horizon={block_horizon}."
        )

    earliest_valid_start = df.index.min() + pd.Timedelta(days=MAX_LOOKBACK_DAYS)
    simulation_start = max(simulation_data.index.min(), earliest_valid_start)

    if simulation_start > simulation_data.index.max():
        raise ValueError(
            "simulation_data does not extend far enough past df's start "
            f"to leave room for MAX_LOOKBACK_DAYS={MAX_LOOKBACK_DAYS} of "
            "history before the first anchor."
        )

    reset_dates = _get_block_boundaries(
        simulation_start,
        simulation_data.index.max(),
        block_horizon,
    )

    if len(reset_dates) < 2:
        raise ValueError(
            "Not enough date range in simulation_data to form even one "
            f"full block at block_horizon={block_horizon}."
        )

    # Build each (horizon, phase) anchor-indexed feature table once, over
    # the full df, and reuse it via .loc lookups for every block -- avoids
    # rebuilding the same shifted columns from scratch per anchor.
    X_lookup = {
        h: {
            phase: build_model0_direct_horizon_table(
                df,
                target_col=phase,
                horizon=h,
                sensor_variables=sensor_variables,
                future_known_variables=future_known_variables,
                sensor_lag_days=sensor_lag_days,
                target_lag_days=target_lag_days,
            )[0]
            for phase in target_cols
        }
        for h in range(1, block_horizon + 1)
    }

    all_rows = []

    for target_col in target_cols:
        if verbose:
            print(
                f"\n[Model 1 direct walk-forward] {target_col}, "
                f"block_horizon={block_horizon}"
            )

        for block_id, (block_start, _block_end) in enumerate(
            zip(reset_dates[:-1], reset_dates[1:]),
            start=1,
        ):
            for h in range(1, block_horizon + 1):
                X_full = X_lookup[h][target_col]

                if block_start not in X_full.index:
                    # Anchor doesn't have enough lookback/lookahead in df
                    # for this horizon (edge of the dataset) -- skip.
                    continue

                model = models_by_horizon[h][target_col]
                X_step = X_full.loc[[block_start]]

                expected_features = getattr(model, "feature_names_in_", None)
                if expected_features is not None:
                    X_step = X_step.loc[:, list(expected_features)]

                prediction = float(model.predict(X_step)[0])

                if clip_negative:
                    prediction = max(prediction, 0.0)

                forecast_date = block_start + pd.Timedelta(days=h)
                actual_value = (
                    df.loc[forecast_date, target_col]
                    if forecast_date in df.index
                    else np.nan
                )

                all_rows.append(
                    {
                        "Date": forecast_date,
                        "Phase": target_col,
                        "Predicted": prediction,
                        "Actual": actual_value,
                        "Days_Since_Reset": h,
                        "Block_Id": block_id,
                        "Is_Reset_Day": h == block_horizon,
                    }
                )

    daily_results_df = pd.DataFrame(all_rows).set_index("Date").sort_index()
    end_of_block_results_df = daily_results_df[
        daily_results_df["Is_Reset_Day"]
    ].copy()

    return daily_results_df, end_of_block_results_df


# ------------------------------------------------------------------#
def summarize_model1_walk_forward_metrics(
    train_daily_results_df,
    val_daily_results_df,
    block_horizon,
    target_cols=TARGET_COLS,
    model_name="Model 1",
):
    """
    Train/validation walk-forward metrics, tables only -- no plots.

    Val_Daily_* scores every predicted day in validation against the true
    historical value. Val_EndOfBlock_* scores only the reset days -- the
    only comparison real deployment could actually make (predicted vs.
    real separator measurement).
    """
    rows = []

    for phase in target_cols:
        train_phase_df = train_daily_results_df[
            train_daily_results_df["Phase"] == phase
        ]
        val_phase_df = val_daily_results_df[
            val_daily_results_df["Phase"] == phase
        ]
        val_reset_df = val_phase_df[val_phase_df["Is_Reset_Day"]]

        eob_r2 = (
            r2_score(val_reset_df["Actual"], val_reset_df["Predicted"])
            if len(val_reset_df) >= 2
            else np.nan
        )

        rows.append(
            {
                "Model": model_name,
                "Block_Horizon": block_horizon,
                "Phase": phase,
                "Train_RMSE": root_mean_squared_error(
                    train_phase_df["Actual"], train_phase_df["Predicted"]
                ),
                "Train_MAE": mean_absolute_error(
                    train_phase_df["Actual"], train_phase_df["Predicted"]
                ),
                "Train_R2": r2_score(
                    train_phase_df["Actual"], train_phase_df["Predicted"]
                ),
                "Val_Daily_RMSE": root_mean_squared_error(
                    val_phase_df["Actual"], val_phase_df["Predicted"]
                ),
                "Val_Daily_MAE": mean_absolute_error(
                    val_phase_df["Actual"], val_phase_df["Predicted"]
                ),
                "Val_Daily_R2": r2_score(
                    val_phase_df["Actual"], val_phase_df["Predicted"]
                ),
                "Val_EndOfBlock_RMSE": root_mean_squared_error(
                    val_reset_df["Actual"], val_reset_df["Predicted"]
                ),
                "Val_EndOfBlock_MAE": mean_absolute_error(
                    val_reset_df["Actual"], val_reset_df["Predicted"]
                ),
                "Val_EndOfBlock_R2": eob_r2,
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------#
def plot_model1_train_val_diagnostics(
    train_daily_results_df,
    val_daily_results_df,
    block_horizon,
    target_cols=TARGET_COLS,
    model_name="Model 1 Direct Walk-Forward",
    start_of_val_date=None,
    clip_negative=True,
):
    """
    Train + validation walk-forward diagnostics, same time-plot /
    crossplot style used throughout this project (plot_train_val_predictions_time /
    plot_train_val_actual_vs_predicted_crossplot from Machine_Learning_Multi.py).
    Reset days (end of each direct-forecast block, where a real
    measurement anchors the next block) are marked on the validation
    portion only, phase-colored to match the rest of the project.

    Returns
    -------
    metrics_df : pd.DataFrame with Train/Val RMSE, MAE, R2 per phase
        (same content as summarize_model1_walk_forward_metrics).
    """
    color_map = {
        "BORE_OIL_VOL_STB_D": "darkgreen",
        "BORE_GAS_VOL_MSCF_D": "crimson",
        "BORE_WAT_VOL_STB_D": "blue",
    }

    metrics_df = summarize_model1_walk_forward_metrics(
        train_daily_results_df=train_daily_results_df,
        val_daily_results_df=val_daily_results_df,
        block_horizon=block_horizon,
        target_cols=target_cols,
        model_name=model_name,
    )

    # ---- Time plots ----
    fig, axes = plt.subplots(
        nrows=len(target_cols),
        ncols=1,
        figsize=(15, 5 * len(target_cols)),
        sharex=False,
    )

    if len(target_cols) == 1:
        axes = [axes]

    train_val_data = {}

    for ax, phase in zip(axes, target_cols):
        train_phase_df = train_daily_results_df[
            train_daily_results_df["Phase"] == phase
        ]
        val_phase_df = val_daily_results_df[
            val_daily_results_df["Phase"] == phase
        ]
        val_reset_rows = val_phase_df[val_phase_df["Is_Reset_Day"]]

        y_train = train_phase_df["Actual"]
        train_pred = train_phase_df["Predicted"]
        y_val = val_phase_df["Actual"]
        val_pred = val_phase_df["Predicted"]

        train_val_data[phase] = (y_train, train_pred, y_val, val_pred)

        row = metrics_df[metrics_df["Phase"] == phase].iloc[0]

        plot_train_val_predictions_time(
            y_train=y_train,
            train_pred=train_pred,
            y_val=y_val,
            val_pred=val_pred,
            phase_string=phase,
            train_rmse=row["Train_RMSE"],
            train_mae=row["Train_MAE"],
            val_rmse=row["Val_Daily_RMSE"],
            val_mae=row["Val_Daily_MAE"],
            ax=ax,
            start_of_val_date=start_of_val_date,
            clip_negative=clip_negative,
        )

        # Restyle "Validation Actual" to match "Train Actual" (no marker,
        # full opacity) -- done here rather than editing
        # plot_train_val_predictions_time itself, since that function is
        # shared with Model 0 and other notebooks in this project.
        for line in ax.get_lines():
            if line.get_label() == "Validation Actual":
                line.set_marker("None")
                line.set_linewidth(2)
                line.set_alpha(1.0)

        ax.scatter(
            val_reset_rows.index,
            val_reset_rows["Actual"],
            label="Reset day (real measurement)",
            color=color_map.get(phase, "black"),
            edgecolor="black",
            linewidth=0.6,
            zorder=6,
        )
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"{model_name} (block={block_horizon}D): Train Fit and "
        "Walk-Forward Validation Actual vs Predicted",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    # ---- Crossplots ----
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(target_cols),
        figsize=(7 * len(target_cols), 7),
        sharex=False,
        sharey=False,
    )

    if len(target_cols) == 1:
        axes = [axes]

    for ax, phase in zip(axes, target_cols):
        y_train, train_pred, y_val, val_pred = train_val_data[phase]
        row = metrics_df[metrics_df["Phase"] == phase].iloc[0]

        plot_train_val_actual_vs_predicted_crossplot(
            y_train=y_train,
            train_pred=train_pred,
            y_val=y_val,
            val_pred=val_pred,
            phase_string=phase,
            train_rmse=row["Train_RMSE"],
            train_mae=row["Train_MAE"],
            val_rmse=row["Val_Daily_RMSE"],
            val_mae=row["Val_Daily_MAE"],
            ax=ax,
            clip_negative=clip_negative,
        )

    fig.suptitle(
        f"{model_name} (block={block_horizon}D): Actual vs Predicted Crossplots",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    return metrics_df
