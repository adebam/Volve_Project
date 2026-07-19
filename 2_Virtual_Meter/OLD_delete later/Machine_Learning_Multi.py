from sklearn.metrics import mean_absolute_error, r2_score
#basics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np




#--------------------------------------------------------------------------------------------------------__#
def plot_train_val_predictions_time(
    y_train,
    train_pred,
    y_val,
    val_pred,
    phase_string,
    train_rmse=None,
    train_mae=None,
    val_rmse=None,
    val_mae=None,
    ax=None,
    start_of_val_date=None,
    clip_negative=True
):
    """
    Plot train actual, train predicted, validation actual,
    and validation predicted for a sklearn/Lasso model.

    This function does NOT calculate predictions.
    You must pass train_pred and val_pred into the function.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    color_map = {
        "BORE_OIL_VOL_STB_D": "darkgreen",
        "BORE_GAS_VOL_MSCF_D": "crimson",
        "BORE_WAT_VOL_STB_D": "blue"
    }

    color = color_map.get(phase_string, "blue")

    # --------------------------------------------------
    # 1. Convert actuals and predictions to Series
    # --------------------------------------------------
    y_train = pd.Series(y_train).astype(float).copy()
    y_val = pd.Series(y_val).astype(float).copy()

    train_pred = pd.Series(
        train_pred,
        index=y_train.index,
        name="train_pred"
    ).astype(float)

    val_pred = pd.Series(
        val_pred,
        index=y_val.index,
        name="val_pred"
    ).astype(float)

    if clip_negative:
        train_pred = train_pred.clip(lower=0)
        val_pred = val_pred.clip(lower=0)

    if start_of_val_date is None:
        start_of_val_date = y_val.index[0]

    # --------------------------------------------------
    # 2. Create axis if standalone
    # --------------------------------------------------
    is_standalone = ax is None

    if is_standalone:
        fig, ax = plt.subplots(figsize=(14, 6))

    # --------------------------------------------------
    # 3. Plot actuals and predictions
    # --------------------------------------------------
    ax.plot(
        y_train.index,
        y_train,
        label="Train Actual",
        linewidth=2,
        color=color,
        alpha=0.65
    )

    ax.plot(
        train_pred.index,
        train_pred,
        label="Train Predicted",
        linestyle="--",
        linewidth=2,
        color="gray"
    )

    ax.plot(
        y_val.index,
        y_val,
        label="Validation Actual",
        linewidth=2.5,
        color=color,
        marker="o",
        markersize=4
    )

    ax.plot(
        val_pred.index,
        val_pred,
        label="Validation Predicted",
        linestyle="--",
        linewidth=2.5,
        color="black",
        marker="s",
        markersize=3
    )

    # --------------------------------------------------
    # 4. Start of validation line
    # --------------------------------------------------
    ax.axvline(
        x=start_of_val_date,
        color="pink",
        linestyle=":",
        linewidth=2,
        label="Start of Validation"
    )

    ax.annotate(
        "Start of Validation",
        xy=(start_of_val_date, ax.get_ylim()[1]),
        xytext=(8, -20),
        textcoords="offset points",
        rotation=90,
        va="top",
        ha="left",
        color="pink",
        fontsize=10,
        fontweight="bold"
    )

    # --------------------------------------------------
    # 5. Title
    # --------------------------------------------------
    title = f"{phase_string}"

    if val_rmse is not None and val_mae is not None:
        title += f" | Val RMSE: {val_rmse:.2f}, Val MAE: {val_mae:.2f}"
    elif val_rmse is not None:
        title += f" | Val RMSE: {val_rmse:.2f}"

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(phase_string, fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    if is_standalone:
        ax.set_xlabel("Date / Time Step", fontsize=12)
        plt.tight_layout()
        plt.show()

    return ax
#----------------------------------------------------------------------------------------------------------------#
def plot_train_val_actual_vs_predicted_crossplot(
    y_train,
    train_pred,
    y_val,
    val_pred,
    phase_string,
    train_rmse=None,
    train_mae=None,
    val_rmse=None,
    val_mae=None,
    ax=None,
    clip_negative=True
):
    """
    Actual vs Predicted crossplot for train + validation
    for a sklearn/Lasso model.

    This function does NOT calculate predictions.
    It only plots the actual and predicted values you pass in.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    color_map = {
        "BORE_OIL_VOL_STB_D": "darkgreen",
        "BORE_GAS_VOL_MSCF_D": "crimson",
        "BORE_WAT_VOL_STB_D": "blue"
    }

    phase_color = color_map.get(phase_string, "blue")

    # --------------------------------------------------
    # 1. Convert to Series and align indexes
    # --------------------------------------------------
    y_train = pd.Series(y_train).astype(float).copy()
    y_val = pd.Series(y_val).astype(float).copy()

    train_pred = pd.Series(
        train_pred,
        index=y_train.index,
        name="train_pred"
    ).astype(float)

    val_pred = pd.Series(
        val_pred,
        index=y_val.index,
        name="val_pred"
    ).astype(float)

    if clip_negative:
        train_pred = train_pred.clip(lower=0)
        val_pred = val_pred.clip(lower=0)

    # --------------------------------------------------
    # 2. Build crossplot dataframe
    # --------------------------------------------------
    train_crossplot_df = pd.DataFrame({
        "actual": y_train,
        "predicted": train_pred,
        "period": "Train"
    }).dropna()

    val_crossplot_df = pd.DataFrame({
        "actual": y_val,
        "predicted": val_pred,
        "period": "Validation"
    }).dropna()

    crossplot_df = pd.concat(
        [train_crossplot_df, val_crossplot_df],
        axis=0
    ).dropna()

    # --------------------------------------------------
    # 3. Create axis if standalone
    # --------------------------------------------------
    is_standalone = ax is None

    if is_standalone:
        fig, ax = plt.subplots(figsize=(7, 7))

    # --------------------------------------------------
    # 4. Scatter points
    # --------------------------------------------------
    train_label = "Train Predicted"
    val_label = "Validation Predicted"

    if train_rmse is not None:
        train_label += f" | RMSE: {train_rmse:.2f}"

    if val_rmse is not None:
        val_label += f" | RMSE: {val_rmse:.2f}"

    ax.scatter(
        train_crossplot_df["actual"],
        train_crossplot_df["predicted"],
        label=train_label,
        color="gray",
        alpha=0.55,
        edgecolor="black",
        linewidth=0.4
    )

    ax.scatter(
        val_crossplot_df["actual"],
        val_crossplot_df["predicted"],
        label=val_label,
        color=phase_color,
        alpha=0.80,
        edgecolor="black",
        linewidth=0.5
    )

    # --------------------------------------------------
    # 5. 45-degree line
    # --------------------------------------------------
    min_value = min(
        crossplot_df["actual"].min(),
        crossplot_df["predicted"].min()
    )

    max_value = max(
        crossplot_df["actual"].max(),
        crossplot_df["predicted"].max()
    )

    padding = (max_value - min_value) * 0.05

    if padding == 0:
        padding = 1

    min_axis = min_value - padding
    max_axis = max_value + padding

    ax.plot(
        [min_axis, max_axis],
        [min_axis, max_axis],
        color="black",
        linestyle="--",
        linewidth=2,
        label="45° Line: Predicted = Actual"
    )

    ax.set_xlim(min_axis, max_axis)
    ax.set_ylim(min_axis, max_axis)
    ax.set_aspect("equal", adjustable="box")

    # --------------------------------------------------
    # 6. Styling
    # --------------------------------------------------
    ax.set_xlabel(f"Actual {phase_string}", fontsize=12)
    ax.set_ylabel(f"Predicted {phase_string}", fontsize=12)

    title = f"{phase_string}"

    if train_rmse is not None and val_rmse is not None:
        title += f"\nTrain RMSE: {train_rmse:.2f}, Val RMSE: {val_rmse:.2f}"

    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="best")

    if is_standalone:
        plt.tight_layout()
        plt.show()

    return crossplot_df, ax
#-------------------------------------------------------------------------------------------------------------------------------------------------

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_multi_phase_train_val_diagnostics(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    target_cols=None,
    model_name="Model",
    start_of_val_date=None,
    clip_negative=True,
    make_combined_plots=True,
):
    """
    Creates train/validation time plots and actual-vs-predicted crossplots
    for multi-output models.

    Works with:
    - MultiTaskLasso
    - MultiOutputRegressor(Lasso)
    - RandomForestRegressor multi-output
    - Any model where model.predict(X) returns shape (n_samples, n_targets)

    Requires your existing functions:
    - plot_train_val_predictions_time
    - plot_train_val_actual_vs_predicted_crossplot

    Returns
    -------
    train_pred_df : pandas DataFrame
        Train predictions in original target units.

    val_pred_df : pandas DataFrame
        Validation predictions in original target units.

    metrics_df : pandas DataFrame
        Train and validation RMSE, MAE, R2 for each phase.
    """

    if target_cols is None:
        target_cols = y_train.columns.tolist()

    # -----------------------------
    # 1. Predict
    # -----------------------------
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # -----------------------------
    # 2. Convert predictions to DataFrames
    # -----------------------------
    train_pred_df = pd.DataFrame(
        train_pred,
        index=X_train.index,
        columns=target_cols,
    )

    val_pred_df = pd.DataFrame(
        val_pred,
        index=X_val.index,
        columns=target_cols,
    )

    # -----------------------------
    # 3. Align actuals
    # -----------------------------
    y_train_aligned = y_train.loc[X_train.index, target_cols].copy()
    y_val_aligned = y_val.loc[X_val.index, target_cols].copy()

    if clip_negative:
        train_pred_df = train_pred_df.clip(lower=0)
        val_pred_df = val_pred_df.clip(lower=0)

    # -----------------------------
    # 4. Calculate metrics
    # -----------------------------
    metrics = []

    for phase in target_cols:
        train_rmse = root_mean_squared_error(
            y_train_aligned[phase],
            train_pred_df[phase],
        )
        train_mae = mean_absolute_error(
            y_train_aligned[phase],
            train_pred_df[phase],
        )
        train_r2 = r2_score(
            y_train_aligned[phase],
            train_pred_df[phase],
        )

        val_rmse = root_mean_squared_error(
            y_val_aligned[phase],
            val_pred_df[phase],
        )
        val_mae = mean_absolute_error(
            y_val_aligned[phase],
            val_pred_df[phase],
        )
        val_r2 = r2_score(
            y_val_aligned[phase],
            val_pred_df[phase],
        )

        metrics.append(
            {
                "Model": model_name,
                "Phase": phase,
                "Train_RMSE": train_rmse,
                "Train_MAE": train_mae,
                "Train_R2": train_r2,
                "Val_RMSE": val_rmse,
                "Val_MAE": val_mae,
                "Val_R2": val_r2,
            }
        )

    metrics_df = pd.DataFrame(metrics)

    # -----------------------------
    # 5. Plot time plots
    # -----------------------------
    if make_combined_plots:
        fig, axes = plt.subplots(
            nrows=len(target_cols),
            ncols=1,
            figsize=(15, 5 * len(target_cols)),
            sharex=False,
        )

        if len(target_cols) == 1:
            axes = [axes]

        for ax, phase in zip(axes, target_cols):
            row = metrics_df[metrics_df["Phase"] == phase].iloc[0]

            plot_train_val_predictions_time(
                y_train=y_train_aligned[phase],
                train_pred=train_pred_df[phase],
                y_val=y_val_aligned[phase],
                val_pred=val_pred_df[phase],
                phase_string=phase,
                train_rmse=row["Train_RMSE"],
                train_mae=row["Train_MAE"],
                val_rmse=row["Val_RMSE"],
                val_mae=row["Val_MAE"],
                ax=ax,
                start_of_val_date=start_of_val_date,
                clip_negative=False,
            )

        fig.suptitle(
            f"{model_name}: Train and Validation Actual vs Predicted",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

        # -----------------------------
        # 6. Plot crossplots
        # -----------------------------
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(target_cols),
            figsize=(7 * len(target_cols), 7),
            sharex=False,
            sharey=False,
        )

        if len(target_cols) == 1:
            axes = [axes]

        crossplot_dfs = {}

        for ax, phase in zip(axes, target_cols):
            row = metrics_df[metrics_df["Phase"] == phase].iloc[0]

            crossplot_df, ax = plot_train_val_actual_vs_predicted_crossplot(
                y_train=y_train_aligned[phase],
                train_pred=train_pred_df[phase],
                y_val=y_val_aligned[phase],
                val_pred=val_pred_df[phase],
                phase_string=phase,
                train_rmse=row["Train_RMSE"],
                train_mae=row["Train_MAE"],
                val_rmse=row["Val_RMSE"],
                val_mae=row["Val_MAE"],
                ax=ax,
                clip_negative=False,
            )

            crossplot_dfs[phase] = crossplot_df

        fig.suptitle(
            f"{model_name}: Actual vs Predicted Crossplots",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    return train_pred_df, val_pred_df, metrics_df

#---------------------------------------------------------------------------------------------------------#
def recursive_forecast_multiphase(
    model,
    feature_pipe,
    history_df,
    future_exog_df,
    target_cols,
    model_feature_cols=None,
    horizon=None,
    clip_negative=True,
    verbose=True,
):
    import numpy as np
    import pandas as pd

    history = history_df.copy()
    future = future_exog_df.copy()

    if horizon is None:
        horizon = len(future)

    future = future.iloc[:horizon].copy()

    # Make sure target columns exist in future dataframe
    for col in target_cols:
        if col not in future.columns:
            future[col] = np.nan

    forecasts = []

    for step, forecast_date in enumerate(future.index, start=1):

        # One future row
        next_row = future.loc[[forecast_date]].copy()

        # --------------------------------------------------
        # IMPORTANT:
        # Fill target NaNs temporarily before feature_pipe.transform().
        # This prevents LagFeatures from failing.
        # These values are only placeholders and will be overwritten.
        # --------------------------------------------------
        last_known_targets = history[target_cols].iloc[-1]

        for col in target_cols:
            if pd.isna(next_row.loc[forecast_date, col]):
                next_row.loc[forecast_date, col] = last_known_targets[col]

        # Combine history with the next forecast row
        temp_df = pd.concat([history, next_row], axis=0)


        # Build features
        temp_feat = feature_pipe.transform(temp_df)
        print("after feature pipe line transformation")


        if forecast_date not in temp_feat.index:
            raise ValueError(
                f"{forecast_date} was dropped by feature_pipe. "
                "Check missing future exogenous variables or DropMissingData."
            )

        X_next = temp_feat.loc[[forecast_date]].copy()


        # Drop target columns just in case they survived the pipeline
        X_next = X_next.drop(columns=target_cols, errors="ignore")

        # Match training feature columns
        if model_feature_cols is not None:
            missing_cols = [
                col for col in model_feature_cols
                if col not in X_next.columns
            ]

            if missing_cols:
                raise ValueError(
                    f"Missing model feature columns at {forecast_date}: "
                    f"{missing_cols}"
                )

            X_next = X_next.reindex(columns=model_feature_cols)

        # Predict
        pred = model.predict(X_next).reshape(-1)

        if clip_negative:
            pred = np.maximum(pred, 0)

        pred_series = pd.Series(
            pred,
            index=target_cols,
            name=forecast_date,
        )

        forecasts.append(pred_series)

        # --------------------------------------------------
        # Overwrite placeholder targets with actual prediction
        # --------------------------------------------------
        for col, value in zip(target_cols, pred):
            next_row.loc[forecast_date, col] = value

        # Add predicted row to recursive history
        history = pd.concat([history, next_row], axis=0)

        if verbose:
            print(f"Step {step}/{horizon}: forecasted {forecast_date}")

    forecast_df = pd.DataFrame(forecasts)

    return forecast_df, history


#---------------------------------------------------------------------------------------------------------#

def walk_forward_recursive_forecast_multiphase(
    model,
    feature_pipe,
    train_data,
    val_data,
    target_cols,
    model_feature_cols,
    forecast_horizon=30,
    exog_cols=None,
    clip_negative=True,
    verbose=True,
):
    """
    Walk-forward recursive validation for multiphase forecasting.

    For each validation block:
    1. Forecast the next forecast_horizon rows recursively.
    2. Compare against actual validation data.
    3. Append the actual validation rows to history.
    4. Continue until all validation data is used.

    Parameters
    ----------
    model : fitted model
        Trained model, e.g. best_lasso_model.

    feature_pipe : fitted transformer or pipeline
        Feature engineering pipeline.

    train_data : pd.DataFrame
        Historical training data with target columns and exogenous columns.

    val_data : pd.DataFrame
        Validation dataframe with target columns and exogenous columns.

    target_cols : list
        Example:
        [
            "BORE_OIL_VOL_STB_D",
            "BORE_GAS_VOL_MSCF_D",
            "BORE_WAT_VOL_STB_D",
        ]

    model_feature_cols : list
        Columns used by the trained model. Usually X_train.columns.

    forecast_horizon : int
        Number of days/rows to forecast recursively before updating history
        with actual validation data.

    exog_cols : list or None
        Columns known in the future. If None, the function uses all validation
        columns except target_cols as future_exog_df.

    clip_negative : bool
        Whether to clip negative predictions to zero.

    verbose : bool
        Whether to print progress.

    Returns
    -------
    all_forecasts_df : pd.DataFrame
        Forecasts for the full validation period.

    all_recursive_history_df : pd.DataFrame
        Recursive histories from each forecast block.

    walk_forward_metrics_df : pd.DataFrame
        RMSE, MAE, and R2 for each phase over the full validation period.

    block_summary_df : pd.DataFrame
        Start/end dates and number of rows for each validation block.
    """

    train_data = train_data.copy().sort_index()
    val_data = val_data.copy().sort_index()

    history_df = train_data.copy()

    all_forecasts = []
    all_recursive_histories = []
    block_summaries = []

    n_val = len(val_data)

    if n_val == 0:
        raise ValueError("val_data is empty.")

    for block_id, start in enumerate(range(0, n_val, forecast_horizon), start=1):
        stop = min(start + forecast_horizon, n_val)

        actual_block_df = val_data.iloc[start:stop].copy()
        block_horizon = len(actual_block_df)

        if exog_cols is None:
            future_exog_df = actual_block_df.drop(
                columns=target_cols,
                errors="ignore",
            ).copy()
        else:
            future_exog_df = actual_block_df[exog_cols].copy()

        if verbose:
            print(
                f"\nBlock {block_id}: forecasting rows {start} to {stop - 1} "
                f"({block_horizon} steps)"
            )
            print(f"Forecast start: {actual_block_df.index.min()}")
            print(f"Forecast end:   {actual_block_df.index.max()}")

        forecast_block_df, recursive_history_block_df = recursive_forecast_multiphase(
            model=model,
            feature_pipe=feature_pipe,
            history_df=history_df.copy(),
            future_exog_df=future_exog_df.copy(),
            target_cols=target_cols,
            model_feature_cols=model_feature_cols,
            horizon=block_horizon,
            clip_negative=clip_negative,
            verbose=verbose,
        )

        forecast_block_df = forecast_block_df.copy()
        forecast_block_df["walk_forward_block"] = block_id

        recursive_history_block_df = recursive_history_block_df.copy()
        recursive_history_block_df["walk_forward_block"] = block_id

        all_forecasts.append(forecast_block_df)
        all_recursive_histories.append(recursive_history_block_df)

        block_summaries.append(
            {
                "block": block_id,
                "start_row": start,
                "end_row": stop - 1,
                "n_forecast_steps": block_horizon,
                "start_date": actual_block_df.index.min(),
                "end_date": actual_block_df.index.max(),
            }
        )

        # Important:
        # After forecasting this block, update history with ACTUAL validation data,
        # not predicted data.
        history_df = pd.concat([history_df, actual_block_df], axis=0)

        # Avoid duplicate index problems
        history_df = history_df[~history_df.index.duplicated(keep="last")]
        history_df = history_df.sort_index()

    all_forecasts_df = pd.concat(all_forecasts, axis=0).sort_index()
    all_recursive_history_df = pd.concat(all_recursive_histories, axis=0).sort_index()
    block_summary_df = pd.DataFrame(block_summaries)

    # Calculate full validation metrics
    common_idx = val_data.index.intersection(all_forecasts_df.index)

    metrics = []

    for col in target_cols:
        y_true = val_data.loc[common_idx, col]
        y_pred = all_forecasts_df.loc[common_idx, col]

        valid_mask = y_true.notna() & y_pred.notna()

        y_true_valid = y_true.loc[valid_mask]
        y_pred_valid = y_pred.loc[valid_mask]

        metrics.append(
            {
                "Phase": col,
                "WalkForward_Val_RMSE": root_mean_squared_error(
                    y_true_valid,
                    y_pred_valid,
                ),
                "WalkForward_Val_MAE": mean_absolute_error(
                    y_true_valid,
                    y_pred_valid,
                ),
                "WalkForward_Val_R2": r2_score(
                    y_true_valid,
                    y_pred_valid,
                ),
                "N_Validation_Points": len(y_true_valid),
            }
        )

    walk_forward_metrics_df = pd.DataFrame(metrics)

    return (
        all_forecasts_df,
        all_recursive_history_df,
        walk_forward_metrics_df,
        block_summary_df,
    )

#---------------------------------------------------------------------------------------------------------#
def plot_multi_phase_train_walkforward_val_diagnostics(
    model,
    X_train,
    y_train,
    y_val,
    walk_forward_forecast_df,
    target_cols=None,
    model_name="Model",
    start_of_val_date=None,
    clip_negative=True,
    make_combined_plots=True,
    walk_forward_metrics_df=None,
):
    """
    Creates train / walk-forward-validation time plots and actual-vs-predicted
    crossplots for multi-output models.

    This version is for recursive / walk-forward validation.

    Train prediction:
        model.predict(X_train)

    Validation prediction:
        already supplied from walk_forward_forecast_df

    Works with:
    - MultiTaskLasso
    - MultiOutputRegressor(Lasso)
    - RandomForestRegressor multi-output
    - GradientBoosting wrapped in MultiOutputRegressor
    - Any model where model.predict(X) returns shape (n_samples, n_targets)

    Requires existing functions:
    - plot_train_val_predictions_time
    - plot_train_val_actual_vs_predicted_crossplot

    Returns
    -------
    train_pred_df : pandas DataFrame
        Train predictions in original target units.

    val_pred_df : pandas DataFrame
        Walk-forward validation predictions in original target units.

    metrics_df : pandas DataFrame
        Train metrics plus walk-forward validation metrics for each phase.
    """

    if target_cols is None:
        target_cols = y_train.columns.tolist()

    # -----------------------------
    # 1. Predict train only
    # -----------------------------
    train_pred = model.predict(X_train)

    # -----------------------------
    # 2. Convert train predictions to DataFrame
    # -----------------------------
    train_pred_df = pd.DataFrame(
        train_pred,
        index=X_train.index,
        columns=target_cols,
    )

    # -----------------------------
    # 3. Validation predictions already exist
    # -----------------------------
    val_pred_df = walk_forward_forecast_df[target_cols].copy()

    # -----------------------------
    # 4. Align actuals
    # -----------------------------
    y_train_aligned = y_train.loc[X_train.index, target_cols].copy()

    # Use only validation dates that exist in both y_val and walk_forward_forecast_df
    common_val_index = y_val.index.intersection(val_pred_df.index)

    y_val_aligned = y_val.loc[common_val_index, target_cols].copy()
    val_pred_df = val_pred_df.loc[common_val_index, target_cols].copy()

    if clip_negative:
        train_pred_df = train_pred_df.clip(lower=0)
        val_pred_df = val_pred_df.clip(lower=0)

    # -----------------------------
    # 5. Calculate train metrics
    #    Use walk_forward_metrics_df for validation metrics if provided
    # -----------------------------
    metrics = []

    for phase in target_cols:
        train_rmse = root_mean_squared_error(
            y_train_aligned[phase],
            train_pred_df[phase],
        )
        train_mae = mean_absolute_error(
            y_train_aligned[phase],
            train_pred_df[phase],
        )
        train_r2 = r2_score(
            y_train_aligned[phase],
            train_pred_df[phase],
        )

        if walk_forward_metrics_df is not None:
            metric_row = walk_forward_metrics_df[
                walk_forward_metrics_df["Phase"] == phase
            ]

            if len(metric_row) == 0:
                raise ValueError(
                    f"Phase '{phase}' not found in walk_forward_metrics_df."
                )

            metric_row = metric_row.iloc[0]

            val_rmse = metric_row["WalkForward_Val_RMSE"]
            val_mae = metric_row["WalkForward_Val_MAE"]
            val_r2 = metric_row["WalkForward_Val_R2"]

        else:
            val_rmse = root_mean_squared_error(
                y_val_aligned[phase],
                val_pred_df[phase],
            )
            val_mae = mean_absolute_error(
                y_val_aligned[phase],
                val_pred_df[phase],
            )
            val_r2 = r2_score(
                y_val_aligned[phase],
                val_pred_df[phase],
            )

        metrics.append(
            {
                "Model": model_name,
                "Phase": phase,
                "Train_RMSE": train_rmse,
                "Train_MAE": train_mae,
                "Train_R2": train_r2,
                "WalkForward_Val_RMSE": val_rmse,
                "WalkForward_Val_MAE": val_mae,
                "WalkForward_Val_R2": val_r2,
                # Aliases so your old plotting helper can still use Val_RMSE / Val_MAE
                "Val_RMSE": val_rmse,
                "Val_MAE": val_mae,
                "Val_R2": val_r2,
            }
        )

    metrics_df = pd.DataFrame(metrics)

    # -----------------------------
    # 6. Plot time plots
    # -----------------------------
    if make_combined_plots:
        fig, axes = plt.subplots(
            nrows=len(target_cols),
            ncols=1,
            figsize=(15, 5 * len(target_cols)),
            sharex=False,
        )

        if len(target_cols) == 1:
            axes = [axes]

        for ax, phase in zip(axes, target_cols):
            row = metrics_df[metrics_df["Phase"] == phase].iloc[0]

            plot_train_val_predictions_time(
                y_train=y_train_aligned[phase],
                train_pred=train_pred_df[phase],
                y_val=y_val_aligned[phase],
                val_pred=val_pred_df[phase],
                phase_string=phase,
                train_rmse=row["Train_RMSE"],
                train_mae=row["Train_MAE"],
                val_rmse=row["WalkForward_Val_RMSE"],
                val_mae=row["WalkForward_Val_MAE"],
                ax=ax,
                start_of_val_date=start_of_val_date,
                clip_negative=False,
            )

        fig.suptitle(
            f"{model_name}: Train Fit and Walk-Forward Validation Actual vs Predicted",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

        # -----------------------------
        # 7. Plot crossplots
        # -----------------------------
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(target_cols),
            figsize=(7 * len(target_cols), 7),
            sharex=False,
            sharey=False,
        )

        if len(target_cols) == 1:
            axes = [axes]

        crossplot_dfs = {}

        for ax, phase in zip(axes, target_cols):
            row = metrics_df[metrics_df["Phase"] == phase].iloc[0]

            crossplot_df, ax = plot_train_val_actual_vs_predicted_crossplot(
                y_train=y_train_aligned[phase],
                train_pred=train_pred_df[phase],
                y_val=y_val_aligned[phase],
                val_pred=val_pred_df[phase],
                phase_string=phase,
                train_rmse=row["Train_RMSE"],
                train_mae=row["Train_MAE"],
                val_rmse=row["WalkForward_Val_RMSE"],
                val_mae=row["WalkForward_Val_MAE"],
                ax=ax,
                clip_negative=False,
            )

            crossplot_dfs[phase] = crossplot_df

        fig.suptitle(
            f"{model_name}: Actual vs Predicted Crossplots",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    return train_pred_df, val_pred_df, metrics_df

#---------------------------------------------------------------------------------------------------------#
import pandas as pd


def _unwrap_outer_model(model):
    """
    Unwraps outer fitted wrappers:
    - GridSearchCV / RandomizedSearchCV
    - Pipeline
    - TransformedTargetRegressor
    """

    while True:
        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_
            continue

        if hasattr(model, "named_steps"):
            model = list(model.named_steps.values())[-1]
            continue

        if hasattr(model, "regressor_"):
            model = model.regressor_
            continue

        break

    return model


def _unwrap_single_target_estimator(estimator):
    """
    Unwraps each target estimator inside MultiOutputRegressor.

    Handles cases like:
    Pipeline([...,
        ("gb", GradientBoostingRegressor(...))
    ])
    """

    while True:
        if hasattr(estimator, "named_steps"):
            estimator = list(estimator.named_steps.values())[-1]
            continue

        if hasattr(estimator, "regressor_"):
            estimator = estimator.regressor_
            continue

        break

    return estimator


def get_multioutput_tree_feature_importance(model, feature_names, target_cols):
    """
    Gets feature importance from nested multi-output tree models.

    Handles:
    - TransformedTargetRegressor
    - MultiOutputRegressor
    - Pipeline inside each target estimator
    - GradientBoostingRegressor / RandomForestRegressor style feature_importances_
    """

    model = _unwrap_outer_model(model)

    if not hasattr(model, "estimators_"):
        raise AttributeError(
            "After unwrapping, the model does not have estimators_. "
            f"Current model type is: {type(model)}"
        )

    importance_dfs = []

    for target_name, estimator in zip(target_cols, model.estimators_):
        final_estimator = _unwrap_single_target_estimator(estimator)

        if not hasattr(final_estimator, "feature_importances_"):
            raise ValueError(
                f"The final estimator for {target_name} does not have feature_importances_. "
                f"Final estimator type: {type(final_estimator)}"
            )

        importances = final_estimator.feature_importances_

        if len(importances) != len(feature_names):
            raise ValueError(
                f"Feature name length mismatch for {target_name}.\n"
                f"Number of feature names: {len(feature_names)}\n"
                f"Number of importances: {len(importances)}\n\n"
                "This usually means the pipeline changed the number of columns. "
                "You may need to use transformed feature names instead of X_train.columns."
            )

        imp_df = pd.DataFrame(
            {
                "Feature": feature_names,
                target_name: importances,
            }
        )

        importance_dfs.append(imp_df)

    importance_df = importance_dfs[0]

    for df in importance_dfs[1:]:
        importance_df = importance_df.merge(df, on="Feature", how="outer")

    importance_df = importance_df.fillna(0)

    importance_df["Mean_Importance"] = importance_df[target_cols].mean(axis=1)

    importance_df = importance_df.sort_values(
        "Mean_Importance",
        ascending=False
    ).reset_index(drop=True)

    return importance_df
#---------------------------------------------------------------------------------------------------------------