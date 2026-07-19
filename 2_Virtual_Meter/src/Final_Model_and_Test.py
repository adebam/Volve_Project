import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator



from Machine_Learning_Multi import (
    plot_train_val_predictions_time,
    plot_train_val_actual_vs_predicted_crossplot,
)

from sklearn.metrics import (mean_absolute_error,r2_score,root_mean_squared_error,)

#------------------------------------------------------------------------------------#
def recursive_forecast_hybrid_multiphase(
    phase_model_map,
    feature_pipe,
    history_df,
    future_exog_df,
    target_cols,
    model_feature_cols,
    horizon=None,
    clip_negative=True,
    verbose=True,
):
    """
    Recursive multiphase forecast where each target can take its prediction
    from a different fitted multitask/multi-output model.

    Each selected model still predicts all target phases, but only the assigned
    target output is retained.

    Parameters
    ----------
    phase_model_map : dict
        Maps each target to a fitted model and the output column to use.

        Example:
        {
            "BORE_OIL_VOL_STB_D": {
                "model": best_multitask_lasso_model,
                "prediction_index": 0,
            },
            "BORE_GAS_VOL_MSCF_D": {
                "model": best_multitask_gb_model,
                "prediction_index": 1,
            },
            "BORE_WAT_VOL_STB_D": {
                "model": best_multitask_lgbm_model,
                "prediction_index": 2,
            },
        }

    feature_pipe : fitted transformer or pipeline
        Fitted feature-engineering pipeline.

    history_df : pd.DataFrame
        Historical data containing targets and exogenous variables.

    future_exog_df : pd.DataFrame
        Future exogenous variables for the forecast dates.

    target_cols : list
        Target columns in the same order used during model training.

    model_feature_cols : list-like
        Exact feature columns used to train the models.

    horizon : int or None
        Number of recursive rows to forecast. When None, all rows in
        future_exog_df are forecast.

    clip_negative : bool
        Clip predictions below zero to zero.

    verbose : bool
        Print recursive progress.

    Returns
    -------
    forecast_df : pd.DataFrame
        Hybrid recursive predictions.

    recursive_history_df : pd.DataFrame
        Original history plus recursively predicted future rows.
    """

    history_df = history_df.copy().sort_index()
    future_exog_df = future_exog_df.copy().sort_index()
    model_feature_cols = list(model_feature_cols)

    if horizon is None:
        horizon = len(future_exog_df)

    if horizon <= 0:
        raise ValueError("horizon must be greater than zero.")

    if horizon > len(future_exog_df):
        raise ValueError(
            f"horizon={horizon}, but future_exog_df contains only "
            f"{len(future_exog_df)} rows."
        )

    if history_df.index.has_duplicates:
        raise ValueError("history_df contains duplicate index values.")

    if future_exog_df.index.has_duplicates:
        raise ValueError("future_exog_df contains duplicate index values.")

    missing_targets = [
        target
        for target in target_cols
        if target not in phase_model_map
    ]

    if missing_targets:
        raise ValueError(
            f"phase_model_map is missing configurations for: "
            f"{missing_targets}"
        )

    for target in target_cols:
        config = phase_model_map[target]

        if "model" not in config:
            raise ValueError(
                f"phase_model_map['{target}'] has no 'model'."
            )

        if "prediction_index" not in config:
            raise ValueError(
                f"phase_model_map['{target}'] has no "
                "'prediction_index'."
            )

        prediction_index = config["prediction_index"]

        if not isinstance(prediction_index, int):
            raise TypeError(
                f"prediction_index for {target} must be an integer."
            )

    recursive_history_df = history_df.copy()
    forecast_rows = []

    forecast_dates = future_exog_df.index[:horizon]

    for step_number, forecast_date in enumerate(forecast_dates, start=1):
        # Exogenous information available for this future date
        next_row_df = future_exog_df.loc[[forecast_date]].copy()

        # Targets must be unknown before predicting this date
        for target in target_cols:
            next_row_df[target] = np.nan

        # Match the historical dataframe structure where possible
        for col in recursive_history_df.columns:
            if col not in next_row_df.columns:
                next_row_df[col] = np.nan

        # Preserve additional exogenous columns that may not yet exist
        # in recursive_history_df.
        working_history_df = pd.concat(
            [recursive_history_df, next_row_df],
            axis=0,
            sort=False,
        )

        working_history_df = working_history_df[
            ~working_history_df.index.duplicated(keep="last")
        ].sort_index()

        # Build lagged, rolling, expanding and calendar features
        transformed_df = feature_pipe.transform(working_history_df)

        if not isinstance(transformed_df, pd.DataFrame):
            raise TypeError(
                "feature_pipe.transform() must return a pandas DataFrame "
                "so features can be selected by name."
            )

        if forecast_date not in transformed_df.index:
            raise ValueError(
                f"The forecast date {forecast_date} was removed by "
                "feature_pipe.transform(). This may be caused by "
                "DropMissingData or insufficient historical lookback."
            )

        missing_features = [
            col
            for col in model_feature_cols
            if col not in transformed_df.columns
        ]

        if missing_features:
            raise ValueError(
                "The transformed dataframe is missing model features: "
                f"{missing_features[:20]}"
            )

        X_step = transformed_df.loc[
            [forecast_date],
            model_feature_cols,
        ]

        if X_step.isna().any().any():
            nan_cols = X_step.columns[
                X_step.isna().any()
            ].tolist()

            raise ValueError(
                f"NaN values found in model input for {forecast_date}. "
                f"Columns: {nan_cols[:20]}"
            )

        selected_predictions = {}

        # Each model may produce all three phases. We retain only the
        # output assigned to the current target.
        for target in target_cols:
            config = phase_model_map[target]

            phase_model = config["model"]
            prediction_index = config["prediction_index"]

            prediction_array = np.asarray(
                phase_model.predict(X_step)
            )

            if prediction_array.ndim == 2:
                if prediction_array.shape[0] != 1:
                    raise ValueError(
                        f"Expected one prediction row from the model "
                        f"assigned to {target}, but received shape "
                        f"{prediction_array.shape}."
                    )

                if prediction_index >= prediction_array.shape[1]:
                    raise IndexError(
                        f"prediction_index={prediction_index} for {target} "
                        f"is outside prediction shape "
                        f"{prediction_array.shape}."
                    )

                prediction = prediction_array[0, prediction_index]

            elif prediction_array.ndim == 1:
                # A true single-output model normally returns shape (1,)
                if prediction_array.size == 1:
                    prediction = prediction_array[0]

                # Supports a model that returns one-dimensional multi-output
                else:
                    if prediction_index >= prediction_array.size:
                        raise IndexError(
                            f"prediction_index={prediction_index} for "
                            f"{target} is outside prediction shape "
                            f"{prediction_array.shape}."
                        )

                    prediction = prediction_array[prediction_index]

            else:
                raise ValueError(
                    f"Unexpected prediction shape for {target}: "
                    f"{prediction_array.shape}"
                )

            prediction = float(prediction)

            if not np.isfinite(prediction):
                raise ValueError(
                    f"The model assigned to {target} produced a "
                    f"non-finite prediction on {forecast_date}: "
                    f"{prediction}"
                )

            if clip_negative:
                prediction = max(prediction, 0.0)

            selected_predictions[target] = prediction

        # Create the combined hybrid row:
        # oil from one model, gas from another, water from another.
        predicted_row_df = next_row_df.copy()

        for target in target_cols:
            predicted_row_df.loc[
                forecast_date,
                target,
            ] = selected_predictions[target]

        # Keep the recursive dataframe aligned with the available columns.
        recursive_history_df = pd.concat(
            [recursive_history_df, predicted_row_df],
            axis=0,
            sort=False,
        )

        recursive_history_df = recursive_history_df[
            ~recursive_history_df.index.duplicated(keep="last")
        ].sort_index()

        forecast_rows.append(
            pd.DataFrame(
                [selected_predictions],
                index=[forecast_date],
                columns=target_cols,
            )
        )

        if verbose:
            formatted_predictions = ", ".join(
                f"{target}={selected_predictions[target]:,.2f}"
                for target in target_cols
            )

            print(
                f"Recursive step {step_number}/{horizon} | "
                f"{forecast_date} | {formatted_predictions}"
            )

    forecast_df = pd.concat(forecast_rows, axis=0)

    return forecast_df, recursive_history_df
#------------------------------------------------------------------------------------#
def walk_forward_recursive_forecast_hybrid_multiphase(
    phase_model_map,
    feature_pipe,
    train_data,
    val_data,
    target_cols,
    model_feature_cols,
    forecast_horizon=30,
    exog_cols=None,
    clip_negative=True,
    verbose=True,
    model_name="Hybrid Multiphase Model",
):
    """
    Walk-forward recursive validation using a different multitask or
    multi-output model output for each production phase.

    For each validation block:
    1. Forecast the next forecast_horizon rows recursively.
    2. Select each phase from its assigned model.
    3. Compare the combined hybrid forecast against actual validation data.
    4. Append the actual validation rows to history.
    5. Continue until all validation data is used.

    Parameters
    ----------
    phase_model_map : dict
        Maps each target to its selected model and output index.

    feature_pipe : fitted transformer or pipeline
        Fitted feature-engineering pipeline.

    train_data : pd.DataFrame
        Historical training data.

    val_data : pd.DataFrame
        Validation data containing actual targets and future exogenous inputs.

    target_cols : list
        Target columns in training-output order.

    model_feature_cols : list-like
        Columns expected by the fitted models. Usually X_train.columns.

    forecast_horizon : int
        Number of rows recursively forecast before history is updated with
        actual validation observations.

    exog_cols : list or None
        Future-known input columns. When None, all columns except target_cols
        are treated as future-known exogenous columns.

    clip_negative : bool
        Clip negative rate predictions to zero.

    verbose : bool
        Print block and recursive-step progress.

    Returns
    -------
    all_forecasts_df : pd.DataFrame
        Hybrid forecasts covering the full validation period.

    all_recursive_history_df : pd.DataFrame
        Recursive histories produced for all blocks.

    walk_forward_metrics_df : pd.DataFrame
        Full-validation metrics for each phase.

    block_summary_df : pd.DataFrame
        Start date, end date and number of rows for every forecast block.
    """

    train_data = train_data.copy().sort_index()
    val_data = val_data.copy().sort_index()
    model_feature_cols = list(model_feature_cols)

    if len(val_data) == 0:
        raise ValueError("val_data is empty.")

    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be greater than zero.")

    missing_targets = [
        target
        for target in target_cols
        if target not in phase_model_map
    ]

    if missing_targets:
        raise ValueError(
            f"phase_model_map is missing configurations for: "
            f"{missing_targets}"
        )

    history_df = train_data.copy()

    all_forecasts = []
    all_recursive_histories = []
    block_summaries = []

    n_val = len(val_data)

    for block_id, start in enumerate(
        range(0, n_val, forecast_horizon),
        start=1,
    ):
        stop = min(start + forecast_horizon, n_val)

        actual_block_df = val_data.iloc[start:stop].copy()
        block_horizon = len(actual_block_df)

        if exog_cols is None:
            future_exog_df = actual_block_df.drop(
                columns=target_cols,
                errors="ignore",
            ).copy()
        else:
            missing_exog = [
                col
                for col in exog_cols
                if col not in actual_block_df.columns
            ]

            if missing_exog:
                raise ValueError(
                    f"Validation block is missing exogenous columns: "
                    f"{missing_exog}"
                )

            future_exog_df = actual_block_df.loc[
                :,
                exog_cols,
            ].copy()

        if verbose:
            print(
                f"\nBlock {block_id}: forecasting rows "
                f"{start} to {stop - 1} "
                f"({block_horizon} steps)"
            )
            print(f"Forecast start: {actual_block_df.index.min()}")
            print(f"Forecast end:   {actual_block_df.index.max()}")

        (
            forecast_block_df,
            recursive_history_block_df,
        ) = recursive_forecast_hybrid_multiphase(
            phase_model_map=phase_model_map,
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

        recursive_history_block_df = (
            recursive_history_block_df.copy()
        )
        recursive_history_block_df[
            "walk_forward_block"
        ] = block_id

        all_forecasts.append(forecast_block_df)
        all_recursive_histories.append(
            recursive_history_block_df
        )

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

        # After evaluating this recursive block, use actual validation
        # observations as the starting history for the next block.
        history_df = pd.concat(
            [history_df, actual_block_df],
            axis=0,
        )

        history_df = history_df[
            ~history_df.index.duplicated(keep="last")
        ].sort_index()

    all_forecasts_df = pd.concat(
        all_forecasts,
        axis=0,
    ).sort_index()

    all_recursive_history_df = pd.concat(
        all_recursive_histories,
        axis=0,
    ).sort_index()

    block_summary_df = pd.DataFrame(block_summaries)

    common_idx = val_data.index.intersection(
        all_forecasts_df.index
    )

    metrics = []

    for col in target_cols:
        y_true = val_data.loc[common_idx, col]
        y_pred = all_forecasts_df.loc[common_idx, col]

        valid_mask = y_true.notna() & y_pred.notna()

        y_true_valid = y_true.loc[valid_mask]
        y_pred_valid = y_pred.loc[valid_mask]

        if len(y_true_valid) == 0:
            metrics.append(
                {
                    "Phase": col,
                    "WalkForward_Val_RMSE": np.nan,
                    "WalkForward_Val_MAE": np.nan,
                    "WalkForward_Val_R2": np.nan,
                    "N_Validation_Points": 0,
                }
            )
            continue

        # R2 is not defined for fewer than two observations.
        phase_r2 = (
            r2_score(y_true_valid, y_pred_valid)
            if len(y_true_valid) >= 2
            else np.nan
        )

        metrics.append(
            {
                "Phase": col,
                "WalkForward_Val_RMSE": (
                    root_mean_squared_error(
                        y_true_valid,
                        y_pred_valid,
                    )
                ),
                "WalkForward_Val_MAE": mean_absolute_error(
                    y_true_valid,
                    y_pred_valid,
                ),
                "WalkForward_Val_R2": phase_r2,
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
#-------------------------------------------------------------------------------------------#

def predict_hybrid_multiphase(
    phase_model_map,
    X,
    target_cols,
    clip_negative=True,
):
    """
    Predict each target using its assigned multitask/multi-output model.
    """

    predictions = {}

    for target in target_cols:
        config = phase_model_map[target]

        model = config["model"]
        prediction_index = config["prediction_index"]

        model_predictions = np.asarray(
            model.predict(X)
        )

        if model_predictions.ndim == 2:
            target_prediction = model_predictions[
                :,
                prediction_index,
            ]

        elif model_predictions.ndim == 1:
            if len(target_cols) == 1:
                target_prediction = model_predictions
            else:
                raise ValueError(
                    f"The model assigned to {target} returned a "
                    f"one-dimensional prediction with shape "
                    f"{model_predictions.shape}. Expected a "
                    "multi-output prediction."
                )

        else:
            raise ValueError(
                f"Unexpected prediction shape for {target}: "
                f"{model_predictions.shape}"
            )

        predictions[target] = target_prediction

    prediction_df = pd.DataFrame(
        predictions,
        index=X.index,
    )

    if clip_negative:
        prediction_df = prediction_df.clip(lower=0)

    return prediction_df
#-------------------------------------------------------------------------------------------#
def plot_multi_phase_train_walkforward_val_diagnostics_FINAL(
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
    train_pred_df=None,
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

    if train_pred_df is None:
        if model is None:
            raise ValueError(
                "Supply either model or train_pred_df."
            )
    
        train_pred = model.predict(X_train)
    
        train_pred_df = pd.DataFrame(
            train_pred,
            index=y_train.index,
            columns=target_cols,
        )
    
    else:
        train_pred_df = train_pred_df.copy()
    
        missing_targets = [
            col
            for col in target_cols
            if col not in train_pred_df.columns
        ]
    
        if missing_targets:
            raise ValueError(
                f"train_pred_df is missing target columns: "
                f"{missing_targets}"
            )
    
        train_pred_df = train_pred_df.loc[
            y_train.index,
            target_cols,
        ]

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
                # Aliases so the old plotting helper can still use Val_RMSE / Val_MAE
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
            f"{model_name}: Train_Val Fit and Walk-Forward Validation Actual vs Predicted on Test Data",
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




#----------------------------------------------------------------------------_#
