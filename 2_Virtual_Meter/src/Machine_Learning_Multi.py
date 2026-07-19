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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)

try:
    from sklearn.metrics import root_mean_squared_error

except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(
            mean_squared_error(y_true, y_pred)
        )


def _get_phase_feature_dataframe(
    X,
    phase,
    model,
    target_cols,
    drop_target_cols=True,
):
    """
    Return the appropriate feature DataFrame for one phase.

    X can be either:

    1. One shared DataFrame:

       X_train

    2. A dictionary containing different feature sets:

       {
           "BORE_OIL_VOL_STB_D": X_train_oil,
           "BORE_GAS_VOL_MSCF_D": X_train_gas,
           "BORE_WAT_VOL_STB_D": X_train_water,
       }
    """

    # --------------------------------------------------
    # Different X DataFrame for each phase
    # --------------------------------------------------
    if isinstance(X, dict):

        if phase not in X:
            raise KeyError(
                f"{phase} is missing from the feature dictionary."
            )

        X_phase = X[phase].copy()

    # --------------------------------------------------
    # Same X DataFrame for all phases
    # --------------------------------------------------
    else:
        X_phase = X.copy()

    if not isinstance(X_phase, pd.DataFrame):
        raise ValueError(
            f"Features for {phase} must be a pandas DataFrame."
        )

    # --------------------------------------------------
    # Remove target columns if they are still present
    # --------------------------------------------------
    if drop_target_cols:
        X_phase = X_phase.drop(
            columns=target_cols,
            errors="ignore",
        )

    # --------------------------------------------------
    # Match the columns used when the model was fitted
    # --------------------------------------------------
    expected_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if expected_features is not None:

        expected_features = list(expected_features)

        missing_features = [
            feature
            for feature in expected_features
            if feature not in X_phase.columns
        ]

        if missing_features:
            raise ValueError(
                f"The following features required by the "
                f"{phase} model are missing:\n"
                f"{missing_features}"
            )

        # Correct both feature selection and feature order
        X_phase = X_phase.loc[
            :,
            expected_features,
        ]

    return X_phase


def plot_multi_phase_train_val_diagnostics(
    models,
    X_train,
    y_train,
    X_val,
    y_val,
    target_cols=None,
    model_name="Independent Phase Models",
    start_of_val_date=None,
    clip_negative=True,
    make_combined_plots=True,
    drop_target_cols_from_X=True,
):
    """
    Create train/validation diagnostics for independently fitted
    oil, gas, and water models.

    Parameters
    ----------
    models : dict
        Dictionary containing one fitted model per target.

        Example:

        {
            "BORE_OIL_VOL_STB_D": oil_model,
            "BORE_GAS_VOL_MSCF_D": gas_model,
            "BORE_WAT_VOL_STB_D": water_model,
        }

    X_train, X_val : pandas DataFrame or dict
        A single shared feature DataFrame can be used for all models.

        Separate feature sets can also be supplied:

        {
            "BORE_OIL_VOL_STB_D": X_train_oil,
            "BORE_GAS_VOL_MSCF_D": X_train_gas,
            "BORE_WAT_VOL_STB_D": X_train_water,
        }

    y_train, y_val : pandas DataFrame
        Actual target values.

    target_cols : list of str or None
        Target columns and plotting order.

        If None, the dictionary keys from models are used.

    model_name : str
        Name shown in plots and the metrics table.

    start_of_val_date : datetime-like or None
        Date used for the validation boundary line.

    clip_negative : bool
        If True, negative predictions are changed to zero before
        calculating metrics and plotting.

    make_combined_plots : bool
        If True, creates time plots and crossplots.

    drop_target_cols_from_X : bool
        If True, target columns are removed from the feature
        DataFrames before prediction.

    Returns
    -------
    train_pred_df : pandas DataFrame
        Train predictions for all phases.

    val_pred_df : pandas DataFrame
        Validation predictions for all phases.

    metrics_df : pandas DataFrame
        Train and validation metrics for all phases.
    """

    # --------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------
    if not isinstance(models, dict) or len(models) == 0:
        raise ValueError(
            "models must be a non-empty dictionary."
        )

    if not isinstance(y_train, pd.DataFrame):
        raise ValueError(
            "y_train must be a pandas DataFrame."
        )

    if not isinstance(y_val, pd.DataFrame):
        raise ValueError(
            "y_val must be a pandas DataFrame."
        )

    if target_cols is None:
        target_cols = list(models.keys())
    else:
        target_cols = list(target_cols)

    missing_models = [
        phase
        for phase in target_cols
        if phase not in models
    ]

    if missing_models:
        raise KeyError(
            f"No fitted model was provided for: "
            f"{missing_models}"
        )

    missing_train_targets = [
        phase
        for phase in target_cols
        if phase not in y_train.columns
    ]

    missing_val_targets = [
        phase
        for phase in target_cols
        if phase not in y_val.columns
    ]

    if missing_train_targets:
        raise KeyError(
            f"Targets missing from y_train: "
            f"{missing_train_targets}"
        )

    if missing_val_targets:
        raise KeyError(
            f"Targets missing from y_val: "
            f"{missing_val_targets}"
        )

    # --------------------------------------------------
    # 2. Containers
    # --------------------------------------------------
    train_prediction_series = {}
    val_prediction_series = {}

    aligned_y_train = {}
    aligned_y_val = {}

    phase_X_train = {}
    phase_X_val = {}

    metrics = []

    # --------------------------------------------------
    # 3. Predict separately for every phase
    # --------------------------------------------------
    for phase in target_cols:

        model = models[phase]

        X_train_phase = _get_phase_feature_dataframe(
            X=X_train,
            phase=phase,
            model=model,
            target_cols=target_cols,
            drop_target_cols=drop_target_cols_from_X,
        )

        X_val_phase = _get_phase_feature_dataframe(
            X=X_val,
            phase=phase,
            model=model,
            target_cols=target_cols,
            drop_target_cols=drop_target_cols_from_X,
        )

        phase_X_train[phase] = X_train_phase
        phase_X_val[phase] = X_val_phase

        # --------------------------------------------------
        # Align actual values with this phase's X index
        # --------------------------------------------------
        missing_train_index = (
            X_train_phase.index.difference(y_train.index)
        )

        missing_val_index = (
            X_val_phase.index.difference(y_val.index)
        )

        if len(missing_train_index) > 0:
            raise ValueError(
                f"Some {phase} training feature indexes "
                f"are missing from y_train."
            )

        if len(missing_val_index) > 0:
            raise ValueError(
                f"Some {phase} validation feature indexes "
                f"are missing from y_val."
            )

        y_train_phase = (
            y_train
            .loc[X_train_phase.index, phase]
            .astype(float)
            .copy()
        )

        y_val_phase = (
            y_val
            .loc[X_val_phase.index, phase]
            .astype(float)
            .copy()
        )

        # --------------------------------------------------
        # Each model returns one-dimensional predictions
        # --------------------------------------------------
        train_pred = np.asarray(
            model.predict(X_train_phase)
        ).reshape(-1)

        val_pred = np.asarray(
            model.predict(X_val_phase)
        ).reshape(-1)

        if len(train_pred) != len(X_train_phase):
            raise ValueError(
                f"The {phase} model produced "
                f"{len(train_pred)} train predictions for "
                f"{len(X_train_phase)} rows."
            )

        if len(val_pred) != len(X_val_phase):
            raise ValueError(
                f"The {phase} model produced "
                f"{len(val_pred)} validation predictions for "
                f"{len(X_val_phase)} rows."
            )

        train_pred_series = pd.Series(
            train_pred,
            index=X_train_phase.index,
            name=phase,
            dtype=float,
        )

        val_pred_series = pd.Series(
            val_pred,
            index=X_val_phase.index,
            name=phase,
            dtype=float,
        )

        if clip_negative:
            train_pred_series = (
                train_pred_series.clip(lower=0)
            )

            val_pred_series = (
                val_pred_series.clip(lower=0)
            )

        train_prediction_series[phase] = (
            train_pred_series
        )

        val_prediction_series[phase] = (
            val_pred_series
        )

        aligned_y_train[phase] = y_train_phase
        aligned_y_val[phase] = y_val_phase

        # --------------------------------------------------
        # 4. Metrics for this independent model
        # --------------------------------------------------
        train_rmse = root_mean_squared_error(
            y_train_phase,
            train_pred_series,
        )

        train_mae = mean_absolute_error(
            y_train_phase,
            train_pred_series,
        )

        train_r2 = r2_score(
            y_train_phase,
            train_pred_series,
        )

        val_rmse = root_mean_squared_error(
            y_val_phase,
            val_pred_series,
        )

        val_mae = mean_absolute_error(
            y_val_phase,
            val_pred_series,
        )

        val_r2 = r2_score(
            y_val_phase,
            val_pred_series,
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

    # --------------------------------------------------
    # 5. Combine predictions into DataFrames
    # --------------------------------------------------
    train_pred_df = pd.concat(
        train_prediction_series.values(),
        axis=1,
    )

    val_pred_df = pd.concat(
        val_prediction_series.values(),
        axis=1,
    )

    train_pred_df = train_pred_df.reindex(
        columns=target_cols
    )

    val_pred_df = val_pred_df.reindex(
        columns=target_cols
    )

    metrics_df = pd.DataFrame(metrics)

    # --------------------------------------------------
    # 6. Combined time plots
    # --------------------------------------------------
    if make_combined_plots:

        fig, axes = plt.subplots(
            nrows=len(target_cols),
            ncols=1,
            figsize=(15, 5 * len(target_cols)),
            sharex=False,
        )

        if len(target_cols) == 1:
            axes = [axes]

        for ax, phase in zip(
            axes,
            target_cols,
        ):
            metric_row = (
                metrics_df[
                    metrics_df["Phase"] == phase
                ]
                .iloc[0]
            )

            plot_train_val_predictions_time(
                y_train=aligned_y_train[phase],
                train_pred=train_prediction_series[phase],
                y_val=aligned_y_val[phase],
                val_pred=val_prediction_series[phase],
                phase_string=phase,
                train_rmse=metric_row["Train_RMSE"],
                train_mae=metric_row["Train_MAE"],
                val_rmse=metric_row["Val_RMSE"],
                val_mae=metric_row["Val_MAE"],
                ax=ax,
                start_of_val_date=start_of_val_date,
                clip_negative=False,
            )

        fig.suptitle(
            f"{model_name}: "
            f"Train and Validation Actual vs Predicted",
            fontsize=16,
            y=1.02,
        )

        plt.tight_layout()
        plt.show()

        # --------------------------------------------------
        # 7. Combined actual-vs-predicted crossplots
        # --------------------------------------------------
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(target_cols),
            figsize=(7 * len(target_cols), 7),
            sharex=False,
            sharey=False,
        )

        if len(target_cols) == 1:
            axes = [axes]

        for ax, phase in zip(
            axes,
            target_cols,
        ):
            metric_row = (
                metrics_df[
                    metrics_df["Phase"] == phase
                ]
                .iloc[0]
            )

            plot_train_val_actual_vs_predicted_crossplot(
                y_train=aligned_y_train[phase],
                train_pred=train_prediction_series[phase],
                y_val=aligned_y_val[phase],
                val_pred=val_prediction_series[phase],
                phase_string=phase,
                train_rmse=metric_row["Train_RMSE"],
                train_mae=metric_row["Train_MAE"],
                val_rmse=metric_row["Val_RMSE"],
                val_mae=metric_row["Val_MAE"],
                ax=ax,
                clip_negative=False,
            )

        fig.suptitle(
            f"{model_name}: Actual vs Predicted Crossplots",
            fontsize=16,
            y=1.02,
        )

        plt.tight_layout()
        plt.show()

    return (
        train_pred_df,
        val_pred_df,
        metrics_df,
    )

#---------------------------------------------------------------------------------------------------------#
def _get_phase_item(item, phase, item_name):
    """
    Return a phase-specific object if item is a dictionary,
    otherwise return the shared object.
    """
    if isinstance(item, dict):
        if phase not in item:
            raise KeyError(
                f"{phase} is missing from {item_name}."
            )

        return item[phase]

    return item


def _get_model_feature_columns(
    model,
    transformed_data,
    model_feature_cols,
    phase,
):
    """
    Resolve the feature columns for one phase model.

    model_feature_cols can be:
    - one shared list
    - a dictionary containing one list per phase
    - None, in which case model.feature_names_in_ is used
    """

    if isinstance(model_feature_cols, dict):
        if phase not in model_feature_cols:
            raise KeyError(
                f"{phase} is missing from model_feature_cols."
            )

        phase_feature_cols = list(
            model_feature_cols[phase]
        )

    elif model_feature_cols is not None:
        phase_feature_cols = list(model_feature_cols)

    else:
        model_feature_names = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if model_feature_names is None:
            raise ValueError(
                f"No model_feature_cols were supplied for {phase}, "
                "and the fitted model does not contain "
                "feature_names_in_."
            )

        phase_feature_cols = list(model_feature_names)

    missing_features = [
        col
        for col in phase_feature_cols
        if col not in transformed_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"The transformed data for {phase} is missing "
            f"{len(missing_features)} required model features:\n"
            f"{missing_features}"
        )

    return phase_feature_cols


def recursive_forecast_multiphase(
    models,
    feature_pipe,
    history_df,
    future_exog_df,
    target_cols=None,
    model_feature_cols=None,
    horizon=None,
    clip_negative=True,
    verbose=True,
):
    """
    Recursively forecast oil, gas, and water using three independently
    fitted models.

    At every forecast step:

    1. Add the future exogenous row with unknown targets.
    2. Transform the full history using the fitted feature pipeline.
    3. Predict oil with the oil model.
    4. Predict gas with the gas model.
    5. Predict water with the water model.
    6. Insert all three predictions into history simultaneously.
    7. Continue to the next forecast step.

    Parameters
    ----------
    models : dict
        One fitted model per target.

        Example:

        {
            "BORE_OIL_VOL_STB_D": oil_model,
            "BORE_GAS_VOL_MSCF_D": gas_model,
            "BORE_WAT_VOL_STB_D": water_model,
        }

    feature_pipe : fitted transformer/pipeline or dict
        A shared feature pipeline, or one pipeline per phase.

        Shared:

        feature_pipe

        Separate:

        {
            "BORE_OIL_VOL_STB_D": oil_feature_pipe,
            "BORE_GAS_VOL_MSCF_D": gas_feature_pipe,
            "BORE_WAT_VOL_STB_D": water_feature_pipe,
        }

    history_df : pandas DataFrame
        Historical raw data containing targets and exogenous columns.

    future_exog_df : pandas DataFrame
        Future known exogenous variables.

    target_cols : list or None
        Target names and prediction order. If None, model keys are used.

    model_feature_cols : list, dict, or None
        Columns used to train each model.

        Shared feature set:

        model_feature_cols = X_train.columns.tolist()

        Separate feature sets:

        {
            "BORE_OIL_VOL_STB_D": X_train_oil.columns.tolist(),
            "BORE_GAS_VOL_MSCF_D": X_train_gas.columns.tolist(),
            "BORE_WAT_VOL_STB_D": X_train_water.columns.tolist(),
        }

        If None, the function attempts to use model.feature_names_in_.

    horizon : int or None
        Number of future rows to forecast. If None, all rows in
        future_exog_df are forecast.

    clip_negative : bool
        If True, negative rate predictions are changed to zero.

    verbose : bool
        If True, print forecast progress.

    Returns
    -------
    forecast_df : pandas DataFrame
        Recursive predictions for all targets.

    recursive_history_df : pandas DataFrame
        Original history plus recursively predicted future rows.
    """

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------
    if not isinstance(models, dict) or not models:
        raise ValueError(
            "models must be a non-empty dictionary."
        )

    if not isinstance(history_df, pd.DataFrame):
        raise ValueError(
            "history_df must be a pandas DataFrame."
        )

    if not isinstance(future_exog_df, pd.DataFrame):
        raise ValueError(
            "future_exog_df must be a pandas DataFrame."
        )

    if target_cols is None:
        target_cols = list(models.keys())
    else:
        target_cols = list(target_cols)

    missing_models = [
        phase
        for phase in target_cols
        if phase not in models
    ]

    if missing_models:
        raise KeyError(
            f"No fitted model was provided for: "
            f"{missing_models}"
        )

    missing_history_targets = [
        phase
        for phase in target_cols
        if phase not in history_df.columns
    ]

    if missing_history_targets:
        raise KeyError(
            f"history_df is missing these target columns: "
            f"{missing_history_targets}"
        )

    if not history_df.index.is_unique:
        raise ValueError(
            "history_df must have a unique index."
        )

    if not future_exog_df.index.is_unique:
        raise ValueError(
            "future_exog_df must have a unique index."
        )

    history_df = history_df.copy().sort_index()
    future_exog_df = future_exog_df.copy().sort_index()

    if horizon is None:
        horizon = len(future_exog_df)

    if horizon <= 0:
        raise ValueError(
            "horizon must be greater than zero."
        )

    if horizon > len(future_exog_df):
        raise ValueError(
            f"horizon={horizon}, but future_exog_df contains "
            f"only {len(future_exog_df)} rows."
        )

    future_exog_df = future_exog_df.iloc[:horizon].copy()

    overlapping_dates = history_df.index.intersection(
        future_exog_df.index
    )

    if len(overlapping_dates) > 0:
        raise ValueError(
            "history_df and future_exog_df contain overlapping "
            f"dates. Example overlap: {overlapping_dates[0]}"
        )

    # --------------------------------------------------
    # 2. Recursive forecast containers
    # --------------------------------------------------
    recursive_history_df = history_df.copy()
    forecast_records = []

    # All columns needed in recursive history
    all_raw_columns = list(
        dict.fromkeys(
            history_df.columns.tolist()
            + future_exog_df.columns.tolist()
            + target_cols
        )
    )

    # --------------------------------------------------
    # 3. Forecast one date at a time
    # --------------------------------------------------
    for step_number, forecast_date in enumerate(
        future_exog_df.index,
        start=1,
    ):
        exog_row = future_exog_df.loc[
            [forecast_date]
        ].copy()

        # Create a future row with all expected raw columns
        future_row = pd.DataFrame(
            np.nan,
            index=[forecast_date],
            columns=all_raw_columns,
        )

        # Insert known future exogenous variables
        for col in exog_row.columns:
            future_row.loc[forecast_date, col] = (
                exog_row.loc[forecast_date, col]
            )

        # Targets must remain unknown while creating features
        for phase in target_cols:
            future_row.loc[forecast_date, phase] = np.nan

        # Add the future placeholder row to history
        history_with_placeholder = pd.concat(
            [
                recursive_history_df,
                future_row,
            ],
            axis=0,
        ).sort_index()

        step_predictions = {}

        # --------------------------------------------------
        # Predict each phase independently
        # --------------------------------------------------
        for phase in target_cols:
            phase_model = models[phase]

            phase_feature_pipe = _get_phase_item(
                item=feature_pipe,
                phase=phase,
                item_name="feature_pipe",
            )

            transformed_history = (
                phase_feature_pipe.transform(
                    history_with_placeholder.copy()
                )
            )

            if not isinstance(
                transformed_history,
                pd.DataFrame,
            ):
                raise TypeError(
                    f"The feature pipeline for {phase} returned "
                    f"{type(transformed_history).__name__}. "
                    "It must return a pandas DataFrame so columns "
                    "can be selected by name."
                )

            if forecast_date not in transformed_history.index:
                raise ValueError(
                    f"The forecast row {forecast_date} was removed "
                    f"by the feature pipeline for {phase}. "
                    "Check DropMissingData and ensure current target "
                    "columns are removed before missing rows are dropped."
                )

            phase_feature_cols = (
                _get_model_feature_columns(
                    model=phase_model,
                    transformed_data=transformed_history,
                    model_feature_cols=model_feature_cols,
                    phase=phase,
                )
            )

            X_current = transformed_history.loc[
                [forecast_date],
                phase_feature_cols,
            ]

            prediction = np.asarray(
                phase_model.predict(X_current)
            ).reshape(-1)

            if len(prediction) != 1:
                raise ValueError(
                    f"The {phase} model returned "
                    f"{len(prediction)} predictions for one row."
                )

            predicted_value = float(prediction[0])

            if clip_negative:
                predicted_value = max(
                    predicted_value,
                    0.0,
                )

            step_predictions[phase] = predicted_value

        # --------------------------------------------------
        # Insert all phase predictions simultaneously
        # --------------------------------------------------
        for phase, predicted_value in (
            step_predictions.items()
        ):
            future_row.loc[
                forecast_date,
                phase,
            ] = predicted_value

        recursive_history_df = pd.concat(
            [
                recursive_history_df,
                future_row,
            ],
            axis=0,
        )

        recursive_history_df = (
            recursive_history_df[
                ~recursive_history_df.index.duplicated(
                    keep="last"
                )
            ]
            .sort_index()
        )

        forecast_record = {
            "forecast_date": forecast_date,
            **step_predictions,
        }

        forecast_records.append(forecast_record)

        if verbose:
            prediction_text = ", ".join(
                [
                    f"{phase}={value:,.2f}"
                    for phase, value
                    in step_predictions.items()
                ]
            )

            print(
                f"Step {step_number}/{horizon} | "
                f"{forecast_date} | "
                f"{prediction_text}"
            )

    # --------------------------------------------------
    # 4. Create forecast DataFrame
    # --------------------------------------------------
    forecast_df = pd.DataFrame(
        forecast_records
    ).set_index("forecast_date")

    forecast_df.index.name = future_exog_df.index.name

    forecast_df = forecast_df[
        target_cols
    ]

    return (
        forecast_df,
        recursive_history_df,
    )


#---------------------------------------------------------------------------------------------------------#

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)

try:
    from sklearn.metrics import root_mean_squared_error

except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(
            mean_squared_error(y_true, y_pred)
        )


def walk_forward_recursive_forecast_multiphase(
    models,
    feature_pipe,
    train_data,
    val_data,
    target_cols=None,
    model_feature_cols=None,
    forecast_horizon=30,
    exog_cols=None,
    clip_negative=True,
    verbose=True,
):
    """
    Walk-forward recursive validation using one independently fitted
    model for each production phase.

    For every validation block:

    1. Forecast the entire block recursively.
    2. Oil, gas, and water are predicted by separate models.
    3. Predicted values are used only within the current recursive block.
    4. At the end of the block, predicted history is discarded.
    5. Actual validation values are appended to history.
    6. Continue to the next validation block.

    Parameters
    ----------
    models : dict
        One fitted model per target.

        Example:

        {
            "BORE_OIL_VOL_STB_D": oil_model,
            "BORE_GAS_VOL_MSCF_D": gas_model,
            "BORE_WAT_VOL_STB_D": water_model,
        }

    feature_pipe : fitted pipeline or dict
        Either one shared feature pipeline or one pipeline per target.

    train_data : pandas DataFrame
        Historical training data with raw target and exogenous columns.

    val_data : pandas DataFrame
        Validation data with actual target and exogenous columns.

    target_cols : list or None
        Target columns. If None, model dictionary keys are used.

    model_feature_cols : list, dict, or None
        Feature columns used by the trained models.

    forecast_horizon : int
        Number of recursive steps per validation block.

    exog_cols : list or None
        Future-known columns. If None, all validation columns except
        target_cols are used.

    clip_negative : bool
        If True, negative predictions are changed to zero.

    verbose : bool
        If True, print progress.

    Returns
    -------
    all_forecasts_df : pandas DataFrame
        Predictions covering the full validation period.

    all_recursive_history_df : pandas DataFrame
        Recursively predicted rows generated within each block.
        Training-history duplicates are not included.

    walk_forward_metrics_df : pandas DataFrame
        Full-validation RMSE, MAE, and R2 for each phase.

    block_summary_df : pandas DataFrame
        Information about each validation block.
    """

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------
    if not isinstance(models, dict) or not models:
        raise ValueError(
            "models must be a non-empty dictionary."
        )

    if target_cols is None:
        target_cols = list(models.keys())
    else:
        target_cols = list(target_cols)

    if forecast_horizon <= 0:
        raise ValueError(
            "forecast_horizon must be greater than zero."
        )

    train_data = train_data.copy().sort_index()
    val_data = val_data.copy().sort_index()

    if len(val_data) == 0:
        raise ValueError(
            "val_data is empty."
        )

    missing_train_targets = [
        col
        for col in target_cols
        if col not in train_data.columns
    ]

    missing_val_targets = [
        col
        for col in target_cols
        if col not in val_data.columns
    ]

    if missing_train_targets:
        raise KeyError(
            f"train_data is missing targets: "
            f"{missing_train_targets}"
        )

    if missing_val_targets:
        raise KeyError(
            f"val_data is missing targets: "
            f"{missing_val_targets}"
        )

    if exog_cols is not None:
        missing_exog = [
            col
            for col in exog_cols
            if col not in val_data.columns
        ]

        if missing_exog:
            raise KeyError(
                f"val_data is missing exogenous columns: "
                f"{missing_exog}"
            )

    # Only actual observations are retained between blocks
    history_df = train_data.copy()

    all_forecasts = []
    all_recursive_rows = []
    block_summaries = []

    n_validation_rows = len(val_data)

    # --------------------------------------------------
    # 2. Walk through validation blocks
    # --------------------------------------------------
    for block_id, start in enumerate(
        range(
            0,
            n_validation_rows,
            forecast_horizon,
        ),
        start=1,
    ):
        stop = min(
            start + forecast_horizon,
            n_validation_rows,
        )

        actual_block_df = val_data.iloc[
            start:stop
        ].copy()

        block_horizon = len(actual_block_df)

        if exog_cols is None:
            future_exog_df = actual_block_df.drop(
                columns=target_cols,
                errors="ignore",
            ).copy()

        else:
            future_exog_df = actual_block_df[
                exog_cols
            ].copy()

        if verbose:
            print(
                "\n"
                + "=" * 80
            )

            print(
                f"Block {block_id}: "
                f"forecasting validation rows "
                f"{start} through {stop - 1}"
            )

            print(
                f"Forecast steps: {block_horizon}"
            )

            print(
                f"Forecast start: "
                f"{actual_block_df.index.min()}"
            )

            print(
                f"Forecast end:   "
                f"{actual_block_df.index.max()}"
            )

        # --------------------------------------------------
        # Recursive forecast for this block
        # --------------------------------------------------
        (
            forecast_block_df,
            recursive_history_block_df,
        ) = recursive_forecast_multiphase(
            models=models,
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

        forecast_block_df[
            "walk_forward_block"
        ] = block_id

        all_forecasts.append(
            forecast_block_df
        )

        # Keep only recursively generated rows for this block.
        # This prevents repeated copies of the full training history.
        recursive_block_rows = (
            recursive_history_block_df
            .reindex(actual_block_df.index)
            .copy()
        )

        recursive_block_rows[
            "walk_forward_block"
        ] = block_id

        all_recursive_rows.append(
            recursive_block_rows
        )

        block_summaries.append(
            {
                "block": block_id,
                "start_row": start,
                "end_row": stop - 1,
                "n_forecast_steps": block_horizon,
                "start_date":
                    actual_block_df.index.min(),
                "end_date":
                    actual_block_df.index.max(),
            }
        )

        # --------------------------------------------------
        # Important:
        # Update permanent history with ACTUAL data,
        # not predicted data.
        # --------------------------------------------------
        history_df = pd.concat(
            [
                history_df,
                actual_block_df,
            ],
            axis=0,
        )

        history_df = (
            history_df[
                ~history_df.index.duplicated(
                    keep="last"
                )
            ]
            .sort_index()
        )

    # --------------------------------------------------
    # 3. Combine blocks
    # --------------------------------------------------
    all_forecasts_df = (
        pd.concat(
            all_forecasts,
            axis=0,
        )
        .sort_index()
    )

    all_recursive_history_df = (
        pd.concat(
            all_recursive_rows,
            axis=0,
        )
        .sort_index()
    )

    block_summary_df = pd.DataFrame(
        block_summaries
    )

    # --------------------------------------------------
    # 4. Full validation metrics
    # --------------------------------------------------
    common_index = val_data.index.intersection(
        all_forecasts_df.index
    )

    metrics = []

    for phase in target_cols:
        y_true = val_data.loc[
            common_index,
            phase,
        ].astype(float)

        y_pred = all_forecasts_df.loc[
            common_index,
            phase,
        ].astype(float)

        valid_mask = (
            y_true.notna()
            & y_pred.notna()
        )

        y_true_valid = y_true.loc[
            valid_mask
        ]

        y_pred_valid = y_pred.loc[
            valid_mask
        ]

        if len(y_true_valid) == 0:
            phase_rmse = np.nan
            phase_mae = np.nan
            phase_r2 = np.nan

        else:
            phase_rmse = root_mean_squared_error(
                y_true_valid,
                y_pred_valid,
            )

            phase_mae = mean_absolute_error(
                y_true_valid,
                y_pred_valid,
            )

            if len(y_true_valid) >= 2:
                phase_r2 = r2_score(
                    y_true_valid,
                    y_pred_valid,
                )
            else:
                phase_r2 = np.nan

        metrics.append(
            {
                "Phase": phase,
                "WalkForward_Val_RMSE":
                    phase_rmse,
                "WalkForward_Val_MAE":
                    phase_mae,
                "WalkForward_Val_R2":
                    phase_r2,
                "N_Validation_Points":
                    len(y_true_valid),
            }
        )

    walk_forward_metrics_df = pd.DataFrame(
        metrics
    )

    return (
        all_forecasts_df,
        all_recursive_history_df,
        walk_forward_metrics_df,
        block_summary_df,
    )

#---------------------------------------------------------------------------------------------------------#
try:
    from sklearn.metrics import root_mean_squared_error

except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


def _get_phase_training_features(
    X,
    phase,
    model,
    target_cols,
    drop_target_cols=True,
):
    """
    Return the appropriate training feature DataFrame for one phase.

    X can be either:

    1. One shared DataFrame for all models.

    2. A dictionary containing one DataFrame per phase:

       {
           "BORE_OIL_VOL_STB_D": X_train_oil,
           "BORE_GAS_VOL_MSCF_D": X_train_gas,
           "BORE_WAT_VOL_STB_D": X_train_water,
       }
    """

    # --------------------------------------------------
    # Separate feature DataFrame for each phase
    # --------------------------------------------------
    if isinstance(X, dict):
        if phase not in X:
            raise KeyError(
                f"No training feature DataFrame was supplied "
                f"for phase '{phase}'."
            )

        X_phase = X[phase].copy()

    # --------------------------------------------------
    # One shared feature DataFrame
    # --------------------------------------------------
    else:
        X_phase = X.copy()

    if not isinstance(X_phase, pd.DataFrame):
        raise TypeError(
            f"Training features for {phase} must be a "
            "pandas DataFrame."
        )

    # Remove target columns if they are still in X
    if drop_target_cols:
        X_phase = X_phase.drop(
            columns=target_cols,
            errors="ignore",
        )

    # --------------------------------------------------
    # Recover the columns used when this model was fitted
    # --------------------------------------------------
    fitted_model = getattr(
        model,
        "best_estimator_",
        model,
    )

    expected_features = getattr(
        fitted_model,
        "feature_names_in_",
        None,
    )

    if expected_features is not None:
        expected_features = list(expected_features)

        missing_features = [
            feature
            for feature in expected_features
            if feature not in X_phase.columns
        ]

        if missing_features:
            raise ValueError(
                f"The training data for {phase} is missing "
                f"{len(missing_features)} features required by "
                f"the fitted model:\n{missing_features}"
            )

        # Select the exact features and order used during fitting
        X_phase = X_phase.loc[
            :,
            expected_features,
        ]

    return X_phase


def plot_multi_phase_train_walkforward_val_diagnostics(
    models,
    X_train,
    y_train,
    y_val,
    walk_forward_forecast_df,
    target_cols=None,
    model_name="Independent Phase Models",
    start_of_val_date=None,
    clip_negative=True,
    make_combined_plots=True,
    walk_forward_metrics_df=None,
    drop_target_cols_from_X=True,
):
    """
    Create train-fit and walk-forward-validation diagnostics for
    independently fitted oil, gas, and water models.

    Train predictions
    -----------------
    Each phase is predicted by its corresponding model:

        models[phase].predict(X_train_phase)

    Validation predictions
    ----------------------
    Walk-forward validation predictions are taken from the already
    generated walk_forward_forecast_df.

    Parameters
    ----------
    models : dict
        One fitted model per target.

        Example:

        {
            "BORE_OIL_VOL_STB_D": oil_model,
            "BORE_GAS_VOL_MSCF_D": gas_model,
            "BORE_WAT_VOL_STB_D": water_model,
        }

    X_train : pandas DataFrame or dict
        Training features.

        A single shared DataFrame can be used for all models, or a
        dictionary containing a different feature DataFrame per phase.

    y_train : pandas DataFrame
        Training target values.

    y_val : pandas DataFrame
        Validation target values.

    walk_forward_forecast_df : pandas DataFrame
        Walk-forward predictions produced by
        walk_forward_recursive_forecast_multiphase.

    target_cols : list or None
        Target columns and plot order. If None, models.keys() is used.

    model_name : str
        Model name displayed in the plots and metrics table.

    start_of_val_date : datetime-like or None
        Validation boundary shown in the time plots.

    clip_negative : bool
        If True, negative train and validation predictions are
        changed to zero before calculating metrics and plotting.

    make_combined_plots : bool
        If True, create the time-series plots and crossplots.

    walk_forward_metrics_df : pandas DataFrame or None
        Metrics previously returned by
        walk_forward_recursive_forecast_multiphase.

        Expected columns:

        - Phase
        - WalkForward_Val_RMSE
        - WalkForward_Val_MAE
        - WalkForward_Val_R2

        If None, validation metrics are recalculated here.

    drop_target_cols_from_X : bool
        If True, oil, gas, and water target columns are removed from
        X_train before calling the models.

    Returns
    -------
    train_pred_df : pandas DataFrame
        Training predictions from the three independent models.

    val_pred_df : pandas DataFrame
        Walk-forward validation predictions.

    metrics_df : pandas DataFrame
        Training and walk-forward validation metrics for each phase.
    """

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------
    if not isinstance(models, dict) or len(models) == 0:
        raise ValueError(
            "models must be a non-empty dictionary."
        )

    if not isinstance(y_train, pd.DataFrame):
        raise TypeError(
            "y_train must be a pandas DataFrame."
        )

    if not isinstance(y_val, pd.DataFrame):
        raise TypeError(
            "y_val must be a pandas DataFrame."
        )

    if not isinstance(
        walk_forward_forecast_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "walk_forward_forecast_df must be a pandas DataFrame."
        )

    if target_cols is None:
        target_cols = list(models.keys())
    else:
        target_cols = list(target_cols)

    missing_models = [
        phase
        for phase in target_cols
        if phase not in models
    ]

    if missing_models:
        raise KeyError(
            f"No fitted models were supplied for: "
            f"{missing_models}"
        )

    missing_train_targets = [
        phase
        for phase in target_cols
        if phase not in y_train.columns
    ]

    if missing_train_targets:
        raise KeyError(
            f"Targets missing from y_train: "
            f"{missing_train_targets}"
        )

    missing_val_targets = [
        phase
        for phase in target_cols
        if phase not in y_val.columns
    ]

    if missing_val_targets:
        raise KeyError(
            f"Targets missing from y_val: "
            f"{missing_val_targets}"
        )

    missing_forecast_targets = [
        phase
        for phase in target_cols
        if phase not in walk_forward_forecast_df.columns
    ]

    if missing_forecast_targets:
        raise KeyError(
            "Targets missing from "
            "walk_forward_forecast_df: "
            f"{missing_forecast_targets}"
        )

    # --------------------------------------------------
    # 2. Align walk-forward validation predictions
    # --------------------------------------------------
    val_pred_df = walk_forward_forecast_df[
        target_cols
    ].copy()

    common_val_index = y_val.index.intersection(
        val_pred_df.index,
        sort=False,
    )

    if len(common_val_index) == 0:
        raise ValueError(
            "There are no common dates between y_val and "
            "walk_forward_forecast_df."
        )

    y_val_aligned = y_val.loc[
        common_val_index,
        target_cols,
    ].copy()

    val_pred_df = val_pred_df.loc[
        common_val_index,
        target_cols,
    ].copy()

    if clip_negative:
        val_pred_df = val_pred_df.clip(lower=0)

    # --------------------------------------------------
    # 3. Containers for separate train predictions
    # --------------------------------------------------
    train_prediction_series = {}
    train_actual_series = {}
    metrics = []

    # --------------------------------------------------
    # 4. Predict training data separately for each phase
    # --------------------------------------------------
    for phase in target_cols:
        model = models[phase]

        X_train_phase = _get_phase_training_features(
            X=X_train,
            phase=phase,
            model=model,
            target_cols=target_cols,
            drop_target_cols=drop_target_cols_from_X,
        )

        missing_train_indexes = (
            X_train_phase.index.difference(
                y_train.index
            )
        )

        if len(missing_train_indexes) > 0:
            raise ValueError(
                f"Some training indexes for {phase} are "
                "missing from y_train."
            )

        y_train_phase = (
            y_train
            .loc[X_train_phase.index, phase]
            .astype(float)
            .copy()
        )

        train_prediction = np.asarray(
            model.predict(X_train_phase)
        ).reshape(-1)

        if len(train_prediction) != len(X_train_phase):
            raise ValueError(
                f"The model for {phase} produced "
                f"{len(train_prediction)} predictions for "
                f"{len(X_train_phase)} rows."
            )

        train_pred_phase = pd.Series(
            train_prediction,
            index=X_train_phase.index,
            name=phase,
            dtype=float,
        )

        if clip_negative:
            train_pred_phase = train_pred_phase.clip(
                lower=0
            )

        train_prediction_series[phase] = (
            train_pred_phase
        )

        train_actual_series[phase] = (
            y_train_phase
        )

        # --------------------------------------------------
        # Remove missing points before calculating metrics
        # --------------------------------------------------
        train_valid_mask = (
            y_train_phase.notna()
            & train_pred_phase.notna()
        )

        y_train_valid = y_train_phase.loc[
            train_valid_mask
        ]

        train_pred_valid = train_pred_phase.loc[
            train_valid_mask
        ]

        y_val_phase = (
            y_val_aligned[phase]
            .astype(float)
            .copy()
        )

        val_pred_phase = (
            val_pred_df[phase]
            .astype(float)
            .copy()
        )

        val_valid_mask = (
            y_val_phase.notna()
            & val_pred_phase.notna()
        )

        y_val_valid = y_val_phase.loc[
            val_valid_mask
        ]

        val_pred_valid = val_pred_phase.loc[
            val_valid_mask
        ]

        if len(y_train_valid) == 0:
            raise ValueError(
                f"No valid training observations remain "
                f"for {phase}."
            )

        if len(y_val_valid) == 0:
            raise ValueError(
                f"No valid walk-forward validation "
                f"observations remain for {phase}."
            )

        # --------------------------------------------------
        # 5. Training metrics
        # --------------------------------------------------
        train_rmse = root_mean_squared_error(
            y_train_valid,
            train_pred_valid,
        )

        train_mae = mean_absolute_error(
            y_train_valid,
            train_pred_valid,
        )

        train_r2 = (
            r2_score(
                y_train_valid,
                train_pred_valid,
            )
            if len(y_train_valid) >= 2
            else np.nan
        )

        # --------------------------------------------------
        # 6. Walk-forward validation metrics
        # --------------------------------------------------
        if walk_forward_metrics_df is not None:
            metric_row = walk_forward_metrics_df[
                walk_forward_metrics_df["Phase"]
                == phase
            ]

            if len(metric_row) == 0:
                raise ValueError(
                    f"Phase '{phase}' was not found in "
                    "walk_forward_metrics_df."
                )

            metric_row = metric_row.iloc[0]

            required_metric_columns = [
                "WalkForward_Val_RMSE",
                "WalkForward_Val_MAE",
                "WalkForward_Val_R2",
            ]

            missing_metric_columns = [
                column
                for column in required_metric_columns
                if column not in metric_row.index
            ]

            if missing_metric_columns:
                raise KeyError(
                    "walk_forward_metrics_df is missing "
                    f"columns: {missing_metric_columns}"
                )

            val_rmse = metric_row[
                "WalkForward_Val_RMSE"
            ]

            val_mae = metric_row[
                "WalkForward_Val_MAE"
            ]

            val_r2 = metric_row[
                "WalkForward_Val_R2"
            ]

        else:
            val_rmse = root_mean_squared_error(
                y_val_valid,
                val_pred_valid,
            )

            val_mae = mean_absolute_error(
                y_val_valid,
                val_pred_valid,
            )

            val_r2 = (
                r2_score(
                    y_val_valid,
                    val_pred_valid,
                )
                if len(y_val_valid) >= 2
                else np.nan
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
                "Val_RMSE": val_rmse,
                "Val_MAE": val_mae,
                "Val_R2": val_r2,
                "N_Train_Points": len(
                    y_train_valid
                ),
                "N_WalkForward_Val_Points": len(
                    y_val_valid
                ),
            }
        )

    # --------------------------------------------------
    # 7. Combine train predictions into one DataFrame
    # --------------------------------------------------
    train_pred_df = pd.concat(
        [
            train_prediction_series[phase].rename(
                phase
            )
            for phase in target_cols
        ],
        axis=1,
    )

    train_pred_df = train_pred_df.reindex(
        columns=target_cols
    )

    metrics_df = pd.DataFrame(metrics)

    # --------------------------------------------------
    # 8. Plot train and walk-forward time series
    # --------------------------------------------------
    if make_combined_plots:
        fig, axes = plt.subplots(
            nrows=len(target_cols),
            ncols=1,
            figsize=(
                15,
                5 * len(target_cols),
            ),
            sharex=False,
        )

        if len(target_cols) == 1:
            axes = [axes]

        for ax, phase in zip(
            axes,
            target_cols,
        ):
            metric_row = (
                metrics_df[
                    metrics_df["Phase"] == phase
                ]
                .iloc[0]
            )

            plot_train_val_predictions_time(
                y_train=train_actual_series[phase],
                train_pred=(
                    train_prediction_series[phase]
                ),
                y_val=y_val_aligned[phase],
                val_pred=val_pred_df[phase],
                phase_string=phase,
                train_rmse=metric_row[
                    "Train_RMSE"
                ],
                train_mae=metric_row[
                    "Train_MAE"
                ],
                val_rmse=metric_row[
                    "WalkForward_Val_RMSE"
                ],
                val_mae=metric_row[
                    "WalkForward_Val_MAE"
                ],
                ax=ax,
                start_of_val_date=(
                    start_of_val_date
                ),
                clip_negative=False,
            )

        fig.suptitle(
            f"{model_name}: Train Fit and "
            "Walk-Forward Validation Actual vs Predicted",
            fontsize=16,
            y=1.02,
        )

        plt.tight_layout()
        plt.show()

        # --------------------------------------------------
        # 9. Actual-vs-predicted crossplots
        # --------------------------------------------------
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(target_cols),
            figsize=(
                7 * len(target_cols),
                7,
            ),
            sharex=False,
            sharey=False,
        )

        if len(target_cols) == 1:
            axes = [axes]

        crossplot_dfs = {}

        for ax, phase in zip(
            axes,
            target_cols,
        ):
            metric_row = (
                metrics_df[
                    metrics_df["Phase"] == phase
                ]
                .iloc[0]
            )

            crossplot_df, _ = (
                plot_train_val_actual_vs_predicted_crossplot(
                    y_train=(
                        train_actual_series[phase]
                    ),
                    train_pred=(
                        train_prediction_series[phase]
                    ),
                    y_val=y_val_aligned[phase],
                    val_pred=val_pred_df[phase],
                    phase_string=phase,
                    train_rmse=metric_row[
                        "Train_RMSE"
                    ],
                    train_mae=metric_row[
                        "Train_MAE"
                    ],
                    val_rmse=metric_row[
                        "WalkForward_Val_RMSE"
                    ],
                    val_mae=metric_row[
                        "WalkForward_Val_MAE"
                    ],
                    ax=ax,
                    clip_negative=False,
                )
            )

            crossplot_dfs[phase] = (
                crossplot_df
            )

        fig.suptitle(
            f"{model_name}: Actual vs Predicted "
            "Crossplots",
            fontsize=16,
            y=1.02,
        )

        plt.tight_layout()
        plt.show()

    return (
        train_pred_df,
        val_pred_df,
        metrics_df,
    )

#---------------------------------------------------------------------------------------------------------#













































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