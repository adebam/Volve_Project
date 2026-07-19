#basics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
#sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

# optuna
import optuna
#--------------------------------------------------------------------------------------------------------------------#
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

#-------------------------------------------------------------------------------------------------#
def create_model_results_table():
    columns = pd.MultiIndex.from_tuples([
        ("Model", "Name"),

        ("Oil", "MAE"),
        ("Oil", "RMSE"),
        ("Oil", "R2"),
        ("Oil", "Features Used"),

        ("Gas", "MAE"),
        ("Gas", "RMSE"),
        ("Gas", "R2"),
        ("Gas", "Features Used"),

        ("Water", "MAE"),
        ("Water", "RMSE"),
        ("Water", "R2"),
        ("Water", "Features Used"),
    ])

    return pd.DataFrame(columns=columns)
#--------------------------------------------------------------------------------------------------------------#
def add_model_result(
    results_df,
    model_name,

    oil_mae=None,
    oil_rmse=None,
    oil_r2=None,
    oil_features=None,

    gas_mae=None,
    gas_rmse=None,
    gas_r2=None,
    gas_features=None,

    water_mae=None,
    water_rmse=None,
    water_r2=None,
    water_features=None,
):
    """
    Adds one row to the model comparison table.

    Each row represents one model.
    Columns are grouped by Oil, Gas, and Water.
    """

    def format_features(features):
        if features is None:
            return None

        if isinstance(features, list):
            return ", ".join(features)

        if isinstance(features, pd.Series):
            return ", ".join(features.astype(str).tolist())

        if isinstance(features, pd.Index):
            return ", ".join(features.astype(str).tolist())

        return str(features)

    row = {
        ("Model", "Name"): model_name,

        ("Oil", "MAE"): oil_mae,
        ("Oil", "RMSE"): oil_rmse,
        ("Oil", "R2"): oil_r2,
        ("Oil", "Features Used"): format_features(oil_features),

        ("Gas", "MAE"): gas_mae,
        ("Gas", "RMSE"): gas_rmse,
        ("Gas", "R2"): gas_r2,
        ("Gas", "Features Used"): format_features(gas_features),

        ("Water", "MAE"): water_mae,
        ("Water", "RMSE"): water_rmse,
        ("Water", "R2"): water_r2,
        ("Water", "Features Used"): format_features(water_features),
    }

    new_row_df = pd.DataFrame([row])

    results_df = pd.concat(
        [results_df, new_row_df],
        axis=0,
        ignore_index=True
    )

    return results_df
#----------------------------------------------------------------------------------------------------------------------------------------#
def style_model_results_table(model_results_df):
    metric_cols = [
        ("Oil", "MAE"),
        ("Oil", "RMSE"),
        ("Oil", "R2"),

        ("Gas", "MAE"),
        ("Gas", "RMSE"),
        ("Gas", "R2"),

        ("Water", "MAE"),
        ("Water", "RMSE"),
        ("Water", "R2"),
    ]

    feature_cols = [
        ("Oil", "Features Used"),
        ("Gas", "Features Used"),
        ("Water", "Features Used"),
    ]

    styled_table = (
        model_results_df.style

        # Round metric values
        .format("{:.3f}", subset=metric_cols)

        # Center model name and metrics
        .set_properties(
            subset=[("Model", "Name")] + metric_cols,
            **{
                "text-align": "center",
                "vertical-align": "middle",
            }
        )

        # Wrap long feature lists
        .set_properties(
            subset=feature_cols,
            **{
                "white-space": "pre-wrap",
                "text-align": "left",
                "vertical-align": "top",
                "min-width": "280px",
                "max-width": "450px",
            }
        )

        # Center the column headers, including Oil, Gas, Water
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("vertical-align", "middle"),
                ],
            },
            {
                "selector": "th.col_heading.level0",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "bold"),
                    ("font-size", "12pt"),
                ],
            },
            {
                "selector": "th.col_heading.level1",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "bold"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid lightgray"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("border", "1px solid lightgray"),
                ],
            },
        ])
    )

    return styled_table

#-------------------------------------------------------------------------------------------------------------------
def plot_single_target_multiphase_train_val_diagnostics(
    wellname,
    model_name="Model",

    # -------------------------
    # Actual targets
    # -------------------------
    y_train_oil=None,
    y_val_oil=None,
    y_train_gas=None,
    y_val_gas=None,
    y_train_water=None,
    y_val_water=None,

    # -------------------------
    # Optional predictions
    # -------------------------
    train_pred_oil=None,
    val_pred_oil=None,
    train_pred_gas=None,
    val_pred_gas=None,
    train_pred_water=None,
    val_pred_water=None,

    # -------------------------
    # Optional models + X data
    # If predictions are not passed, these can be used
    # -------------------------
    oil_model=None,
    gas_model=None,
    water_model=None,
    X_train_oil=None,
    X_val_oil=None,
    X_train_gas=None,
    X_val_gas=None,
    X_train_water=None,
    X_val_water=None,

    # -------------------------
    # Options
    # -------------------------
    clip_negative=False,
    save_dir=None,
    show=True,
):
    """
    Creates:
    1. 3-row time plot (oil, gas, water)
    2. 1x3 crossplot (oil, gas, water)

    Works for separate single-target models/predictions.

    You can either:
    - pass train/val predictions directly, OR
    - pass fitted models and X_train/X_val so predictions are computed inside.

    Returns
    -------
    results : dict
        Dictionary containing:
        - metrics_df
        - time_fig
        - crossplot_fig
        - phase_crossplot_dfs
        - predictions
    """

    # ----------------------------------------------------
    # Helper to compute predictions if not supplied
    # ----------------------------------------------------
    def _get_predictions(train_pred, val_pred, model, X_train, X_val, phase_name):
        if train_pred is None or val_pred is None:
            if model is None or X_train is None or X_val is None:
                raise ValueError(
                    f"For phase '{phase_name}', either provide "
                    f"train_pred/val_pred OR provide model + X_train + X_val."
                )
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

        train_pred = np.asarray(train_pred).ravel()
        val_pred = np.asarray(val_pred).ravel()

        if clip_negative:
            train_pred = np.clip(train_pred, 0, None)
            val_pred = np.clip(val_pred, 0, None)

        return train_pred, val_pred

    # ----------------------------------------------------
    # Prepare all 3 phases
    # ----------------------------------------------------
    train_pred_oil, val_pred_oil = _get_predictions(
        train_pred_oil, val_pred_oil,
        oil_model, X_train_oil, X_val_oil,
        phase_name="oil"
    )

    train_pred_gas, val_pred_gas = _get_predictions(
        train_pred_gas, val_pred_gas,
        gas_model, X_train_gas, X_val_gas,
        phase_name="gas"
    )

    train_pred_water, val_pred_water = _get_predictions(
        train_pred_water, val_pred_water,
        water_model, X_train_water, X_val_water,
        phase_name="water"
    )

    # Convert y to 1D arrays for metrics, but preserve original series for plotting
    y_train_oil_arr = np.asarray(y_train_oil).ravel()
    y_val_oil_arr = np.asarray(y_val_oil).ravel()

    y_train_gas_arr = np.asarray(y_train_gas).ravel()
    y_val_gas_arr = np.asarray(y_val_gas).ravel()

    y_train_water_arr = np.asarray(y_train_water).ravel()
    y_val_water_arr = np.asarray(y_val_water).ravel()

    # ----------------------------------------------------
    # Metrics
    # ----------------------------------------------------
    def _calc_metrics(y_train, train_pred, y_val, val_pred, phase_label):
        return {
            "Phase": phase_label,
            "Train_RMSE": root_mean_squared_error(y_train, train_pred),
            "Train_MAE": mean_absolute_error(y_train, train_pred),
            "Train_R2": r2_score(y_train, train_pred),
            "Val_RMSE": root_mean_squared_error(y_val, val_pred),
            "Val_MAE": mean_absolute_error(y_val, val_pred),
            "Val_R2": r2_score(y_val, val_pred),
        }

    metrics_list = [
        _calc_metrics(y_train_oil_arr, train_pred_oil, y_val_oil_arr, val_pred_oil, "BORE_OIL_VOL_STB_D"),
        _calc_metrics(y_train_gas_arr, train_pred_gas, y_val_gas_arr, val_pred_gas, "BORE_GAS_VOL_MSCF_D"),
        _calc_metrics(y_train_water_arr, train_pred_water, y_val_water_arr, val_pred_water, "BORE_WAT_VOL_STB_D"),
    ]

    metrics_df = pd.DataFrame(metrics_list)

    # ----------------------------------------------------
    # Time plot
    # ----------------------------------------------------
    time_fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(16, 12),
        sharex=True
    )

    plot_train_val_predictions_time(
        y_train=y_train_oil,
        train_pred=train_pred_oil,
        y_val=y_val_oil,
        val_pred=val_pred_oil,
        phase_string="BORE_OIL_VOL_STB_D",
        train_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_OIL_VOL_STB_D", "Train_RMSE"].values[0],
        train_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_OIL_VOL_STB_D", "Train_MAE"].values[0],
        val_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_OIL_VOL_STB_D", "Val_RMSE"].values[0],
        val_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_OIL_VOL_STB_D", "Val_MAE"].values[0],
        ax=axes[0],
        start_of_val_date=y_val_oil.index[0]
    )

    plot_train_val_predictions_time(
        y_train=y_train_gas,
        train_pred=train_pred_gas,
        y_val=y_val_gas,
        val_pred=val_pred_gas,
        phase_string="BORE_GAS_VOL_MSCF_D",
        train_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_GAS_VOL_MSCF_D", "Train_RMSE"].values[0],
        train_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_GAS_VOL_MSCF_D", "Train_MAE"].values[0],
        val_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_GAS_VOL_MSCF_D", "Val_RMSE"].values[0],
        val_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_GAS_VOL_MSCF_D", "Val_MAE"].values[0],
        ax=axes[1],
        start_of_val_date=y_val_gas.index[0]
    )

    plot_train_val_predictions_time(
        y_train=y_train_water,
        train_pred=train_pred_water,
        y_val=y_val_water,
        val_pred=val_pred_water,
        phase_string="BORE_WAT_VOL_STB_D",
        train_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_WAT_VOL_STB_D", "Train_RMSE"].values[0],
        train_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_WAT_VOL_STB_D", "Train_MAE"].values[0],
        val_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_WAT_VOL_STB_D", "Val_RMSE"].values[0],
        val_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_WAT_VOL_STB_D", "Val_MAE"].values[0],
        ax=axes[2],
        start_of_val_date=y_val_water.index[0]
    )

    axes[-1].set_xlabel("Date")

    time_fig.suptitle(
        f"{wellname} {model_name}: Actual vs Predicted Time Plot (Training & Validation)",
        fontsize=14,
        y=0.99
    )

    time_fig.tight_layout()

    # ----------------------------------------------------
    # Crossplot
    # ----------------------------------------------------
    crossplot_fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 6)
    )

    oil_crossplot_df, _ = plot_train_val_actual_vs_predicted_crossplot(
        y_train=y_train_oil,
        train_pred=train_pred_oil,
        y_val=y_val_oil,
        val_pred=val_pred_oil,
        phase_string="BORE_OIL_VOL_STB_D",
        train_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_OIL_VOL_STB_D", "Train_RMSE"].values[0],
        train_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_OIL_VOL_STB_D", "Train_MAE"].values[0],
        val_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_OIL_VOL_STB_D", "Val_RMSE"].values[0],
        val_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_OIL_VOL_STB_D", "Val_MAE"].values[0],
        ax=axes[0]
    )

    gas_crossplot_df, _ = plot_train_val_actual_vs_predicted_crossplot(
        y_train=y_train_gas,
        train_pred=train_pred_gas,
        y_val=y_val_gas,
        val_pred=val_pred_gas,
        phase_string="BORE_GAS_VOL_MSCF_D",
        train_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_GAS_VOL_MSCF_D", "Train_RMSE"].values[0],
        train_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_GAS_VOL_MSCF_D", "Train_MAE"].values[0],
        val_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_GAS_VOL_MSCF_D", "Val_RMSE"].values[0],
        val_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_GAS_VOL_MSCF_D", "Val_MAE"].values[0],
        ax=axes[1]
    )

    water_crossplot_df, _ = plot_train_val_actual_vs_predicted_crossplot(
        y_train=y_train_water,
        train_pred=train_pred_water,
        y_val=y_val_water,
        val_pred=val_pred_water,
        phase_string="BORE_WAT_VOL_STB_D",
        train_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_WAT_VOL_STB_D", "Train_RMSE"].values[0],
        train_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_WAT_VOL_STB_D", "Train_MAE"].values[0],
        val_rmse=metrics_df.loc[metrics_df["Phase"] == "BORE_WAT_VOL_STB_D", "Val_RMSE"].values[0],
        val_mae=metrics_df.loc[metrics_df["Phase"] == "BORE_WAT_VOL_STB_D", "Val_MAE"].values[0],
        ax=axes[2]
    )

    crossplot_fig.suptitle(
        f"{wellname} {model_name}: Actual vs Predicted Crossplot (Training & Validation)",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )

    crossplot_fig.tight_layout()

    # ----------------------------------------------------
    # Save figures if requested
    # ----------------------------------------------------
    saved_paths = {}

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        safe_well_name = wellname.replace("/", "-")

        time_path = os.path.join(
            save_dir,
            f"{safe_well_name}_{model_name.lower().replace(' ', '_')}_prediction_time_plot.png"
        )
        crossplot_path = os.path.join(
            save_dir,
            f"{safe_well_name}_{model_name.lower().replace(' ', '_')}_prediction_crossplot.png"
        )

        time_fig.savefig(time_path, dpi=300, bbox_inches="tight")
        crossplot_fig.savefig(crossplot_path, dpi=300, bbox_inches="tight")

        saved_paths["time_plot"] = time_path
        saved_paths["crossplot"] = crossplot_path

    if show:
        plt.show()
    else:
        plt.close(time_fig)
        plt.close(crossplot_fig)

    return {
        "metrics_df": metrics_df,
        "time_fig": time_fig,
        "crossplot_fig": crossplot_fig,
        "phase_crossplot_dfs": {
            "oil": oil_crossplot_df,
            "gas": gas_crossplot_df,
            "water": water_crossplot_df,
        },
        "predictions": {
            "train_pred_oil": train_pred_oil,
            "val_pred_oil": val_pred_oil,
            "train_pred_gas": train_pred_gas,
            "val_pred_gas": val_pred_gas,
            "train_pred_water": train_pred_water,
            "val_pred_water": val_pred_water,
        },
        "saved_paths": saved_paths,
    }
#----------------------------------------------------------------------------------------------------_#
def tune_single_target_random_forest_optuna(
    X_train,
    y_train,
    X_val,
    y_val,
    phase_name="Oil",
    n_trials=75,
    n_splits=5,
    random_state=42,
    clip_negative=True,
):
    y_train_1d = np.ravel(y_train)
    y_val_1d = np.ravel(y_val)

    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):

        rf_params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": random_state,
            "n_jobs": -1,
        }

        if rf_params["bootstrap"]:
            rf_params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)

        pipe = Pipeline(
            steps=[
                ("model", RandomForestRegressor(**rf_params))
            ]
        )

        cv_rmse_scores = []

        for train_idx, valid_idx in ts_cv.split(X_train):
            X_cv_train = X_train.iloc[train_idx]
            X_cv_valid = X_train.iloc[valid_idx]

            y_cv_train = y_train_1d[train_idx]
            y_cv_valid = y_train_1d[valid_idx]

            model = clone(pipe)
            model.fit(X_cv_train, y_cv_train)

            y_cv_pred = model.predict(X_cv_valid)

            if clip_negative:
                y_cv_pred = np.clip(y_cv_pred, 0, None)

            rmse = root_mean_squared_error(y_cv_valid, y_cv_pred)
            cv_rmse_scores.append(rmse)

        return np.mean(cv_rmse_scores)

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{phase_name.lower()}_random_forest_single_target"
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()

    best_model = Pipeline(
        steps=[
            (
                "model",
                RandomForestRegressor(
                    **best_params,
                    random_state=random_state,
                    n_jobs=-1,
                )
            )
        ]
    )

    best_model.fit(X_train, y_train_1d)

    val_pred = best_model.predict(X_val)

    if clip_negative:
        val_pred = np.clip(val_pred, 0, None)

    metrics = {
        "Model": "Single-target Random Forest",
        "Phase": phase_name,
        "Best_CV_RMSE": study.best_value,
        "Val_RMSE": root_mean_squared_error(y_val_1d, val_pred),
        "Val_MAE": mean_absolute_error(y_val_1d, val_pred),
        "Val_R2": r2_score(y_val_1d, val_pred),
        "Best_Params": study.best_params,
    }

    importance_df = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": best_model.named_steps["model"].feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    val_pred_series = pd.Series(
        val_pred,
        index=X_val.index,
        name=f"{phase_name}_RF_pred"
    )

    return best_model, study, pd.DataFrame([metrics]), importance_df, val_pred_series

#--------------------------------------------------------------------------------------------------------------------------_#
def tune_single_target_gradient_boosting_optuna(
    X_train,
    y_train,
    X_val,
    y_val,
    phase_name="Oil",
    n_trials=50,
    n_splits=5,
    random_state=42,
    clip_negative=True,
):
    """
    Tune a single-target GradientBoostingRegressor using Optuna + TimeSeriesSplit.

    Returns:
    - best_model
    - study
    - metrics_df
    - importance_df
    - val_pred_series
    """

    y_train_1d = np.ravel(y_train)
    y_val_1d = np.ravel(y_val)

    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):

        loss = trial.suggest_categorical(
            "loss",
            ["squared_error", "absolute_error", "huber"]
        )

        gb_params = {
            "loss": loss,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 25),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "max_features": trial.suggest_categorical(
                "max_features",
                [None, "sqrt", "log2", 0.3, 0.5, 0.7, 1.0]
            ),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.02),
            "random_state": random_state,
        }

        # alpha is only used for huber loss here
        if loss == "huber":
            gb_params["alpha"] = trial.suggest_float("alpha", 0.80, 0.99)

        gb_pipe = Pipeline(
            steps=[
                ("model", GradientBoostingRegressor(**gb_params))
            ]
        )

        cv_rmse_scores = []

        for train_idx, valid_idx in ts_cv.split(X_train):

            X_cv_train = X_train.iloc[train_idx]
            X_cv_valid = X_train.iloc[valid_idx]

            y_cv_train = y_train_1d[train_idx]
            y_cv_valid = y_train_1d[valid_idx]

            model = clone(gb_pipe)
            model.fit(X_cv_train, y_cv_train)

            y_cv_pred = model.predict(X_cv_valid)

            if clip_negative:
                y_cv_pred = np.clip(y_cv_pred, 0, None)

            rmse = root_mean_squared_error(y_cv_valid, y_cv_pred)
            cv_rmse_scores.append(rmse)

        return np.mean(cv_rmse_scores)

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{phase_name.lower()}_gradient_boosting_single_target"
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()

    best_model = Pipeline(
        steps=[
            (
                "model",
                GradientBoostingRegressor(
                    **best_params,
                    random_state=random_state,
                )
            )
        ]
    )

    best_model.fit(X_train, y_train_1d)

    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)

    if clip_negative:
        train_pred = np.clip(train_pred, 0, None)
        val_pred = np.clip(val_pred, 0, None)

    metrics = {
        "Model": "Single-target Gradient Boosting",
        "Phase": phase_name,
        "Best_CV_RMSE": study.best_value,

        "Train_RMSE": root_mean_squared_error(y_train_1d, train_pred),
        "Train_MAE": mean_absolute_error(y_train_1d, train_pred),
        "Train_R2": r2_score(y_train_1d, train_pred),

        "Val_RMSE": root_mean_squared_error(y_val_1d, val_pred),
        "Val_MAE": mean_absolute_error(y_val_1d, val_pred),
        "Val_R2": r2_score(y_val_1d, val_pred),

        "Best_Params": study.best_params,
    }

    metrics_df = pd.DataFrame([metrics])

    importance_df = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": best_model.named_steps["model"].feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    val_pred_series = pd.Series(
        val_pred,
        index=X_val.index,
        name=f"{phase_name}_GB_val_pred"
    )

    return best_model, study, metrics_df, importance_df, val_pred_series
#-----------------------------------------------------------------------------------------------#
def tune_single_target_lightgbm_optuna(
    X_train,
    y_train,
    X_val,
    y_val,
    phase_name="Oil",
    n_trials=50,
    n_splits=5,
    random_state=42,
    clip_negative=True,
    linear_tree=False,
):
    """
    Tune a single-target LightGBM model using Optuna + TimeSeriesSplit.

    linear_tree=False -> normal LightGBM
    linear_tree=True  -> LightGBM with linear models at the leaves
    """

    y_train_1d = np.ravel(y_train)
    y_val_1d = np.ravel(y_val)

    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):

        # regression_l1 is not allowed with linear_tree=True,
        # so keep the objective choices simple and compatible.
        objective_name = trial.suggest_categorical(
            "objective",
            ["regression", "huber"]
        )

        max_depth = trial.suggest_int("max_depth", 2, 10)
        max_num_leaves = min(2 ** max_depth, 128)

        lgbm_params = {
            "objective": objective_name,

            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),

            "max_depth": max_depth,
            "num_leaves": trial.suggest_int("num_leaves", 4, max_num_leaves),

            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),

            "linear_tree": linear_tree,

            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

        if objective_name == "huber":
            lgbm_params["alpha"] = trial.suggest_float("alpha", 0.80, 0.99)

        if linear_tree:
            lgbm_params["linear_lambda"] = trial.suggest_float(
                "linear_lambda",
                1e-8,
                100.0,
                log=True
            )

        if linear_tree:
            lgbm_pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LGBMRegressor(**lgbm_params)),
                ]
            )
        else:
            lgbm_pipe = Pipeline(
                steps=[
                    ("model", LGBMRegressor(**lgbm_params)),
                ]
            )

        cv_rmse_scores = []

        for train_idx, valid_idx in ts_cv.split(X_train):

            X_cv_train = X_train.iloc[train_idx]
            X_cv_valid = X_train.iloc[valid_idx]

            y_cv_train = y_train_1d[train_idx]
            y_cv_valid = y_train_1d[valid_idx]

            model = clone(lgbm_pipe)
            model.fit(X_cv_train, y_cv_train)

            y_cv_pred = model.predict(X_cv_valid)

            if clip_negative:
                y_cv_pred = np.clip(y_cv_pred, 0, None)

            rmse = root_mean_squared_error(y_cv_valid, y_cv_pred)
            cv_rmse_scores.append(rmse)

        return np.mean(cv_rmse_scores)

    study_name = (
        f"{phase_name.lower()}_lightgbm_linear_tree_single_target"
        if linear_tree
        else f"{phase_name.lower()}_lightgbm_normal_single_target"
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_params["linear_tree"] = linear_tree

    final_lgbm_params = best_params.copy()
    final_lgbm_params.update(
        {
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
    )

    if linear_tree:
        best_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LGBMRegressor(**final_lgbm_params)),
            ]
        )
    else:
        best_model = Pipeline(
            steps=[
                ("model", LGBMRegressor(**final_lgbm_params)),
            ]
        )

    best_model.fit(X_train, y_train_1d)

    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)

    if clip_negative:
        train_pred = np.clip(train_pred, 0, None)
        val_pred = np.clip(val_pred, 0, None)

    metrics = {
        "Model": "Single-target LightGBM",
        "Phase": phase_name,
        "Linear_Tree": linear_tree,
        "Best_CV_RMSE": study.best_value,

        "Train_RMSE": root_mean_squared_error(y_train_1d, train_pred),
        "Train_MAE": mean_absolute_error(y_train_1d, train_pred),
        "Train_R2": r2_score(y_train_1d, train_pred),

        "Val_RMSE": root_mean_squared_error(y_val_1d, val_pred),
        "Val_MAE": mean_absolute_error(y_val_1d, val_pred),
        "Val_R2": r2_score(y_val_1d, val_pred),

        "Best_Params": best_params,
    }

    metrics_df = pd.DataFrame([metrics])

    importance_df = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": best_model.named_steps["model"].feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    val_pred_series = pd.Series(
        val_pred,
        index=X_val.index,
        name=f"{phase_name}_LGBM_val_pred"
    )

    return best_model, study, metrics_df, importance_df, val_pred_series