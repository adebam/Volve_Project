#basics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# stats models
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.tsaplots import month_plot
from statsmodels.tsa.statespace.sarimax import SARIMAX

# sk-learn
from sklearn.metrics import mean_squared_error, mean_absolute_error

# others
import warnings
import itertools

# optuna
import optuna



#############################################################################
# Optuna  grid search & feature selection for EXO
#############################################################################

warnings.filterwarnings("ignore")


def sarimax_optuna_search_EXO(
    train_y,
    test_y,
    train_exog=None,
    test_exog=None,
    seasonal_period=7,
    metric="rmse",
    log_transform=False,
    n_trials=50,
    maxiter=50,
    random_state=42,
    p_max=2,
    d_max=1,
    q_max=2,
    P_max=1,
    D_max=1,
    Q_max=1,
    trend_options=("n",),
    feature_selection=True,
    min_exog_features=1,
    max_exog_features=None,
):
    """
    Bayesian-style hyperparameter search for SARIMA/SARIMAX using Optuna.

    This version can also do exogenous feature selection.

    If train_exog/test_exog are None:
        Runs normal SARIMA.

    If train_exog/test_exog are provided:
        Runs SARIMAX.

    If feature_selection=True:
        Optuna chooses which exogenous features to include.

    If log_transform=True:
        Uses np.log1p(y), then converts forecast back with np.expm1().
    """

    if metric not in ["rmse", "mse", "mae"]:
        raise ValueError("metric must be 'rmse', 'mse', or 'mae'")

    # -----------------------------
    # Target preparation
    # -----------------------------
    train_y_original = pd.Series(train_y).astype(float).reset_index(drop=True)
    test_y_original = pd.Series(test_y).astype(float).reset_index(drop=True)

    if log_transform:
        if (train_y_original < 0).any() or (test_y_original < 0).any():
            raise ValueError(
                "log_transform=True requires target values >= 0 when using log1p."
            )

        train_y_model = np.log1p(train_y_original)
    else:
        train_y_model = train_y_original.copy()

    # -----------------------------
    # Exogenous preparation
    # -----------------------------
    using_exog = train_exog is not None

    if using_exog:
        if test_exog is None:
            raise ValueError("You must provide test_exog when using train_exog.")

        train_exog_df = pd.DataFrame(train_exog).reset_index(drop=True)
        test_exog_df = pd.DataFrame(test_exog).reset_index(drop=True)

        if len(train_exog_df) != len(train_y_model):
            raise ValueError("train_exog must have the same number of rows as train_y.")

        if len(test_exog_df) != len(test_y_original):
            raise ValueError("test_exog must have the same number of rows as test_y.")

        exog_columns = list(train_exog_df.columns)

        # Make sure test exog has same columns and order
        test_exog_df = test_exog_df[exog_columns]

        if max_exog_features is None:
            max_exog_features = len(exog_columns)

    else:
        train_exog_df = None
        test_exog_df = None
        exog_columns = []
        feature_selection = False

    def objective(trial):

        # -----------------------------
        # SARIMA/SARIMAX order search
        # -----------------------------
        p = trial.suggest_int("p", 0, p_max)
        d = trial.suggest_int("d", 0, d_max)
        q = trial.suggest_int("q", 0, q_max)

        P = trial.suggest_int("P", 0, P_max)
        D = trial.suggest_int("D", 0, D_max)
        Q = trial.suggest_int("Q", 0, Q_max)

        trend = trial.suggest_categorical("trend", trend_options)

        order = (p, d, q)
        seasonal_order = (P, D, Q, seasonal_period)

        # -----------------------------
        # Feature selection
        # -----------------------------
        selected_features = []

        if using_exog and feature_selection:
            for col in exog_columns:
                use_feature = trial.suggest_categorical(f"use_{col}", [True, False])

                if use_feature:
                    selected_features.append(col)

            # Prevent trials with too few or too many features
            if len(selected_features) < min_exog_features:
                return np.inf

            if len(selected_features) > max_exog_features:
                return np.inf

            train_exog_selected = train_exog_df[selected_features]
            test_exog_selected = test_exog_df[selected_features]

        elif using_exog and not feature_selection:
            selected_features = exog_columns
            train_exog_selected = train_exog_df
            test_exog_selected = test_exog_df

        else:
            selected_features = []
            train_exog_selected = None
            test_exog_selected = None

        try:
            model = SARIMAX(
                train_y_model,
                exog=train_exog_selected,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            fitted_model = model.fit(disp=False, maxiter=maxiter)

            forecast_model_scale = fitted_model.forecast(
                steps=len(test_y_original),
                exog=test_exog_selected,
            )

            if log_transform:
                forecast_original = np.expm1(forecast_model_scale)
            else:
                forecast_original = forecast_model_scale

            mse = mean_squared_error(test_y_original, forecast_original)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_y_original, forecast_original)

            trial.set_user_attr("order", order)
            trial.set_user_attr("seasonal_order", seasonal_order)
            trial.set_user_attr("trend", trend)
            trial.set_user_attr("selected_features", selected_features)
            trial.set_user_attr("aic", fitted_model.aic)
            trial.set_user_attr("bic", fitted_model.bic)
            trial.set_user_attr("rmse", rmse)
            trial.set_user_attr("mse", mse)
            trial.set_user_attr("mae", mae)

            if metric == "rmse":
                return rmse
            elif metric == "mse":
                return mse
            else:
                return mae

        except Exception:
            return np.inf

    # -----------------------------
    # Run Optuna
    # -----------------------------
    sampler = optuna.samplers.TPESampler(seed=random_state)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # -----------------------------
    # Build results table
    # -----------------------------
    rows = []

    for trial in study.trials:
        if trial.value is None or not np.isfinite(trial.value):
            continue

        row = {
            **trial.params,
            "value": trial.value,
            "order": trial.user_attrs.get("order"),
            "seasonal_order": trial.user_attrs.get("seasonal_order"),
            "trend": trial.user_attrs.get("trend"),
            "selected_features": trial.user_attrs.get("selected_features"),
            "n_selected_features": len(trial.user_attrs.get("selected_features", [])),
            "aic": trial.user_attrs.get("aic"),
            "bic": trial.user_attrs.get("bic"),
            "rmse": trial.user_attrs.get("rmse"),
            "mse": trial.user_attrs.get("mse"),
            "mae": trial.user_attrs.get("mae"),
        }

        rows.append(row)

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        raise ValueError(
            "No successful trials. Try reducing model complexity, increasing maxiter, "
            "or checking your data/exog variables."
        )

    results_df = results_df.sort_values("value").reset_index(drop=True)

    # -----------------------------
    # Extract best setup
    # -----------------------------
    best_row = results_df.iloc[0]

    best_order = best_row["order"]
    best_seasonal_order = best_row["seasonal_order"]
    best_trend = best_row["trend"]
    best_features = best_row["selected_features"]

    if using_exog:
        if len(best_features) > 0:
            best_train_exog = train_exog_df[best_features]
        else:
            best_train_exog = None
    else:
        best_train_exog = None

    # -----------------------------
    # Refit best model
    # -----------------------------
    best_model = SARIMAX(
        train_y_model,
        exog=best_train_exog,
        order=best_order,
        seasonal_order=best_seasonal_order,
        trend=best_trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=maxiter)

    return best_model, results_df, study, best_features

#-------------------------------------------------------------------------------------------------------------------------------------------------------#

def find_best_sarima_model_for_all_phases_optuna_EXO(
    models_dict,
    order_dict,
    target_col,
    train_data,
    test_data,
    train_exog=None,
    test_exog=None,
    p_max=2,
    d_max=1,
    q_max=2,
    P_max=1,
    D_max=1,
    Q_max=1,
    seasonal_period=7,
    n_trials=50,
    maxiter=50,
    feature_selection=True,
    min_exog_features=1,
    max_exog_features=None,
):
    """
    Finds the best SARIMA/SARIMAX model for oil, gas, and water.

    For each target column:
    1. Runs SARIMAX Optuna search without log transform.
    2. Runs SARIMAX Optuna search with log transform, if target values are non-negative.
    3. Compares both using RMSE.
    4. Stores the better model, order, selected features, and metadata in dictionaries.
    """

    well_name = train_data["NPD_WELL_BORE_NAME"].unique()[0]

    for col in target_col:

        print(f"\nRunning SARIMA/SARIMAX Optuna search for: {well_name} | {col}")

        # -----------------------------
        # Model without log transform
        # -----------------------------
        best_model, results_df, study, best_features = sarimax_optuna_search_EXO(
            train_y=train_data[col],
            test_y=test_data[col],
            train_exog=train_exog,
            test_exog=test_exog,
            seasonal_period=seasonal_period,
            metric="rmse",
            log_transform=False,
            n_trials=n_trials,
            maxiter=maxiter,
            random_state=42,
            p_max=p_max,
            d_max=d_max,
            q_max=q_max,
            P_max=P_max,
            D_max=D_max,
            Q_max=Q_max,
            trend_options=("n",),
            feature_selection=feature_selection,
            min_exog_features=min_exog_features,
            max_exog_features=max_exog_features,
        )

        best_row = results_df.loc[results_df["rmse"].idxmin()]
        rmse = best_row["rmse"]

        # -----------------------------
        # Model with log transform
        # Only run if target values are >= 0
        # -----------------------------
        can_use_log = (train_data[col] >= 0).all() and (test_data[col] >= 0).all()

        if can_use_log:
            best_model_log, results_df_log, study_log, best_features_log = sarimax_optuna_search_EXO(
                train_y=train_data[col],
                test_y=test_data[col],
                train_exog=train_exog,
                test_exog=test_exog,
                seasonal_period=seasonal_period,
                metric="rmse",
                log_transform=True,
                n_trials=n_trials,
                maxiter=maxiter,
                random_state=42,
                p_max=p_max,
                d_max=d_max,
                q_max=q_max,
                P_max=P_max,
                D_max=D_max,
                Q_max=Q_max,
                trend_options=("n",),
                feature_selection=feature_selection,
                min_exog_features=min_exog_features,
                max_exog_features=max_exog_features,
            )

            best_row_log = results_df_log.loc[results_df_log["rmse"].idxmin()]
            rmse_log = best_row_log["rmse"]

        else:
            best_model_log = None
            results_df_log = None
            study_log = None
            best_features_log = []
            best_row_log = None
            rmse_log = float("inf")

            print(f"Skipping log model for {col}: negative values found.")
            print("Train min:", train_data[col].min())
            print("Test min:", test_data[col].min())

        # -----------------------------
        # Print comparison
        # -----------------------------
        print(f"\n{well_name} | {col}")
        print(f"Normal RMSE: {rmse}")
        print(f"Log RMSE:    {rmse_log}")

        # -----------------------------
        # Choose better model
        # -----------------------------
        if rmse <= rmse_log:
            model_key = f"{well_name}_{col}_sarima_model"
            order_key = f"{well_name}_{col}_sarima_model_order"

            models_dict[model_key] = best_model

            order_dict[order_key] = {
                "order": best_row["order"],
                "seasonal_order": best_row["seasonal_order"],
                "trend": best_row.get("trend", "n"),
                "log_transform": False,
                "rmse": rmse,
                "mse": best_row.get("mse", None),
                "mae": best_row.get("mae", None),
                "aic": best_row.get("aic", None),
                "bic": best_row.get("bic", None),
                "selected_features": best_features,
                "model_key": model_key,
            }

            print("Selected model: normal scale")
            print("Selected features:", best_features)

        else:
            model_key = f"{well_name}_{col}_sarima_model_log"
            order_key = f"{well_name}_{col}_sarima_model_log_order"

            models_dict[model_key] = best_model_log

            order_dict[order_key] = {
                "order": best_row_log["order"],
                "seasonal_order": best_row_log["seasonal_order"],
                "trend": best_row_log.get("trend", "n"),
                "log_transform": True,
                "transform_type": "log1p",
                "rmse": rmse_log,
                "mse": best_row_log.get("mse", None),
                "mae": best_row_log.get("mae", None),
                "aic": best_row_log.get("aic", None),
                "bic": best_row_log.get("bic", None),
                "selected_features": best_features_log,
                "model_key": model_key,
            }

            print("Selected model: log1p scale")
            print("Selected features:", best_features_log)

    return models_dict, order_dict

#-----------------------------------------------------------------------------------------------------------------------------------------#

def plot_train_val_predictions_time_EXO(
    train_data,
    val_data,
    phase_string,
    model,
    val_exog=None,
    selected_features=None,
    log_transform=False,
    transform_type="log1p",
    ax=None,
    start_of_val_date=None
):
    """
    Plots train actual, train fitted values, validation actual,
    and validation forecast for a fitted SARIMAX model with optional exogenous variables.

    Train period:
        Uses model.fittedvalues

    Validation period:
        Uses model.forecast(steps=len(val_data), exog=val_exog)

    Important:
        If the model was trained with selected exogenous features,
        pass selected_features so val_exog is filtered correctly.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    color_map = {
        "BORE_OIL_VOL_STB_D": "darkgreen",
        "BORE_GAS_VOL_MSCF_D": "crimson",
        "BORE_WAT_VOL_STB_D": "blue"
    }

    # --------------------------------------------------
    # 1. Actual train and validation data
    # --------------------------------------------------
    y_train = train_data[phase_string].astype(float)
    y_val = val_data[phase_string].astype(float)

    if start_of_val_date is None:
        start_of_val_date = y_val.index[0]

    # --------------------------------------------------
    # 2. In-sample fitted values for train
    # --------------------------------------------------
    train_pred = pd.Series(model.fittedvalues).copy()

    if len(train_pred) > len(y_train):
        train_pred = train_pred.iloc[-len(y_train):]

    train_pred.index = y_train.index[-len(train_pred):]
    y_train_aligned = y_train.loc[train_pred.index]

    # --------------------------------------------------
    # 3. Prepare validation exogenous data
    # --------------------------------------------------
    if val_exog is not None:
        val_exog_used = val_exog.copy()

        if selected_features is not None and len(selected_features) > 0:
            missing_features = [
                col for col in selected_features
                if col not in val_exog_used.columns
            ]

            if missing_features:
                raise ValueError(
                    f"These selected_features are missing from val_exog: {missing_features}"
                )

            val_exog_used = val_exog_used[selected_features]

        # Make sure exog aligns with validation period
        val_exog_used = val_exog_used.loc[y_val.index]

    else:
        val_exog_used = None

    # --------------------------------------------------
    # 4. Forecast validation period
    # --------------------------------------------------
    if val_exog_used is not None:
        val_forecast = model.forecast(
            steps=len(y_val),
            exog=val_exog_used
        )
    else:
        val_forecast = model.forecast(
            steps=len(y_val)
        )

    val_pred = pd.Series(val_forecast).copy()
    val_pred.index = y_val.index

    # --------------------------------------------------
    # 5. Back-transform predictions if needed
    # --------------------------------------------------
    if log_transform:
        if transform_type == "log1p":
            train_pred = np.expm1(train_pred)
            val_pred = np.expm1(val_pred)

        elif transform_type == "log":
            train_pred = np.exp(train_pred)
            val_pred = np.exp(val_pred)

        else:
            raise ValueError("transform_type must be 'log1p' or 'log'")

    # Convert back to Series and clip negative rate predictions
    train_pred = pd.Series(
        train_pred,
        index=y_train_aligned.index
    ).clip(lower=0)

    val_pred = pd.Series(
        val_pred,
        index=y_val.index
    ).clip(lower=0)

    # --------------------------------------------------
    # 6. Validation metrics only
    # --------------------------------------------------
    metric_df = pd.DataFrame({
        "actual": y_val,
        "predicted": val_pred
    }).dropna()

    val_mse = mean_squared_error(
        metric_df["actual"],
        metric_df["predicted"]
    )

    val_rmse = np.sqrt(val_mse)

    val_mae = mean_absolute_error(
        metric_df["actual"],
        metric_df["predicted"]
    )

    # --------------------------------------------------
    # 7. Plot
    # --------------------------------------------------
    is_standalone = ax is None

    if is_standalone:
        plt.figure(figsize=(14, 6))
        ax = plt.gca()

    color = color_map.get(phase_string, "blue")

    # Train actual
    ax.plot(
        y_train.index,
        y_train,
        label="Train Actual",
        linewidth=2,
        color=color,
        alpha=0.65
    )

    # Validation actual
    ax.plot(
        y_val.index,
        y_val,
        label="Validation Actual",
        linewidth=2.5,
        color=color,
        marker="o",
        markersize=4
    )

    # Train fitted
    ax.plot(
        train_pred.index,
        train_pred,
        label="Train Fitted",
        linestyle="--",
        linewidth=2,
        color="gray"
    )

    # Validation forecast
    ax.plot(
        val_pred.index,
        val_pred,
        label="Validation Forecast",
        linestyle="--",
        linewidth=2.5,
        color="black",
        marker="s",
        markersize=3
    )

    # Start of validation line
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

    ax.set_title(
        f"{phase_string} | Validation RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}",
        fontsize=12
    )

    ax.set_ylabel(phase_string, fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    if is_standalone:
        ax.set_xlabel("Date / Time Step", fontsize=12)
        plt.tight_layout()
        plt.show()

    return train_pred, val_pred, val_rmse, val_mae


#-----------------------------------------------------------------------------------------------------------------_#
def plot_train_val_actual_vs_predicted_crossplot_EXO(
    train_data,
    val_data,
    phase_string,
    model,
    val_exog=None,
    selected_features=None,
    log_transform=False,
    transform_type="log1p",
    ax=None
):
    """
    Creates an Actual vs Predicted crossplot for train + validation
    for a SARIMAX model with optional exogenous variables.

    Train period:
        Uses model.fittedvalues

    Validation period:
        Uses model.forecast(steps=len(val_data), exog=val_exog)

    If selected_features is provided, val_exog is filtered to only
    the features used during model training.

    The 45-degree line represents perfect prediction:
        Predicted = Actual
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    color_map = {
        "BORE_OIL_VOL_STB_D": "darkgreen",
        "BORE_GAS_VOL_MSCF_D": "crimson",
        "BORE_WAT_VOL_STB_D": "blue"
    }

    # -----------------------------
    # 1. Actual train and validation data
    # -----------------------------
    y_train = train_data[phase_string].astype(float)
    y_val = val_data[phase_string].astype(float)

    # -----------------------------
    # 2. In-sample fitted values for train
    # -----------------------------
    train_pred = pd.Series(model.fittedvalues).copy()

    # Align fitted values to training index
    if len(train_pred) > len(y_train):
        train_pred = train_pred.iloc[-len(y_train):]

    train_pred.index = y_train.index[-len(train_pred):]
    y_train_aligned = y_train.loc[train_pred.index]

    # -----------------------------
    # 3. Prepare validation exogenous data
    # -----------------------------
    if val_exog is not None:
        val_exog_used = val_exog.copy()

        if selected_features is not None and len(selected_features) > 0:
            missing_features = [
                col for col in selected_features
                if col not in val_exog_used.columns
            ]

            if missing_features:
                raise ValueError(
                    f"These selected_features are missing from val_exog: {missing_features}"
                )

            val_exog_used = val_exog_used[selected_features]

        # Make sure exog has the same index as validation data
        missing_index = y_val.index.difference(val_exog_used.index)

        if len(missing_index) > 0:
            raise ValueError(
                "val_exog is missing some validation dates/index values. "
                f"Example missing index values: {list(missing_index[:5])}"
            )

        val_exog_used = val_exog_used.loc[y_val.index]

    else:
        val_exog_used = None

    # -----------------------------
    # 4. Forecast validation period
    # -----------------------------
    if val_exog_used is not None:
        val_forecast = model.forecast(
            steps=len(y_val),
            exog=val_exog_used
        )
    else:
        val_forecast = model.forecast(
            steps=len(y_val)
        )

    val_pred = pd.Series(val_forecast).copy()
    val_pred.index = y_val.index

    # -----------------------------
    # 5. Back-transform predictions if needed
    # -----------------------------
    if log_transform:
        if transform_type == "log1p":
            train_pred = np.expm1(train_pred)
            val_pred = np.expm1(val_pred)

        elif transform_type == "log":
            train_pred = np.exp(train_pred)
            val_pred = np.exp(val_pred)

        else:
            raise ValueError("transform_type must be 'log1p' or 'log'")

    # Avoid negative rate predictions
    train_pred = pd.Series(
        train_pred,
        index=y_train_aligned.index
    ).clip(lower=0)

    val_pred = pd.Series(
        val_pred,
        index=y_val.index
    ).clip(lower=0)

    # -----------------------------
    # 6. Create crossplot dataframe
    # -----------------------------
    train_crossplot_df = pd.DataFrame({
        "actual": y_train_aligned,
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

    # -----------------------------
    # 7. Metrics
    # -----------------------------
    train_mse = mean_squared_error(
        train_crossplot_df["actual"],
        train_crossplot_df["predicted"]
    )
    train_rmse = np.sqrt(train_mse)

    train_mae = mean_absolute_error(
        train_crossplot_df["actual"],
        train_crossplot_df["predicted"]
    )

    val_mse = mean_squared_error(
        val_crossplot_df["actual"],
        val_crossplot_df["predicted"]
    )
    val_rmse = np.sqrt(val_mse)

    val_mae = mean_absolute_error(
        val_crossplot_df["actual"],
        val_crossplot_df["predicted"]
    )

    # -----------------------------
    # 8. Plot logic
    # -----------------------------
    is_standalone = ax is None

    if is_standalone:
        plt.figure(figsize=(7, 7))
        ax = plt.gca()

    phase_color = color_map.get(phase_string, "blue")

    # Train scatter
    train_points = crossplot_df[crossplot_df["period"] == "Train"]

    ax.scatter(
        train_points["actual"],
        train_points["predicted"],
        label=f"Train Fitted | RMSE: {train_rmse:.2f}",
        color="gray",
        alpha=0.55,
        edgecolor="black",
        linewidth=0.4
    )

    # Validation scatter
    val_points = crossplot_df[crossplot_df["period"] == "Validation"]

    ax.scatter(
        val_points["actual"],
        val_points["predicted"],
        label=f"Validation Forecast | RMSE: {val_rmse:.2f}",
        color=phase_color,
        alpha=0.80,
        edgecolor="black",
        linewidth=0.5
    )

    # -----------------------------
    # 9. 45-degree perfect prediction line
    # -----------------------------
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

    # -----------------------------
    # 10. Styling
    # -----------------------------
    ax.set_xlabel(f"Actual {phase_string}", fontsize=12)
    ax.set_ylabel(f"Predicted {phase_string}", fontsize=12)

    ax.set_title(
        f"{phase_string}\n"
        f"Train RMSE: {train_rmse:.2f}, Val RMSE: {val_rmse:.2f}",
        fontsize=12
    )

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="best")

    if is_standalone:
        well_name = train_data["NPD_WELL_BORE_NAME"].unique()[0]

        plt.suptitle(
            f"{well_name} SARIMAX EXO Actual vs Predicted Crossplot",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout()
        plt.show()

    return crossplot_df, train_rmse, train_mae, val_rmse, val_mae