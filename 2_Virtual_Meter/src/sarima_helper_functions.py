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
#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def plot_decomposition(df, columns,model="additive", period=7 ):
    '''
    This make a decomposition plot for ONLY the target variables
    The dataframe, also a column that have variables to decompose and also a period for the seasonality
    '''
    color_map = {
            'BORE_OIL_VOL_STB_D': 'darkgreen',
            'BORE_GAS_VOL_MSCF_D': 'crimson',
            'BORE_WAT_VOL_STB_D': 'blue'
        }
    for col in columns:
        #Get the assigned color, defaulting to black if it's missing from the map
        col_color = color_map.get(col, 'black')
            
        # 1. Run the decomposition with your specified period
        decompose_plot = seasonal_decompose(df[col], model=model, period=period)
        
        # 2. Generate the plot and capture the figure object
        fig = decompose_plot.plot()
        fig.set_size_inches(12, 8)
    
        # 3. Modify the main overall figure title
        fig.suptitle(f"Decomposition for well {df['NPD_WELL_BORE_NAME'].min()}", fontsize=14, color=col_color)
    
        # 4. Loop through every subplot to change colors to green
        for ax in fig.axes:
            # Change line colors (Observed, Trend, Seasonal, Residual)
            for line in ax.get_lines():
                line.set_color(col_color)
                line.set_linewidth(1.5)
                
            # Change text colors (Y-axis labels and titles)
            ax.title.set_color(col_color)

    # 5. Adjust layout so titles do not overlap, then display
        plt.tight_layout()
        plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def plot_acf_pacf_channels(df, columns, lags=40):
    '''
    Generates side-by-side ACF and PACF plots for specified columns/channels.
    The PACF plot is placed to the right of the ACF plot.
    '''
    color_map = {
        'BORE_OIL_VOL_STB_D': 'darkgreen',
        'BORE_GAS_VOL_MSCF_D': 'crimson',
        'BORE_WAT_VOL_STB_D': 'blue'
    }
    
    # Drop rows with NaNs in the specified columns to prevent statsmodels from crashing
    df_clean = df.dropna(subset=columns)

    for col in columns:
        # Get the assigned color, defaulting to black if missing from the map
        col_color = color_map.get(col, 'black')
        
        # 1. Create a figure with 1 row and 2 columns side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 2. Plot ACF on the left axis (ax1)
        plot_acf(df_clean[col], ax=ax1, lags=lags, color=col_color, vlines_kwargs={"colors": col_color})
        ax1.set_title(f"ACF - {col}", color=col_color, fontsize=12)
        
        # 3. Plot PACF on the right axis (ax2)
        plot_pacf(df_clean[col], ax=ax2, lags=lags, color=col_color, vlines_kwargs={"colors": col_color})
        ax2.set_title(f"PACF - {col}", color=col_color, fontsize=12)
        
        # 4. Global figure title customization
        well_name = df['NPD_WELL_BORE_NAME'].min() if 'NPD_WELL_BORE_NAME' in df.columns else 'Unknown Well'
        fig.suptitle(f"Correlation Analysis for Well {well_name}", fontsize=14, color=col_color, y=1.02)
        
        # 5. Clean up aesthetics for both axes (markers, text colors, grid)
        for ax in [ax1, ax2]:
            ax.tick_params(colors=col_color)
            ax.xaxis.label.set_color(col_color)
            ax.yaxis.label.set_color(col_color)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Optional: Changes the color of the scatter marker dots to match
            for collection in ax.collections:
                collection.set_color(col_color)
                
        # 6. Adjust layout and show the current column's plots
        plt.tight_layout()
        plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_monthly_seasonality(df, columns):
    '''
    Generates seasonal month_plots for specified production channels.
    Resamples the data internally to monthly means before plotting.
    '''
    color_map = {
        'BORE_OIL_VOL_STB_D': 'darkgreen',
        'BORE_GAS_VOL_MSCF_D': 'crimson',
        'BORE_WAT_VOL_STB_D': 'blue'
    }

    for col in columns:
        # 1. Get the assigned color, defaulting to black if missing from the map
        col_color = color_map.get(col, 'black')
        
        # 2. Resample the specific column to monthly average (handling missing values)
        df_monthly = df[col].dropna().resample('ME').mean()
        
        if df_monthly.empty:
            print(f"Skipping {col}: No valid data found for resampling.")
            continue
            
        # 3. Generate the month plot and capture its figure object
        fig = month_plot(df_monthly)
        fig.set_size_inches(12, 5)
        
        # 4. Global figure title customization
        well_name = df['NPD_WELL_BORE_NAME'].min() if 'NPD_WELL_BORE_NAME' in df.columns else 'Unknown Well'
        fig.suptitle(f"Monthly Seasonality for Well {well_name} ({col})", fontsize=14, color=col_color, y=1.02)
        
        # 5. Loop through the internal axes to apply your custom aesthetic styles
        for ax in fig.axes:
            ax.set_ylabel(col, color=col_color, fontsize=11)
            ax.set_xlabel("Month", color=col_color, fontsize=11)
            ax.tick_params(colors=col_color)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Recolor the individual yearly lines/scatter markers
            for line in ax.get_lines():
                line.set_color(col_color)
                line.set_alpha(0.6)  # Slight transparency for overlapping data points
                
            # Recolor the main horizontal mean lines generated by statsmodels
            for collection in ax.collections:
                collection.set_color(col_color)

        # 6. Adjust layout and render
        plt.tight_layout()
        plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def sarima_split_train_test_keep(df, train_frac,val_frac,test_frac):
    ### split data in test and train, and keep
    train=round(train_frac*len(df))
    val=round(val_frac*len(df))
    test=round(test_frac*len(df))

    
    train_data = df.iloc[:train]
    val_data=df.iloc[train:train+val]
    test_data = df.iloc[train+val:train+val+test]
    keep_data = df.iloc[train+val+test:]
    
    return train_data, val_data, test_data, keep_data
#-------------------------------------------------------------------------------------------------------------------------------------------------------#


#############################################################################
# Optuna  grid search
#############################################################################


warnings.filterwarnings("ignore")

def sarimax_optuna_search(
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
    trend_options=("n",)
):
    """
    Bayesian-style hyperparameter search for SARIMA/SARIMAX using Optuna.

    Fits SARIMAX models on train_y and evaluates forecasts on test_y.
    If log_transform=True, uses log1p(y), then converts forecast back with expm1()
    before calculating error on the original scale.
    """

    if metric not in ["rmse", "mse", "mae"]:
        raise ValueError("metric must be 'rmse', 'mse', or 'mae'")

    train_y_original = pd.Series(train_y).astype(float)
    test_y_original = pd.Series(test_y).astype(float)

    if log_transform:
        if (train_y_original < 0).any() or (test_y_original < 0).any():
            raise ValueError("log_transform=True requires target values >= 0 when using log1p.")

        train_y_model = np.log1p(train_y_original)
    else:
        train_y_model = train_y_original.copy()

    using_exog = train_exog is not None

    if using_exog and test_exog is None:
        raise ValueError("You must provide test_exog when using train_exog.")

    def objective(trial):
        p = trial.suggest_int("p", 0, p_max)
        d = trial.suggest_int("d", 0, d_max)
        q = trial.suggest_int("q", 0, q_max)

        P = trial.suggest_int("P", 0, P_max)
        D = trial.suggest_int("D", 0, D_max)
        Q = trial.suggest_int("Q", 0, Q_max)

        trend = trial.suggest_categorical("trend", trend_options)

        order = (p, d, q)
        seasonal_order = (P, D, Q, seasonal_period)

        try:
            model = SARIMAX(
                train_y_model,
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fitted_model = model.fit(disp=False, maxiter=maxiter)

            forecast_model_scale = fitted_model.forecast(
                steps=len(test_y_original),
                exog=test_exog if using_exog else None
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

    sampler = optuna.samplers.TPESampler(seed=random_state)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Build results table
    rows = []

    for trial in study.trials:
        if trial.value is None or not np.isfinite(trial.value):
            continue

        row = {
            **trial.params,
            "value": trial.value,
            "order": trial.user_attrs.get("order"),
            "seasonal_order": trial.user_attrs.get("seasonal_order"),
            "aic": trial.user_attrs.get("aic"),
            "bic": trial.user_attrs.get("bic"),
            "rmse": trial.user_attrs.get("rmse"),
            "mse": trial.user_attrs.get("mse"),
            "mae": trial.user_attrs.get("mae")
        }

        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("value").reset_index(drop=True)

    best_params = study.best_trial.params

    best_order = (
        best_params["p"],
        best_params["d"],
        best_params["q"]
    )

    best_seasonal_order = (
        best_params["P"],
        best_params["D"],
        best_params["Q"],
        seasonal_period
    )

    best_trend = best_params["trend"]

    best_model = SARIMAX(
        train_y_model,
        exog=train_exog,
        order=best_order,
        seasonal_order=best_seasonal_order,
        trend=best_trend,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False, maxiter=maxiter)

    return best_model, results_df, study


#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def find_best_sarima_model_for_all_phases_optuna(
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
):
    """
    Finds the best SARIMA/SARIMAX model for oil, gas, and water.

    For each target column:
    1. Runs SARIMAX Optuna search without log transform.
    2. Runs SARIMAX Optuna search with log transform.
    3. Compares both using RMSE.
    4. Stores the better model and its order in dictionaries.
    """

    well_name = train_data["NPD_WELL_BORE_NAME"].unique()[0]

    for col in target_col:

        # -----------------------------
        # Model without log transform
        # -----------------------------
        best_model, results_df, study = sarimax_optuna_search(
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
        )

        # -----------------------------
        # Model with log transform
        # -----------------------------
        best_model_log, results_df_log, study_log = sarimax_optuna_search(
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
        )

        # Get the actual best rows
        best_row = results_df.loc[results_df["rmse"].idxmin()]
        best_row_log = results_df_log.loc[results_df_log["rmse"].idxmin()]

        rmse = best_row["rmse"]
        rmse_log = best_row_log["rmse"]

        print(f"\n{well_name} | {col}")
        print(f"Normal RMSE: {rmse}")
        print(f"Log RMSE:    {rmse_log}")

        # -----------------------------
        # Choose better model
        # -----------------------------
        if rmse < rmse_log:
            model_key = f"{well_name}_{col}_sarima_model"
            order_key = f"{well_name}_{col}_sarima_model_order"

            models_dict[model_key] = best_model
            order_dict[order_key] = {
                "order": best_row["order"],
                "seasonal_order": best_row["seasonal_order"],
                "log_transform": False,
                "rmse": rmse,
            }

        else:
            model_key = f"{well_name}_{col}_sarima_model_log"
            order_key = f"{well_name}_{col}_sarima_model_log_order"

            models_dict[model_key] = best_model_log
            order_dict[order_key] = {
                "order": best_row_log["order"],
                "seasonal_order": best_row_log["seasonal_order"],
                "log_transform": True,
                "rmse": rmse_log,
            }

    return models_dict, order_dict


#############################################################################################
# PLOTTING FUNCTIONS
#############################################################################################


#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def plot_train_val_predictions_time(
    train_data,
    val_data,
    phase_string,
    model,
    val_exog=None,
    log_transform=False,
    transform_type="log1p",
    ax=None,
    start_of_val_date=None
):
    """
    Plots train actual, train fitted values, validation actual,
    and validation forecast for a fitted SARIMA/SARIMAX model.

    Train period:
        Uses model.fittedvalues

    Validation period:
        Uses model.forecast(steps=len(val_data))

    Parameters
    ----------
    train_data : pd.DataFrame
        Training dataframe.

    val_data : pd.DataFrame
        Validation dataframe.

    phase_string : str
        Target column name.

    model : fitted SARIMAX results object
        Model fitted on the training data.

    val_exog : pd.DataFrame, optional
        Exogenous variables for the validation period.

    log_transform : bool
        True if model was trained on transformed target values.

    transform_type : str
        "log1p" or "log".

    ax : matplotlib axis, optional
        Axis to plot on.

    start_of_val_date : datetime/index value, optional
        Date where validation starts. If None, uses val_data.index[0].
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

    # Align fitted values with training index
    if len(train_pred) > len(y_train):
        train_pred = train_pred.iloc[-len(y_train):]

    train_pred.index = y_train.index[-len(train_pred):]
    y_train_aligned = y_train.loc[train_pred.index]

    # --------------------------------------------------
    # 3. Forecast validation period
    # --------------------------------------------------
    if val_exog is not None:
        val_forecast = model.forecast(
            steps=len(y_val),
            exog=val_exog
        )
    else:
        val_forecast = model.forecast(
            steps=len(y_val)
        )

    val_pred = pd.Series(val_forecast).copy()
    val_pred.index = y_val.index

    # --------------------------------------------------
    # 4. Back-transform predictions if needed
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

    # Convert back to Series and clip negative rates
    train_pred = pd.Series(
        train_pred,
        index=y_train_aligned.index
    ).clip(lower=0)

    val_pred = pd.Series(
        val_pred,
        index=y_val.index
    ).clip(lower=0)

    # --------------------------------------------------
    # 5. Validation metrics only
    # --------------------------------------------------
    val_mse = mean_squared_error(y_val, val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, val_pred)

    # --------------------------------------------------
    # 6. Plot
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

    # Styling
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




#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def plot_train_val_actual_vs_predicted_crossplot(
    train_data,
    val_data,
    phase_string,
    model,
    val_exog=None,
    log_transform=False,
    transform_type="log1p",
    ax=None
):
    """
    Creates an Actual vs Predicted crossplot for train + validation.

    Train period:
        Uses model.fittedvalues

    Validation period:
        Uses model.forecast(steps=len(val_data))

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
    # 3. Forecast validation period
    # -----------------------------
    if val_exog is not None:
        val_forecast = model.forecast(
            steps=len(y_val),
            exog=val_exog
        )
    else:
        val_forecast = model.forecast(
            steps=len(y_val)
        )

    val_pred = pd.Series(val_forecast).copy()
    val_pred.index = y_val.index

    # -----------------------------
    # 4. Back-transform predictions if needed
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
    # 5. Create crossplot dataframe
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
    # 6. Metrics
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
    # 7. Plot logic
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
    # 8. 45-degree perfect prediction line
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
    # 9. Styling
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
            f"{well_name} Actual vs Predicted Crossplot",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout()
        plt.show()

    return crossplot_df, train_rmse, train_mae, val_rmse, val_mae

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

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