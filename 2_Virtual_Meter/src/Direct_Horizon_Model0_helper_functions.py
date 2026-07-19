"""
MODEL 0: naive direct multi-horizon Lasso models.

Renamed from the original "Model 1" once we clarified what Model 1 actually
needed to be (see Direct_Horizon_Model1a_helper_functions.py). Model 0 is
kept because it answers a different, still-useful question: "at a single
fixed distance from a real, fully-known anchor, how does accuracy degrade
as that distance grows?" It is NOT a simulation of how the virtual meter
would run in production -- there is no block, no reset, no recursion. Every
anchor (train or val, any day) is treated completely independently: each
(phase, horizon) pair gets its own Lasso model that predicts the phase
target exactly `horizon` days after an anchor date, using only information
available at the anchor date, plus the planned future value of
ON_STREAM_HRS at anchor + horizon. No day's prediction ever depends on
another day's prediction.

AVG_CHOKE_SIZE_P and DP_CHOKE_SIZE_PSI are deliberately excluded (both as
lag inputs and as the future-known planned feature). Choke size ramps from
~0-26 in the training period up to a ceiling of 100 later in the well's
life, and a linear model extrapolates that relationship catastrophically
once validation-period choke values run well past the training range
(ablation: Val R2 went from -9.97 to +0.94 for oil, -8.57 to +0.94 for gas,
after dropping these columns). Add choke back in only with an encoding that
doesn't break under that kind of unbounded linear extrapolation.

No recursion, no GOR/WCUT, and each phase only sees its own target lags
(full phase independence).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Machine_Learning_Multi import (
    plot_train_val_actual_vs_predicted_crossplot,
    plot_train_val_predictions_time,
)

SENSOR_VARIABLES = [
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE_PSI",
    "AVG_DOWNHOLE_TEMPERATURE_F",
    "AVG_DP_TUBING_PSI",
    "AVG_ANNULUS_PRESS_PSI",
    "AVG_WHP_P_PSI",
    "AVG_WHT_P",
]

FUTURE_KNOWN_VARIABLES = ["ON_STREAM_HRS"]

TARGET_COLS = [
    "BORE_OIL_VOL_STB_D",
    "BORE_GAS_VOL_MSCF_D",
    "BORE_WAT_VOL_STB_D",
]

SENSOR_LAG_DAYS = [1, 2, 3, 7, 14, 30]
TARGET_LAG_DAYS = [1, 3, 7, 14, 30]

MAX_LOOKBACK_DAYS = max(SENSOR_LAG_DAYS + TARGET_LAG_DAYS)


# ------------------------------------------------------------------#
def build_model0_direct_horizon_table(
    full_df,
    target_col,
    horizon,
    anchor_start=None,
    anchor_end=None,
    sensor_variables=SENSOR_VARIABLES,
    future_known_variables=FUTURE_KNOWN_VARIABLES,
    sensor_lag_days=SENSOR_LAG_DAYS,
    target_lag_days=TARGET_LAG_DAYS,
):
    """
    Build a table of anchor-date features and the target value at
    anchor + horizon, for one phase and one fixed forecast horizon.

    Every feature is computed strictly from information available at the
    anchor date, except `future_known_variables`, whose value at
    anchor + horizon is used directly (the "planned schedule" assumption).

    `full_df` must contain enough history before `anchor_start`
    (>= MAX_LOOKBACK_DAYS) and enough future data after `anchor_end`
    (>= horizon days) for lag / future features to be computed without
    truncation at the edges of the requested anchor window.

    Returns
    -------
    X : pd.DataFrame indexed by anchor date
    y : pd.Series indexed by anchor date, the target at anchor + horizon
    """
    full_df = full_df.sort_index()

    if full_df.index.has_duplicates:
        raise ValueError("full_df contains duplicate index values.")

    features = pd.DataFrame(index=full_df.index)

    for var in sensor_variables:
        for lag in sensor_lag_days:
            features[f"{var}_lag_{lag}D"] = full_df[var].shift(lag)

    for lag in target_lag_days:
        features[f"{target_col}_lag_{lag}D"] = full_df[target_col].shift(lag)

    forecast_dates = full_df.index + pd.Timedelta(days=horizon)

    iso_week = forecast_dates.isocalendar()["week"].astype(float).to_numpy()
    month = forecast_dates.month.astype(float).to_numpy()

    features["horizon_week_sin"] = np.sin(2 * np.pi * iso_week / 52)
    features["horizon_week_cos"] = np.cos(2 * np.pi * iso_week / 52)
    features["horizon_month_sin"] = np.sin(2 * np.pi * month / 12)
    features["horizon_month_cos"] = np.cos(2 * np.pi * month / 12)

    for var in future_known_variables:
        features[f"{var}_future_h"] = full_df[var].shift(-horizon)

    target = full_df[target_col].shift(-horizon)

    table = features.copy()
    table["__target__"] = target
    table = table.dropna()

    if anchor_start is not None:
        table = table[table.index >= pd.Timestamp(anchor_start)]

    if anchor_end is not None:
        table = table[table.index <= pd.Timestamp(anchor_end)]

    y = table.pop("__target__")
    X = table

    return X, y


# ------------------------------------------------------------------#
def build_single_phase_lasso_grid(n_splits=10, alphas=None):
    if alphas is None:
        alphas = np.logspace(-3, 3, 200)

    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    lasso_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                Lasso(max_iter=500000, tol=1e-5, random_state=42),
            ),
        ]
    )

    lasso_model = TransformedTargetRegressor(
        regressor=lasso_pipeline,
        transformer=StandardScaler(),
    )

    param_grid = {"regressor__model__alpha": alphas}

    return GridSearchCV(
        estimator=lasso_model,
        param_grid=param_grid,
        cv=ts_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )


# ------------------------------------------------------------------#
def build_single_phase_ridge_grid(n_splits=10, alphas=None):
    """
    Ridge for one (phase, horizon). Same shape as the Lasso builder:
    scaled features, scaled target, tuned by time-series CV.

    Ridge keeps every lag with a shrunken coefficient instead of zeroing
    most of them out, which is the useful contrast against Lasso here.
    """
    if alphas is None:
        alphas = np.logspace(-4, 6, 100)

    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    ridge_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(solver="auto", tol=1e-5)),
        ]
    )

    ridge_model = TransformedTargetRegressor(
        regressor=ridge_pipeline,
        transformer=StandardScaler(),
    )

    return GridSearchCV(
        estimator=ridge_model,
        param_grid={"regressor__model__alpha": alphas},
        cv=ts_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )


# ------------------------------------------------------------------#
def build_single_phase_elasticnet_grid(n_splits=10, alphas=None, l1_ratios=None):
    """
    ElasticNet for one (phase, horizon) -- a blend of the Lasso and Ridge
    behaviour above.

    The grid is deliberately small: this builder is called once per
    (phase, horizon) pair, so 3 x 30 = 90 grid searches per sweep.
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 15)

    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9, 1.0]

    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    elasticnet_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    max_iter=500000,
                    tol=1e-5,
                    selection="cyclic",
                    random_state=42,
                ),
            ),
        ]
    )

    elasticnet_model = TransformedTargetRegressor(
        regressor=elasticnet_pipeline,
        transformer=StandardScaler(),
    )

    return GridSearchCV(
        estimator=elasticnet_model,
        param_grid={
            "regressor__model__alpha": alphas,
            "regressor__model__l1_ratio": l1_ratios,
        },
        cv=ts_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )


# ------------------------------------------------------------------#
def build_single_phase_random_forest_grid(
    n_splits=5,
    n_estimators=200,
    max_depth_grid=None,
    min_samples_leaf_grid=None,
):
    """
    Random Forest for one (phase, horizon).

    Unlike the linear models, a forest cannot extrapolate past the range of
    the training targets at all -- it can only average training values. It
    is included as the non-extrapolating baseline to compare the linear and
    linear-tree models against.

    Both the grid and the CV fold count are kept small on purpose: this
    builder runs 90 times per sweep (3 phases x 30 horizons), so a wide
    grid here costs hours. Trees need no feature scaling, but the pipeline
    is kept for a consistent interface with the other builders.
    """
    if max_depth_grid is None:
        max_depth_grid = [3, 6, None]

    if min_samples_leaf_grid is None:
        min_samples_leaf_grid = [2, 5]

    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    random_forest_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return GridSearchCV(
        estimator=random_forest_pipeline,
        param_grid={
            "model__max_depth": max_depth_grid,
            "model__min_samples_leaf": min_samples_leaf_grid,
        },
        cv=ts_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,  # the forest itself already uses every core
        refit=True,
    )


# ------------------------------------------------------------------#
def build_single_phase_gradient_boost_grid(
    n_splits=5,
    n_estimators=200,
    learning_rate=0.05,
    max_depth_grid=None,
    min_samples_leaf_grid=None,
):
    """
    Gradient Boosting for one (phase, horizon).

    Shallow trees, a low learning rate and a small grid, for the same
    reason as the Random Forest builder: 90 grid searches per sweep.
    """
    if max_depth_grid is None:
        max_depth_grid = [2, 3]

    if min_samples_leaf_grid is None:
        min_samples_leaf_grid = [5, 20]

    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    gradient_boost_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42,
                ),
            ),
        ]
    )

    return GridSearchCV(
        estimator=gradient_boost_pipeline,
        param_grid={
            "model__max_depth": max_depth_grid,
            "model__min_samples_leaf": min_samples_leaf_grid,
        },
        cv=ts_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )


# ------------------------------------------------------------------#
def build_single_phase_lightgbm_grid(
    n_splits=10,
    linear_tree=True,
    num_leaves_grid=None,
    min_child_samples_grid=None,
    linear_lambda_grid=None,
    learning_rate=0.05,
    n_estimators=150,
):
    """
    Conservative, tunable LightGBM regressor for one phase.

    linear_tree=True fits a linear regression at each leaf instead of a
    constant value. Unlike a plain lag-based linear model (Lasso), which
    can only ever echo a weighted combination of past values, a linear
    tree can fit a different local slope per leaf -- directly targets the
    trend-lag bias found in the Lasso models (Lasso structurally
    undershoots a steadily rising series like water, since it has no way
    to extrapolate a trend, only echo recent lags).

    Deliberately conservative given ~700 training rows:
    - num_leaves is capped low (a handful of leaves, not LightGBM's
      default 31) so the tree stays shallow.
    - min_child_samples (min_data_in_leaf) is kept high so every leaf's
      linear regression is fit on enough rows to be a stable estimate,
      not a regression through a handful of points.
    - linear_lambda regularizes the per-leaf linear regressions directly
      (separate from num_leaves/min_child_samples, which only control
      tree structure).
    - learning_rate is low and n_estimators modest, rather than a large
      boosted ensemble that could overfit a small dataset.

    Set linear_tree=False to fall back to standard (constant-leaf)
    LightGBM for comparison.
    """
    if num_leaves_grid is None:
        num_leaves_grid = [3, 5, 7]

    if min_child_samples_grid is None:
        min_child_samples_grid = [50, 100]

    if linear_lambda_grid is None:
        linear_lambda_grid = [0.1, 1.0]

    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    base_model = LGBMRegressor(
        linear_tree=linear_tree,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        verbose=-1,
        # GridSearchCV already parallelizes across folds/combinations --
        # keep each individual fit single-threaded to avoid nested
        # parallelism.
        n_jobs=1,
    )

    param_grid = {
        "num_leaves": num_leaves_grid,
        "min_child_samples": min_child_samples_grid,
        "linear_lambda": linear_lambda_grid,
    }

    return GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=ts_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )


# ------------------------------------------------------------------#
def get_lasso_selected_features(models):
    """
    For each phase's fitted Lasso pipeline, return the features Lasso
    selected -- those with a nonzero coefficient after regularization --
    sorted by |coefficient| descending within each phase.

    Coefficients are in standardized-X, standardized-y units (both scaled
    by StandardScaler inside the pipeline / TransformedTargetRegressor),
    so magnitude is comparable across features within a phase, but not a
    physical-unit sensitivity.

    models : dict[target_col -> fitted TransformedTargetRegressor]
        A TransformedTargetRegressor wrapping Pipeline([StandardScaler,
        Lasso]), as returned by fit_model0_for_horizon /
        train_model1a_base_models with the default (Lasso) model_builder.

    Returns
    -------
    pd.DataFrame with columns Phase, Feature, Coefficient, Abs_Coefficient
    -- one row per selected feature per phase.
    """
    rows = []

    for phase, model in models.items():
        lasso = model.regressor_.named_steps["model"]
        feature_names = model.feature_names_in_

        for name, coef in zip(feature_names, lasso.coef_):
            if coef != 0:
                rows.append(
                    {
                        "Phase": phase,
                        "Feature": name,
                        "Coefficient": coef,
                        "Abs_Coefficient": abs(coef),
                    }
                )

    selected_df = pd.DataFrame(rows)

    if not selected_df.empty:
        selected_df = selected_df.sort_values(
            ["Phase", "Abs_Coefficient"], ascending=[True, False]
        ).reset_index(drop=True)

    return selected_df


# ------------------------------------------------------------------#
def fit_model0_for_horizon(
    df,
    train_data,
    val_data,
    horizon,
    target_cols=TARGET_COLS,
    sensor_variables=SENSOR_VARIABLES,
    future_known_variables=FUTURE_KNOWN_VARIABLES,
    n_splits=10,
    alphas=None,
    model_builder=None,
    verbose=True,
):
    """
    Fit and evaluate one independent model per phase for a single fixed
    forecast horizon.

    Training anchors are restricted to train_data's own date range (no
    peeking past the end of train_data for targets). Validation anchors
    are restricted to val_data's own date range, but features/targets are
    built from the full `df` so lag features have enough lookback and
    targets have enough lookahead near the edges of val_data.

    sensor_variables, future_known_variables : list of str
        Passed straight through to build_model0_direct_horizon_table --
        override these (e.g. from a notebook cell) to change which
        columns are used, without editing this module.

    model_builder : callable(n_splits) -> unfitted GridSearchCV, or None
        Defaults to build_single_phase_lasso_grid(n_splits=n_splits,
        alphas=alphas). Pass e.g.
        functools.partial(build_single_phase_lightgbm_grid, linear_tree=True)
        to swap the algorithm -- everything downstream (walk-forward
        simulation, diagnostics, plotting) only ever calls model.predict()
        and reads model.feature_names_in_, so it doesn't need to change.

    Returns
    -------
    models : dict[target_col -> fitted estimator]
    X_train_dict, y_train_dict, X_val_dict, y_val_dict : dict[target_col -> DataFrame/Series]
    walk_forward_forecast_df : pd.DataFrame indexed by forecast date
        (anchor + horizon), columns = target_cols, containing the model's
        validation predictions.
    metrics_df : pd.DataFrame with Train/Val metrics per phase.
    """
    models = {}
    X_train_dict = {}
    y_train_dict = {}
    X_val_dict = {}
    y_val_dict = {}

    forecast_predictions = {}
    metrics_rows = []

    for target_col in target_cols:
        if verbose:
            print(f"\n[horizon={horizon}] Fitting {target_col}...")

        X_train, y_train = build_model0_direct_horizon_table(
            train_data,
            target_col=target_col,
            horizon=horizon,
            anchor_start=train_data.index.min(),
            anchor_end=train_data.index.max(),
            sensor_variables=sensor_variables,
            future_known_variables=future_known_variables,
        )

        X_val, y_val = build_model0_direct_horizon_table(
            df,
            target_col=target_col,
            horizon=horizon,
            anchor_start=val_data.index.min(),
            anchor_end=val_data.index.max(),
            sensor_variables=sensor_variables,
            future_known_variables=future_known_variables,
        )

        # Align validation feature columns to whatever training produced.
        X_val = X_val[X_train.columns]

        if model_builder is not None:
            grid = model_builder(n_splits=n_splits)
        else:
            grid = build_single_phase_lasso_grid(n_splits=n_splits, alphas=alphas)

        grid.fit(X_train, y_train)

        model = grid.best_estimator_

        models[target_col] = model
        X_train_dict[target_col] = X_train
        y_train_dict[target_col] = y_train
        X_val_dict[target_col] = X_val
        y_val_dict[target_col] = y_val

        train_pred = np.clip(model.predict(X_train), 0, None)
        val_pred = np.clip(model.predict(X_val), 0, None)

        forecast_dates = X_val.index + pd.Timedelta(days=horizon)

        forecast_predictions[target_col] = pd.Series(
            val_pred,
            index=forecast_dates,
            name=target_col,
        )

        # Forecast dates (anchor + horizon) can extend past val_data's own
        # last date into test_data's date range. Score against the full
        # continuous series so every validation anchor gets a fair score.
        val_actual_at_forecast_dates = df.reindex(forecast_dates)[
            target_col
        ]

        # Generic across algorithms: Lasso's grid.best_params_ looks like
        # {"regressor__model__alpha": ...}, LightGBM's looks like
        # {"num_leaves": ..., "min_child_samples": ..., "linear_lambda": ...}.
        # Store the whole dict, and additionally surface "Best_Alpha" when
        # it's actually a Lasso alpha, for backward compatibility with
        # code that reads that column specifically.
        best_params = grid.best_params_
        best_alpha = best_params.get("regressor__model__alpha", np.nan)

        metrics_rows.append(
            {
                "Horizon": horizon,
                "Phase": target_col,
                "Best_Params": best_params,
                "Best_Alpha": best_alpha,
                "Train_RMSE": root_mean_squared_error(y_train, train_pred),
                "Train_MAE": mean_absolute_error(y_train, train_pred),
                "Train_R2": r2_score(y_train, train_pred),
                "Val_RMSE": root_mean_squared_error(
                    val_actual_at_forecast_dates, val_pred
                ),
                "Val_MAE": mean_absolute_error(
                    val_actual_at_forecast_dates, val_pred
                ),
                "Val_R2": r2_score(val_actual_at_forecast_dates, val_pred),
                "N_Train_Points": len(y_train),
                "N_Val_Points": len(val_pred),
            }
        )

    walk_forward_forecast_df = pd.concat(
        forecast_predictions.values(), axis=1
    ).sort_index()

    metrics_df = pd.DataFrame(metrics_rows)

    return (
        models,
        X_train_dict,
        y_train_dict,
        X_val_dict,
        y_val_dict,
        walk_forward_forecast_df,
        metrics_df,
    )


# ------------------------------------------------------------------#
def run_model0_horizon_sweep(
    df,
    train_data,
    val_data,
    horizons=(1, 7, 14, 30),
    target_cols=TARGET_COLS,
    sensor_variables=SENSOR_VARIABLES,
    future_known_variables=FUTURE_KNOWN_VARIABLES,
    n_splits=10,
    alphas=None,
    model_builder=None,
    verbose=True,
):
    """
    Run fit_model0_for_horizon() across a grid of horizon values.

    sensor_variables, future_known_variables : list of str
        Passed straight through to fit_model0_for_horizon -- override
        these (e.g. from a notebook cell) to change which columns are
        used, without editing this module.

    model_builder : callable(n_splits) -> unfitted GridSearchCV, or None
        Defaults to Lasso (build_single_phase_lasso_grid). Pass e.g.
        functools.partial(build_single_phase_lightgbm_grid, linear_tree=True)
        to swap the algorithm for every horizon in the sweep.

    Returns
    -------
    results : dict[horizon -> dict with keys
        'models', 'X_train', 'y_train', 'X_val', 'y_val',
        'walk_forward_forecast_df', 'metrics_df']
    all_metrics_df : pd.DataFrame, concatenation of every horizon's
        metrics_df, for the horizon-vs-accuracy comparison.
    """
    results = {}
    all_metrics = []

    for horizon in horizons:
        (
            models,
            X_train_dict,
            y_train_dict,
            X_val_dict,
            y_val_dict,
            walk_forward_forecast_df,
            metrics_df,
        ) = fit_model0_for_horizon(
            df=df,
            train_data=train_data,
            val_data=val_data,
            horizon=horizon,
            target_cols=target_cols,
            sensor_variables=sensor_variables,
            future_known_variables=future_known_variables,
            n_splits=n_splits,
            alphas=alphas,
            model_builder=model_builder,
            verbose=verbose,
        )

        results[horizon] = {
            "models": models,
            "X_train": X_train_dict,
            "y_train": y_train_dict,
            "X_val": X_val_dict,
            "y_val": y_val_dict,
            "walk_forward_forecast_df": walk_forward_forecast_df,
            "metrics_df": metrics_df,
        }

        all_metrics.append(metrics_df)

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)

    return results, all_metrics_df


# ------------------------------------------------------------------#
def plot_multi_phase_direct_horizon_diagnostics(
    models,
    X_train,
    y_train,
    horizon,
    df,
    walk_forward_forecast_df,
    target_cols=None,
    model_name="Model",
    start_of_val_date=None,
    clip_negative=True,
    make_combined_plots=True,
):
    """
    Train / validation diagnostics for direct multi-horizon models.

    plot_multi_phase_train_walkforward_val_diagnostics (in
    Machine_Learning_Multi.py) was built for one-step / recursive models,
    where the prediction date equals the row's own index. That assumption
    breaks for direct multi-horizon models: X_train/y_train are indexed by
    *anchor* date, but each prediction is *about* anchor + horizon.

    This function plots every point -- train and validation alike -- at
    its forecast date (anchor + horizon), so the calendar position always
    matches the date the value is actually about, and the train/val curves
    use the same convention.

    Parameters
    ----------
    models : dict[target_col -> fitted estimator]
    X_train : dict[target_col -> DataFrame], anchor-date indexed
    y_train : dict[target_col -> Series], anchor-date indexed, values are
        the target at anchor + horizon
    horizon : int
        Forecast horizon in days, used to shift train's anchor index to
        its forecast date.
    df : pd.DataFrame
        Full continuous dataframe with true target values, used to look
        up validation actuals at each forecast date.
    walk_forward_forecast_df : pd.DataFrame
        Validation predictions, already indexed by forecast date
        (anchor + horizon), columns = target_cols.
    target_cols : list or None
        Defaults to models.keys().
    start_of_val_date : datetime-like or None
        The validation split's anchor-date boundary (e.g. val_data.index.min()).
        Since points are plotted at forecast date, not anchor date, this is
        shifted by +horizon internally before being drawn, so the marker
        lines up with where the validation curve actually starts rather
        than sitting `horizon` days before it.

    Returns
    -------
    train_pred_df, val_pred_df : pd.DataFrame, forecast-date indexed
    metrics_df : pd.DataFrame
    """
    if target_cols is None:
        target_cols = list(models.keys())
    else:
        target_cols = list(target_cols)

    start_of_val_forecast_date = (
        None
        if start_of_val_date is None
        else pd.Timestamp(start_of_val_date) + pd.Timedelta(days=horizon)
    )

    train_actual_series = {}
    train_pred_series = {}
    val_actual_series = {}
    val_pred_series = {}
    metrics = []

    for phase in target_cols:
        model = models[phase]
        X_train_phase = X_train[phase]
        y_train_phase = y_train[phase]

        train_forecast_dates = X_train_phase.index + pd.Timedelta(days=horizon)

        train_pred = pd.Series(
            model.predict(X_train_phase),
            index=train_forecast_dates,
            name=phase,
            dtype=float,
        )
        train_actual = pd.Series(
            y_train_phase.to_numpy(),
            index=train_forecast_dates,
            name=phase,
            dtype=float,
        )

        val_pred = walk_forward_forecast_df[phase].astype(float).copy()
        val_actual = df.reindex(val_pred.index)[phase].astype(float)

        if clip_negative:
            train_pred = train_pred.clip(lower=0)
            val_pred = val_pred.clip(lower=0)

        val_valid_mask = val_actual.notna() & val_pred.notna()
        val_actual_valid = val_actual.loc[val_valid_mask]
        val_pred_valid = val_pred.loc[val_valid_mask]

        metrics.append(
            {
                "Model": model_name,
                "Phase": phase,
                "Horizon": horizon,
                "Train_RMSE": root_mean_squared_error(train_actual, train_pred),
                "Train_MAE": mean_absolute_error(train_actual, train_pred),
                "Train_R2": r2_score(train_actual, train_pred),
                "WalkForward_Val_RMSE": root_mean_squared_error(
                    val_actual_valid, val_pred_valid
                ),
                "WalkForward_Val_MAE": mean_absolute_error(
                    val_actual_valid, val_pred_valid
                ),
                "WalkForward_Val_R2": r2_score(val_actual_valid, val_pred_valid),
                "N_Train_Points": len(train_actual),
                "N_WalkForward_Val_Points": len(val_actual_valid),
            }
        )

        train_actual_series[phase] = train_actual
        train_pred_series[phase] = train_pred
        val_actual_series[phase] = val_actual_valid
        val_pred_series[phase] = val_pred_valid

    metrics_df = pd.DataFrame(metrics)

    train_pred_df = pd.concat(
        [train_pred_series[phase].rename(phase) for phase in target_cols],
        axis=1,
    )
    val_pred_df = pd.concat(
        [val_pred_series[phase].rename(phase) for phase in target_cols],
        axis=1,
    )

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
                y_train=train_actual_series[phase],
                train_pred=train_pred_series[phase],
                y_val=val_actual_series[phase],
                val_pred=val_pred_series[phase],
                phase_string=phase,
                train_rmse=row["Train_RMSE"],
                train_mae=row["Train_MAE"],
                val_rmse=row["WalkForward_Val_RMSE"],
                val_mae=row["WalkForward_Val_MAE"],
                ax=ax,
                start_of_val_date=start_of_val_forecast_date,
                clip_negative=False,
            )

        fig.suptitle(
            f"{model_name} (horizon={horizon}D): Train Fit and "
            "Walk-Forward Validation Actual vs Predicted",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

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
            row = metrics_df[metrics_df["Phase"] == phase].iloc[0]

            plot_train_val_actual_vs_predicted_crossplot(
                y_train=train_actual_series[phase],
                train_pred=train_pred_series[phase],
                y_val=val_actual_series[phase],
                val_pred=val_pred_series[phase],
                phase_string=phase,
                train_rmse=row["Train_RMSE"],
                train_mae=row["Train_MAE"],
                val_rmse=row["WalkForward_Val_RMSE"],
                val_mae=row["WalkForward_Val_MAE"],
                ax=ax,
                clip_negative=False,
            )

        fig.suptitle(
            f"{model_name} (horizon={horizon}D): Actual vs Predicted Crossplots",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    return train_pred_df, val_pred_df, metrics_df
