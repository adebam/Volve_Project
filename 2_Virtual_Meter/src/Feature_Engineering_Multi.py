
#basics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

#sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import  ParameterGrid, cross_val_score
from sklearn.linear_model import MultiTaskLasso

# feature engine
from feature_engine.timeseries.forecasting import LagFeatures
# sktime
from sktime.transformations.series.time_since import TimeSince

# statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.seasonal import STL

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted



#-----------------------------------------------------------------------------------------------------#

class GOR_WCUT_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, oil_col='oil_vol', gas_col='gas_vol', water_col='water_vol'):
        self.oil_col = oil_col
        self.gas_col = gas_col
        self.water_col = water_col
        
    def fit(self, X, y=None):
        # Check that required columns exist
        required_cols = [self.oil_col, self.gas_col, self.water_col]
        missing_cols = [col for col in required_cols if col not in X.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # This tells sklearn that the transformer has been fitted
        self.is_fitted_ = True
        self.feature_names_in_ = X.columns.to_numpy()

        return self
        
    def transform(self, X):
        # Check that fit() has been called
        check_is_fitted(self, "is_fitted_")

        X_out = X.copy()
        
        oil = X_out[self.oil_col].astype(float)
        gas = X_out[self.gas_col].astype(float)
        water = X_out[self.water_col].astype(float)
        
        total_liquid = oil + water
        
        X_out["GOR_MSCF_STB"] = np.divide(
            gas,
            oil,
            out=np.zeros_like(gas, dtype=float),
            where=oil > 0
        )
        
        X_out["WCUT"] = np.divide(
            water,
            total_liquid,
            out=np.zeros_like(water, dtype=float),
            where=total_liquid > 0
        )
        
        return X_out
#---------------------------------------------------------------------------------------------------------------------------------#
class MultiPhaseMultiOutputLassoSelectedLagFeatureAdder(
    BaseEstimator,
    TransformerMixin,
):
    """
    Creates lag features and uses MultiOutputRegressor(Lasso) to select
    lag features independently for oil, gas, and water.

    Each target receives its own independent Lasso estimator.

    The final selected feature set is the union of:

    1. Features selected by at least one target-specific Lasso model.
    2. Features explicitly supplied through force_keep_lag_map.

    Notes
    -----
    - X is scaled before fitting the target-specific Lasso models.
    - Each target is scaled independently using StandardScaler.
    - All target models use the same alpha selected by GridSearchCV.
    - transform() does not remove missing rows.
    - Missing rows should be removed later in the main feature pipeline.
    """

    def __init__(
        self,
        target_cols=None,
        variables=None,
        max_lag_days=45,
        force_keep_lag_map=None,
        alphas=None,
        n_splits=5,
        max_iter=500000,
        tol=1e-5,
        min_features=10,
        verbose=True,
    ):
        self.target_cols = target_cols
        self.variables = variables
        self.max_lag_days = max_lag_days
        self.force_keep_lag_map = force_keep_lag_map
        self.alphas = alphas
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.tol = tol
        self.min_features = min_features
        self.verbose = verbose

    def fit(self, X, y=None):

        # ----------------------------------------------------------
        # Input validation
        # ----------------------------------------------------------
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        X = X.copy()

        self.original_columns_ = X.columns.tolist()

        if self.target_cols is None:
            self.target_cols_ = [
                "BORE_OIL_VOL_STB_D",
                "BORE_GAS_VOL_MSCF_D",
                "BORE_WAT_VOL_STB_D",
            ]
        else:
            self.target_cols_ = list(self.target_cols)

        if y is None:
            missing_targets = [
                col
                for col in self.target_cols_
                if col not in X.columns
            ]

            if missing_targets:
                raise ValueError(
                    f"These target columns are missing from X: "
                    f"{missing_targets}. Either retain the targets in X "
                    f"during feature engineering or pass y explicitly."
                )

        # ----------------------------------------------------------
        # Hyperparameter grid
        # ----------------------------------------------------------
        if self.alphas is None:
            self.alphas_ = [
                0.001,
                0.01,
                0.05,
                0.1,
                0.2,
                0.5,
                1,
                5,
                10,
                20,
                50,
                100,
            ]
        else:
            self.alphas_ = list(self.alphas)

        if self.force_keep_lag_map is None:
            self.force_keep_lag_map_ = {}
        else:
            self.force_keep_lag_map_ = dict(
                self.force_keep_lag_map
            )

        # max_lag_days=45 creates 1D through 44D
        self.freqs_ = [
            f"{lag}D"
            for lag in range(1, self.max_lag_days)
        ]

        # ----------------------------------------------------------
        # Determine variables for which lags will be generated
        # ----------------------------------------------------------
        if self.variables is None:
            self.variables_ = [
                col
                for col in X.columns
                if pd.api.types.is_numeric_dtype(X[col])
                and "_lag_" not in col
            ]
        else:
            self.variables_ = list(self.variables)

        missing_variables = [
            col
            for col in self.variables_
            if col not in X.columns
        ]

        if missing_variables:
            raise ValueError(
                f"Variables missing from X: {missing_variables}"
            )

        # ----------------------------------------------------------
        # Create lag features
        # ----------------------------------------------------------
        self.lag_transformer_ = LagFeatures(
            variables=self.variables_,
            freq=self.freqs_,
            missing_values="ignore",
        )

        self.lag_transformer_.fit(X)

        X_with_lags = self.lag_transformer_.transform(X)

        all_lag_features = [
            col
            for col in X_with_lags.columns
            if "_lag_" in col
        ]

        if not all_lag_features:
            raise ValueError("No lag features were created.")

        self.candidate_lag_features_ = all_lag_features

        X_lasso = X_with_lags[
            self.candidate_lag_features_
        ].copy()

        # ----------------------------------------------------------
        # Construct multi-output target matrix
        # ----------------------------------------------------------
        if y is None:
            Y_lasso = X_with_lags[
                self.target_cols_
            ].copy()

        else:
            if not isinstance(y, pd.DataFrame):
                raise ValueError(
                    "For multi-output feature selection, y must be "
                    "a pandas DataFrame containing the target columns."
                )

            missing_y_targets = [
                col
                for col in self.target_cols_
                if col not in y.columns
            ]

            if missing_y_targets:
                raise ValueError(
                    f"These target columns are missing from y: "
                    f"{missing_y_targets}"
                )

            Y_lasso = y[self.target_cols_].copy()

            # Match X index after lag creation
            Y_lasso = Y_lasso.reindex(X_lasso.index)

        # ----------------------------------------------------------
        # Remove unusable rows only for internal selector fitting
        # ----------------------------------------------------------
        valid_X_idx = X_lasso.dropna().index
        valid_y_idx = Y_lasso.dropna().index

        valid_idx = valid_X_idx.intersection(valid_y_idx)

        X_lasso_fit = X_lasso.loc[valid_idx]
        Y_lasso_fit = Y_lasso.loc[valid_idx]

        if len(X_lasso_fit) <= self.n_splits:
            raise ValueError(
                "Not enough rows for TimeSeriesSplit. "
                f"Rows after internal NaN removal: "
                f"{len(X_lasso_fit)}, "
                f"n_splits: {self.n_splits}."
            )

        # ----------------------------------------------------------
        # Scale each target independently
        # ----------------------------------------------------------
        self.y_scaler_ = StandardScaler()

        Y_lasso_fit_scaled = self.y_scaler_.fit_transform(
            Y_lasso_fit
        )

        ts_cv = TimeSeriesSplit(
            n_splits=self.n_splits
        )

        # ----------------------------------------------------------
        # Independent Lasso model for each output
        # ----------------------------------------------------------
        base_lasso = Lasso(
            max_iter=self.max_iter,
            tol=self.tol,
            selection="cyclic",
        )

        multioutput_lasso = MultiOutputRegressor(
            estimator=base_lasso,

            # GridSearchCV is already parallelized.
            # Avoid nested parallelism here.
            n_jobs=1,
        )

        multioutput_lasso_pipeline = Pipeline(
            steps=[
                (
                    "x_scaler",
                    StandardScaler(),
                ),
                (
                    "multioutput_lasso",
                    multioutput_lasso,
                ),
            ]
        )

        # MultiOutputRegressor parameter path:
        # pipeline step -> estimator -> alpha
        param_grid = {
            "multioutput_lasso__estimator__alpha":
                self.alphas_
        }

        grid_search = GridSearchCV(
            estimator=multioutput_lasso_pipeline,
            param_grid=param_grid,
            cv=ts_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            refit=True,
        )

        grid_search.fit(
            X_lasso_fit,
            Y_lasso_fit_scaled,
        )

        # Retain the entire fitted selector pipeline
        self.best_selector_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

        self.best_alpha_ = self.best_params_[
            "multioutput_lasso__estimator__alpha"
        ]

        self.best_cv_mse_ = -grid_search.best_score_
        self.best_cv_rmse_ = np.sqrt(self.best_cv_mse_)

        best_multioutput_model = (
            self.best_selector_
            .named_steps["multioutput_lasso"]
        )

        # ----------------------------------------------------------
        # Extract coefficients from each independent Lasso
        # ----------------------------------------------------------
        # Each estimator corresponds to one target.
        #
        # Shape after stacking:
        # n_targets x n_features
        coef_matrix = np.vstack(
            [
                estimator.coef_
                for estimator
                in best_multioutput_model.estimators_
            ]
        )

        self.coef_matrix_ = coef_matrix

        # Combined importance across oil, gas, and water
        self.feature_importance_ = np.linalg.norm(
            coef_matrix,
            axis=0,
        )

        summary_df = pd.DataFrame(
            {
                "feature":
                    self.candidate_lag_features_,
                "importance":
                    self.feature_importance_,
            }
        )

        # Store coefficients and selected features separately
        # for each target.
        self.target_selected_lag_features_ = {}

        for target_index, target in enumerate(
            self.target_cols_
        ):
            target_coef = coef_matrix[target_index, :]

            coef_col = f"{target}_coef_scaled_y"

            summary_df[coef_col] = target_coef

            target_selected_mask = (
                np.abs(target_coef) > 0
            )

            self.target_selected_lag_features_[target] = (
                summary_df.loc[
                    target_selected_mask,
                    "feature",
                ].tolist()
            )

        # A feature is selected if at least one output-specific
        # Lasso assigns it a nonzero coefficient.
        selected_summary_df = (
            summary_df[
                summary_df["importance"] > 0
            ]
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------------
        # Fallback if all target models select zero features
        # ----------------------------------------------------------
        if (
            selected_summary_df.empty
            and self.min_features > 0
        ):
            selected_summary_df = (
                summary_df
                .sort_values(
                    "importance",
                    ascending=False,
                )
                .head(self.min_features)
                .reset_index(drop=True)
            )

        self.lasso_selected_lag_features_ = (
            selected_summary_df["feature"].tolist()
        )

        # ----------------------------------------------------------
        # Force-keep specified lag features
        # ----------------------------------------------------------
        self.force_keep_lag_features_ = []
        self.missing_force_keep_lag_features_ = []

        for variable, lags in (
            self.force_keep_lag_map_.items()
        ):
            for lag in lags:
                lag_col = (
                    f"{variable}_lag_{lag}D"
                )

                if lag_col in X_with_lags.columns:
                    self.force_keep_lag_features_.append(
                        lag_col
                    )
                else:
                    self.missing_force_keep_lag_features_.append(
                        lag_col
                    )

        # Union of force-kept and Lasso-selected features
        self.selected_lag_features_ = (
            self.force_keep_lag_features_
            + self.lasso_selected_lag_features_
        )

        self.selected_lag_features_ = list(
            dict.fromkeys(
                self.selected_lag_features_
            )
        )

        # ----------------------------------------------------------
        # Final output columns
        # ----------------------------------------------------------
        self.final_columns_ = (
            self.original_columns_
            + self.selected_lag_features_
        )

        self.final_columns_ = list(
            dict.fromkeys(self.final_columns_)
        )

        self.feature_summary_df_ = (
            summary_df[
                summary_df["feature"].isin(
                    self.selected_lag_features_
                )
            ]
            .copy()
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        self.is_fitted_ = True

        # ----------------------------------------------------------
        # Fitting summary
        # ----------------------------------------------------------
        if self.verbose:
            print(
                "MultiPhaseMultiOutputLassoSelectedLagFeatureAdder"
                f"__Original columns: "
                f"{len(self.original_columns_)}"
            )

            print(
                "MultiPhaseMultiOutputLassoSelectedLagFeatureAdder"
                f"__All lag features created: "
                f"{len(all_lag_features)}"
            )

            print(
                "MultiPhaseMultiOutputLassoSelectedLagFeatureAdder"
                f"__Candidate lag features: "
                f"{len(self.candidate_lag_features_)}"
            )

            print(
                "MultiPhaseMultiOutputLassoSelectedLagFeatureAdder"
                f"__Rows used internally: "
                f"{len(X_lasso_fit)}"
            )

            print(
                "MultiPhaseMultiOutputLassoSelectedLagFeatureAdder"
                f"__Best shared alpha: "
                f"{self.best_alpha_}"
            )

            print(
                "MultiPhaseMultiOutputLassoSelectedLagFeatureAdder"
                f"__Best CV RMSE: "
                f"{self.best_cv_rmse_:,.4f}"
            )

            for target in self.target_cols_:
                print(
                    "MultiPhaseMultiOutputLassoSelectedLagFeatureAdder"
                    f"__{target} selected features: "
                    f"{len(self.target_selected_lag_features_[target])}"
                )

            print(
                "MultiPhaseMultiOutputLassoSelectedLagFeatureAdder"
                f"__Union of Lasso-selected features: "
                f"{len(self.lasso_selected_lag_features_)}"
            )

            print(
                "MultiPhaseMultiOutputLassoSelectedLagFeatureAdder"
                f"__Force-kept lag features: "
                f"{len(self.force_keep_lag_features_)}"
            )

            print(
                "MultiPhaseMultiOutputLassoSelectedLagFeatureAdder"
                f"__Final selected lag features: "
                f"{len(self.selected_lag_features_)}"
            )

            print(
                "MultiPhaseMultiOutputLassoSelectedLagFeatureAdder"
                f"__Final dataframe columns: "
                f"{len(self.final_columns_)}"
            )

            if self.missing_force_keep_lag_features_:
                print(
                    "Missing force-keep lag features:"
                )
                print(
                    self.missing_force_keep_lag_features_
                )

        return self

    def transform(self, X):
        check_is_fitted(
            self,
            "is_fitted_",
        )

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame."
            )

        X = X.copy()

        X_with_lags = (
            self.lag_transformer_.transform(X)
        )

        missing_selected_features = [
            col
            for col in self.selected_lag_features_
            if col not in X_with_lags.columns
        ]

        if missing_selected_features:
            raise ValueError(
                "Missing selected lag features in "
                f"transform data: "
                f"{missing_selected_features}"
            )

        available_final_columns = [
            col
            for col in self.final_columns_
            if col in X_with_lags.columns
        ]

        return X_with_lags[
            available_final_columns
        ].copy()

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        check_is_fitted(
            self,
            "is_fitted_",
        )

        return np.asarray(
            self.final_columns_,
            dtype=object,
        )


#--------------------------------------------------------------------------------------------------------------------------------#
def plot_ccf(x, y, lags):
    # Compute CCF and confidence interval
    cross_corrs = ccf(x, y)
    ci = 2 / np.sqrt(len(y))
    # Create plot
    fig, ax = plt.subplots(figsize=[10, 5])
    ax.stem(range(0, lags + 1), cross_corrs[: lags + 1])
    ax.fill_between(range(0, lags + 1), ci, y2=-ci, alpha=0.2)
    ax.set_title("Cross-correlation")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return ax
#--------------------------------------------------------------------------------------------------------------------------------#
def lag_plot(x, y, lag, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=[5, 5])
    ax.scatter(y=y, x=x.shift(periods=lag), s=10)
    ax.set_ylabel("$y_t$")
    ax.set_xlabel(f"$x_{{t-{lag}}}$")
    return ax
#--------------------------------------------------------------------------------------------------------------------------------#
class TrendFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Adds time trend features to a dataframe.

    Creates:
        time_since_<freq>
        time_since_<freq>_power_2
        time_since_<freq>_power_3
        etc.

    Example:
        freq="d", degree=2 creates:
            time_since_d
            time_since_d_power_2

    Important:
    - Does not drop rows.
    - Uses the DatetimeIndex.
    - Fit on train, transform validation/test.
    """

    def __init__(
        self,
        freq="d",
        degree=2,
        include_bias=False,
        prefix="time_since",
        overwrite=True,
        verbose=True,
    ):
        self.freq = freq
        self.degree = degree
        self.include_bias = include_bias
        self.prefix = prefix
        self.overwrite = overwrite
        self.verbose = verbose

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(
                "X must have a DatetimeIndex because TimeSince uses the time index."
            )

        self.original_columns_ = X.columns.tolist()

        # This is where TimeSince(freq="d") is used
        self.time_since_transformer_ = TimeSince(freq=self.freq)

        self.poly_transformer_ = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias
        )

        # Fit TimeSince on train data
        time_since_array = self.time_since_transformer_.fit_transform(X)

        # Convert to 2D array if needed
        time_since_array = np.asarray(time_since_array)

        if time_since_array.ndim == 1:
            time_since_array = time_since_array.reshape(-1, 1)

        # Fit PolynomialFeatures on the time_since column
        self.poly_transformer_.fit(time_since_array)

        # Create output column names
        powers = range(0, self.degree + 1) if self.include_bias else range(1, self.degree + 1)

        self.trend_feature_names_ = []

        for power in powers:
            if power == 0:
                col_name = f"{self.prefix}_bias"
            elif power == 1:
                col_name = f"{self.prefix}_{self.freq}"
            else:
                col_name = f"{self.prefix}_{self.freq}_power_{power}"

            self.trend_feature_names_.append(col_name)

        self.final_columns_ = self.original_columns_ + self.trend_feature_names_
        self.final_columns_ = list(dict.fromkeys(self.final_columns_))

        self.is_fitted_ = True

        return self

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(
                "X must have a DatetimeIndex because TimeSince uses the time index."
            )

        X_out = X.copy()

        # Create raw time_since column
        time_since_array = self.time_since_transformer_.transform(X_out)

        time_since_array = np.asarray(time_since_array)

        if time_since_array.ndim == 1:
            time_since_array = time_since_array.reshape(-1, 1)

        # Create polynomial trend features
        trend_array = self.poly_transformer_.transform(time_since_array)

        trend_df = pd.DataFrame(
            trend_array,
            index=X_out.index,
            columns=self.trend_feature_names_,
        )

        if self.overwrite:
            X_out = X_out.drop(columns=self.trend_feature_names_, errors="ignore")

        X_out = pd.concat([X_out, trend_df], axis=1)

        if self.verbose:
            print(f"Trend features added: {self.trend_feature_names_}")

        return X_out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "is_fitted_")
        return np.array(self.final_columns_)

#--------------------------------------------------------------------------------------------------------------------#


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskLasso
from sklearn.metrics import mean_squared_error


class AddSelectedCalendarFourierFeaturesMultiPhase(BaseEstimator, TransformerMixin):
    """
    Creates calendar Fourier features, uses MultiTaskLasso internally to select
    useful Fourier features jointly across oil, gas, and water, and appends only
    the selected Fourier columns to the dataframe.

    This transformer is designed for multi-output time-series forecasting where
    oil, gas, and water are predicted at the same time.

    Supported seasonality options
    -----------------------------
    "daily"        -> day of year, period = 365.25
    "weekly"       -> day of week, period = 7
    "monthly"      -> month of year, period = 12
    "quarterly"    -> quarter of year, period = 4
    "week_of_year" -> ISO week of year, period = 52

    Important
    ---------
    - Fourier features are created from the date index or date column.
    - MultiTaskLasso is used only to select Fourier columns.
    - X Fourier features are scaled inside each time-series CV fold.
    - The multi-output target matrix Y is scaled inside each time-series CV fold.
    - This prevents oil/gas/water scale differences from dominating selection.
    - transform() returns original dataframe + selected Fourier features.
    - transform() does not drop rows.

    Parameters
    ----------
    target_cols : list of str or None, default=None
        Target columns used for multi-output Fourier feature selection.

        If None, defaults to:

        [
            "BORE_OIL_VOL_STB_D",
            "BORE_GAS_VOL_MSCF_D",
            "BORE_WAT_VOL_STB_D",
        ]

    param_grid : list of dict or None, default=None
        Grid of Fourier settings and MultiTaskLasso alpha values.

        Example:

        [
            {
                "fourier__seasonality": ["monthly"],
                "fourier__K": [1, 2, 3, 4, 5, 6],
                "model__alpha": np.logspace(-4, 1, 50),
            },
            {
                "fourier__seasonality": ["week_of_year"],
                "fourier__K": [1, 2, 3, 4, 5, 6, 8, 10],
                "model__alpha": np.logspace(-4, 1, 50),
            },
        ]

    date_col : str or None, default=None
        Name of date column to use for Fourier features.

        If None, X must have a DatetimeIndex.

    keep_original : bool, default=True
        If True, returns original columns plus selected Fourier features.

        If False, returns only selected Fourier features.

    drop_date_col : bool, default=False
        If True and date_col is not None, drops the date column from the final
        output.

    min_features : int, default=1
        Minimum number of Fourier features to keep if MultiTaskLasso selects
        zero non-zero coefficients.

    n_splits : int, default=5
        Number of TimeSeriesSplit folds used during internal CV.

    max_iter : int, default=800000
        Maximum iterations for MultiTaskLasso.

    tol : float, default=1e-5
        Optimization tolerance for MultiTaskLasso.

    verbose : bool, default=True
        If True, prints selected Fourier information after fitting.

    Attributes after fitting
    ------------------------
    original_columns_ : list of str
        Columns present before selected Fourier features are added.

    feature_names_in_ : numpy.ndarray
        Input feature names seen during fit.

    target_cols_ : list of str
        Final target columns used for multi-output selection.

    param_grid_ : list of dict
        Final parameter grid used for Fourier selection.

    best_seasonality_ : str
        Best selected seasonality.

    best_K_ : int
        Best selected Fourier order.

    best_alpha_ : float
        Best selected MultiTaskLasso alpha.

    best_score_ : float
        Best mean CV score. This is negative MSE on scaled multi-output targets.

    best_params_ : dict
        Best Fourier and model parameters.

    candidate_fourier_features_ : list of str
        Fourier features created under the best seasonality and K.

    x_scaler_ : sklearn.preprocessing.StandardScaler
        Final fitted scaler for the best Fourier feature matrix.

    y_scaler_ : sklearn.preprocessing.StandardScaler
        Final fitted scaler for the multi-output target matrix.

    best_estimator_ : sklearn.pipeline.Pipeline
        Final fitted pipeline containing StandardScaler and MultiTaskLasso.

    selected_fourier_features_ : list of str
        Fourier features selected by MultiTaskLasso.

    feature_summary_df_ : pandas.DataFrame
        Summary of selected Fourier features and coefficients.

        Contains:
        - feature
        - importance
        - one coefficient column per target

    final_columns_ : list of str
        Final dataframe columns returned by transform().

    is_fitted_ : bool
        Flag indicating that the transformer has been fitted.
    """

    def __init__(
        self,
        target_cols=None,
        param_grid=None,
        date_col=None,
        keep_original=True,
        drop_date_col=False,
        min_features=1,
        n_splits=5,
        max_iter=800000,
        tol=1e-5,
        verbose=True,
    ):
        self.target_cols = target_cols
        self.param_grid = param_grid
        self.date_col = date_col
        self.keep_original = keep_original
        self.drop_date_col = drop_date_col
        self.min_features = min_features
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        X = X.copy()

        self.original_columns_ = X.columns.tolist()
        self.feature_names_in_ = X.columns.to_numpy()

        if self.target_cols is None:
            self.target_cols_ = [
                "BORE_OIL_VOL_STB_D",
                "BORE_GAS_VOL_MSCF_D",
                "BORE_WAT_VOL_STB_D",
            ]
        else:
            self.target_cols_ = list(self.target_cols)

        # -----------------------------
        # Build multi-output target Y
        # -----------------------------
        if y is None:
            missing_targets = [
                col for col in self.target_cols_
                if col not in X.columns
            ]

            if missing_targets:
                raise ValueError(
                    f"These target columns are missing from X: {missing_targets}. "
                    f"Either keep them in X during feature selection or pass y."
                )

            Y_fit_all = X[self.target_cols_].copy()

        else:
            if isinstance(y, pd.DataFrame):
                missing_targets = [
                    col for col in self.target_cols_
                    if col not in y.columns
                ]

                if missing_targets:
                    raise ValueError(
                        f"These target columns are missing from y: {missing_targets}"
                    )

                Y_fit_all = y[self.target_cols_].copy()
                Y_fit_all = Y_fit_all.loc[X.index]

            elif isinstance(y, np.ndarray):
                if y.ndim != 2:
                    raise ValueError(
                        "For multi-output selection, y must be 2D with shape "
                        "(n_samples, n_targets)."
                    )

                if y.shape[1] != len(self.target_cols_):
                    raise ValueError(
                        f"y has {y.shape[1]} columns, but target_cols has "
                        f"{len(self.target_cols_)} columns."
                    )

                Y_fit_all = pd.DataFrame(
                    y,
                    index=X.index,
                    columns=self.target_cols_,
                )

            else:
                raise ValueError(
                    "For multi-output selection, y must be a pandas DataFrame "
                    "or a 2D numpy array."
                )

        # -----------------------------
        # Default parameter grid
        # -----------------------------
        if self.param_grid is None:
            alpha_grid = np.logspace(-4, 1, 50)

            self.param_grid_ = [
                {
                    "fourier__seasonality": ["daily"],
                    "fourier__K": [1, 2, 3, 4, 5, 6, 8, 10],
                    "model__alpha": alpha_grid,
                },
                {
                    "fourier__seasonality": ["weekly"],
                    "fourier__K": [1, 2, 3],
                    "model__alpha": alpha_grid,
                },
                {
                    "fourier__seasonality": ["monthly"],
                    "fourier__K": [1, 2, 3, 4, 5, 6],
                    "model__alpha": alpha_grid,
                },
                {
                    "fourier__seasonality": ["quarterly"],
                    "fourier__K": [1, 2],
                    "model__alpha": alpha_grid,
                },
                {
                    "fourier__seasonality": ["week_of_year"],
                    "fourier__K": [1, 2, 3, 4, 5, 6, 8, 10],
                    "model__alpha": alpha_grid,
                },
            ]
        else:
            self.param_grid_ = self.param_grid

        dates = self._get_dates(X)

        best_score = -np.inf
        best_result = None

        # -----------------------------
        # Manual TimeSeriesSplit CV
        # This lets us scale Y inside each fold.
        # -----------------------------
        for params in ParameterGrid(self.param_grid_):
            seasonality = params["fourier__seasonality"]
            K = params["fourier__K"]
            alpha = params["model__alpha"]

            X_fourier = self._make_fourier_df(
                dates=dates,
                seasonality=seasonality,
                K=K,
                index=X.index,
            )

            valid_idx = X_fourier.dropna().index.intersection(
                Y_fit_all.dropna().index
            )

            X_fourier_fit = X_fourier.loc[valid_idx]
            Y_lasso_fit = Y_fit_all.loc[valid_idx]

            if len(X_fourier_fit) <= self.n_splits:
                continue

            ts_cv = TimeSeriesSplit(n_splits=self.n_splits)
            fold_scores = []

            for train_idx, val_idx in ts_cv.split(X_fourier_fit):
                X_train = X_fourier_fit.iloc[train_idx]
                X_val = X_fourier_fit.iloc[val_idx]

                Y_train = Y_lasso_fit.iloc[train_idx]
                Y_val = Y_lasso_fit.iloc[val_idx]

                x_scaler = StandardScaler()
                y_scaler = StandardScaler()

                X_train_scaled = x_scaler.fit_transform(X_train)
                X_val_scaled = x_scaler.transform(X_val)

                Y_train_scaled = y_scaler.fit_transform(Y_train)
                Y_val_scaled = y_scaler.transform(Y_val)

                model = MultiTaskLasso(
                    alpha=alpha,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )

                model.fit(X_train_scaled, Y_train_scaled)

                Y_val_pred_scaled = model.predict(X_val_scaled)

                mse = mean_squared_error(
                    Y_val_scaled,
                    Y_val_pred_scaled,
                    multioutput="uniform_average",
                )

                fold_scores.append(-mse)

            mean_score = np.mean(fold_scores)

            if mean_score > best_score:
                best_score = mean_score

                best_result = {
                    "seasonality": seasonality,
                    "K": K,
                    "alpha": alpha,
                    "score": mean_score,
                    "feature_names": X_fourier.columns.tolist(),
                    "X_fourier_fit": X_fourier_fit,
                    "Y_lasso_fit": Y_lasso_fit,
                }

        if best_result is None:
            raise ValueError(
                "No valid Fourier model was fitted. "
                "Check your data length, n_splits, date index, and target columns."
            )

        # -----------------------------
        # Refit final best model on all valid data
        # -----------------------------
        self.best_seasonality_ = best_result["seasonality"]
        self.best_K_ = best_result["K"]
        self.best_alpha_ = best_result["alpha"]
        self.best_score_ = best_result["score"]

        self.best_params_ = {
            "fourier__seasonality": self.best_seasonality_,
            "fourier__K": self.best_K_,
            "model__alpha": self.best_alpha_,
        }

        self.candidate_fourier_features_ = best_result["feature_names"]

        X_best_fit = best_result["X_fourier_fit"]
        Y_best_fit = best_result["Y_lasso_fit"]

        self.y_scaler_ = StandardScaler()
        Y_best_fit_scaled = self.y_scaler_.fit_transform(Y_best_fit)

        self.best_estimator_ = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "multitask_lasso",
                    MultiTaskLasso(
                        alpha=self.best_alpha_,
                        max_iter=self.max_iter,
                        tol=self.tol,
                    ),
                ),
            ]
        )

        self.best_estimator_.fit(X_best_fit, Y_best_fit_scaled)

        best_lasso = self.best_estimator_.named_steps["multitask_lasso"]

        # MultiTaskLasso coef_ shape: n_targets x n_features
        coef_matrix = best_lasso.coef_

        # One importance value per feature across oil/gas/water
        feature_importance = np.linalg.norm(coef_matrix, axis=0)

        summary_df = pd.DataFrame(
            {
                "feature": self.candidate_fourier_features_,
                "importance": feature_importance,
            }
        )

        for i, target in enumerate(self.target_cols_):
            summary_df[f"{target}_coef_scaled_y"] = coef_matrix[i, :]

        selected_summary_df = (
            summary_df[summary_df["importance"] > 0]
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        if len(selected_summary_df) == 0 and self.min_features > 0:
            selected_summary_df = (
                summary_df
                .sort_values("importance", ascending=False)
                .head(self.min_features)
                .reset_index(drop=True)
            )

        self.feature_summary_df_ = selected_summary_df
        self.selected_fourier_features_ = self.feature_summary_df_[
            "feature"
        ].tolist()

        if self.keep_original:
            self.final_columns_ = (
                self.original_columns_
                + self.selected_fourier_features_
            )

            if self.date_col is not None and self.drop_date_col:
                self.final_columns_ = [
                    col for col in self.final_columns_
                    if col != self.date_col
                ]
        else:
            self.final_columns_ = self.selected_fourier_features_

        self.final_columns_ = list(dict.fromkeys(self.final_columns_))

        self.is_fitted_ = True

        if self.verbose:
            print(f"AddSelectedCalendarFourierFeaturesMultiPhase__Best seasonality: {self.best_seasonality_}")
            print(f"AddSelectedCalendarFourierFeaturesMultiPhase__Best K: {self.best_K_}")
            print(f"AddSelectedCalendarFourierFeaturesMultiPhase__Best alpha: {self.best_alpha_}")
            print(f"AddSelectedCalendarFourierFeaturesMultiPhase__Best CV score: {self.best_score_}")
            print(f"AddSelectedCalendarFourierFeaturesMultiPhase__Candidate Fourier features: {len(self.candidate_fourier_features_)}")
            print(f"AddSelectedCalendarFourierFeaturesMultiPhase__Selected Fourier features: {len(self.selected_fourier_features_)}")
            print(f"AddSelectedCalendarFourierFeaturesMultiPhase__Final dataframe columns returned: {len(self.final_columns_)}")

        return self

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        X = X.copy()
        dates = self._get_dates(X)

        X_fourier = self._make_fourier_df(
            dates=dates,
            seasonality=self.best_seasonality_,
            K=self.best_K_,
            index=X.index,
        )

        missing_cols = [
            col for col in self.selected_fourier_features_
            if col not in X_fourier.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Selected Fourier columns missing in transform data: {missing_cols}"
            )

        selected_fourier_df = X_fourier[
            self.selected_fourier_features_
        ].copy()

        if self.keep_original:
            X_out = X.copy()

            if self.date_col is not None and self.drop_date_col:
                X_out = X_out.drop(columns=[self.date_col], errors="ignore")

            X_out = X_out.drop(
                columns=self.selected_fourier_features_,
                errors="ignore",
            )

            X_out = pd.concat([X_out, selected_fourier_df], axis=1)

        else:
            X_out = selected_fourier_df.copy()

        return X_out

    def _get_dates(self, X):
        if self.date_col is not None:
            if self.date_col not in X.columns:
                raise ValueError(f"{self.date_col} not found in X columns.")

            return pd.to_datetime(X[self.date_col])

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(
                "X must have a DatetimeIndex if date_col is not provided."
            )

        return pd.Series(X.index, index=X.index)

    def _get_time_values_and_period(self, dates, seasonality):
        seasonality = seasonality.lower()

        if seasonality == "daily":
            t = dates.dt.dayofyear.astype(float)
            period = 365.25
            prefix = "day_of_year"

        elif seasonality == "weekly":
            t = dates.dt.dayofweek.astype(float)
            period = 7
            prefix = "day_of_week"

        elif seasonality == "monthly":
            t = dates.dt.month.astype(float)
            period = 12
            prefix = "month"

        elif seasonality == "quarterly":
            t = dates.dt.quarter.astype(float)
            period = 4
            prefix = "quarter"

        elif seasonality == "week_of_year":
            t = dates.dt.isocalendar().week.astype(float)
            period = 52
            prefix = "week_of_year"

        else:
            raise ValueError(
                "seasonality must be one of: "
                "'daily', 'weekly', 'monthly', 'quarterly', 'week_of_year'"
            )

        return t, period, prefix

    def _make_fourier_df(self, dates, seasonality, K, index):
        t, period, prefix = self._get_time_values_and_period(
            dates=dates,
            seasonality=seasonality,
        )

        fourier_df = pd.DataFrame(index=index)

        for k in range(1, K + 1):
            fourier_df[f"{prefix}_sin_{k}"] = np.sin(
                2 * np.pi * k * t / period
            )
            fourier_df[f"{prefix}_cos_{k}"] = np.cos(
                2 * np.pi * k * t / period
            )

        return fourier_df

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "is_fitted_")
        return np.array(self.final_columns_)



#-------------------------------------------------------------------------------------------------#

class AddSpecificLag1Features(BaseEstimator, TransformerMixin):
    """
    Adds 1-day lag features for specific columns using Feature-engine.

    Naming convention:
        variable -> variable_lag_1D

    Example:
        AVG_WHT_P -> AVG_WHT_P_lag_1D

    Important:
    - If variable_lag_1D already exists, it is not added again.
    - Does not drop rows.
    - Newly created lag columns will have NaN at the beginning.
    - Drop NaNs later with your final missing-row dropper.
    """

    def __init__(
        self,
        variables=None,
        freq="1D",
        verbose=True,
    ):
        self.variables = variables
        self.freq = freq
        self.verbose = verbose

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(
                "X must have a DatetimeIndex because Feature-engine LagFeatures "
                "with freq='1D' uses the time index."
            )

        if self.variables is None:
            raise ValueError("You must provide variables to lag.")

        X = X.copy()

        self.original_columns_ = X.columns.tolist()
        self.variables_ = list(self.variables)

        missing_variables = [
            var for var in self.variables_
            if var not in X.columns
        ]

        if missing_variables:
            raise ValueError(
                f"These variables are not in X and cannot be lagged: {missing_variables}"
            )

        # Example:
        # AVG_WHT_P -> AVG_WHT_P_lag_1D
        self.expected_lag_features_ = [
            f"{var}_lag_{self.freq}"
            for var in self.variables_
        ]

        # Lag columns that already exist in the dataframe
        self.existing_lag_features_ = [
            lag_col for lag_col in self.expected_lag_features_
            if lag_col in X.columns
        ]

        # Only lag variables whose lag column does not already exist
        self.variables_to_lag_ = [
            var for var in self.variables_
            if f"{var}_lag_{self.freq}" not in X.columns
        ]

        self.lag_features_to_add_ = [
            f"{var}_lag_{self.freq}"
            for var in self.variables_to_lag_
        ]

        if len(self.variables_to_lag_) > 0:
            self.lag_transformer_ = LagFeatures(
                variables=self.variables_to_lag_,
                freq=[self.freq],
            )

            self.lag_transformer_.fit(X)
        else:
            self.lag_transformer_ = None

        self.final_columns_ = (
            self.original_columns_
            + self.lag_features_to_add_
        )

        self.final_columns_ = list(dict.fromkeys(self.final_columns_))

        self.is_fitted_ = True

        if self.verbose:
            print(f"AddSpecificLag1Features__Requested variables: {len(self.variables_)}")
            print(f"AddSpecificLag1Features__Existing lag columns skipped: {len(self.existing_lag_features_)}")
            print(f"AddSpecificLag1Features__New lag columns to add: {len(self.lag_features_to_add_)}")

            if len(self.existing_lag_features_) > 0:
                print("Skipped existing lag columns:")
                print(self.existing_lag_features_)

        return self

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(
                "X must have a DatetimeIndex because Feature-engine LagFeatures "
                "with freq='1D' uses the time index."
            )

        X_out = X.copy()

        # If all requested lag columns already existed during fit,
        # there is nothing new to add.
        if self.lag_transformer_ is None:
            return X_out

        X_with_lags = self.lag_transformer_.transform(X_out)

        # Add only lag columns that are still missing in this dataframe
        lag_cols_to_add_now = [
            col for col in self.lag_features_to_add_
            if col in X_with_lags.columns and col not in X_out.columns
        ]

        if len(lag_cols_to_add_now) > 0:
            X_out = pd.concat(
                [X_out, X_with_lags[lag_cols_to_add_now]],
                axis=1
            )

        if self.verbose:
            print(f"AddSpecificLag1Features__New lag columns added: {lag_cols_to_add_now}")

        return X_out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "is_fitted_")
        return np.array(self.final_columns_)

#---------------------------------------------------------------------#
class ProductionCurveFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Adds fitted production-curve trend features to a dataframe.

    Supports:
    - logistic
    - gompertz
    - richards
    - hyperbolic

    Important:
    ----------
    This transformer fits curves on the data passed to .fit() only.
    So if you use it inside a pipeline fitted on train data, it is leakage-safe.

    Typical use:
        Oil/Gas  -> hyperbolic, logistic, gompertz, richards
        Water    -> logistic, gompertz, richards

    Parameters
    ----------
    target_cols : list[str]
        Target columns to fit curves on.

    curve_types : list[str] or dict
        Curves to try. Example:
            ["logistic", "gompertz", "richards", "hyperbolic"]

        Or target-specific:
            {
                "BORE_OIL_VOL_STB_D": ["hyperbolic", "logistic"],
                "BORE_WAT_VOL_STB_D": ["logistic", "gompertz", "richards"],
            }

    output : {"best", "all"}
        "best" -> adds only best curve trend per target.
        "all"  -> adds every successfully fitted curve trend per target.

    date_col : str or None
        If None, the dataframe index must be DatetimeIndex.
        If str, that column is used as datetime.

    on_stream_col : str or None
        Optional column used to filter fitting days.

    min_on_stream : float
        Minimum ON_STREAM_HRS used during curve fitting.

    fit_start_strategy : str or dict
        "all"         -> fit on all valid training points.
        "peak_to_end" -> fit only after peak rate. Useful for oil/gas decline.

        Can also be target-specific:
            {
                "BORE_OIL_VOL_STB_D": "peak_to_end",
                "BORE_GAS_VOL_MSCF_D": "peak_to_end",
                "BORE_WAT_VOL_STB_D": "all",
            }

    smoothing_window : int or None
        Rolling median smoothing window used before fitting.
        Helps avoid fitting shut-ins/noise.

    t_scale_days : float
        Converts elapsed days to curve time units.
        365.25 means curve parameters are roughly per-year.

    clip_negative : bool
        Clip curve predictions below zero.

    add_slope : bool
        Add approximate curve slope feature.

    add_residual_lag : bool
        Add lagged residual feature:
            target_actual - curve_trend

        Be careful: this should only be used if your future/validation dataframe
        has target history available in a leakage-safe way.
    """

    def __init__(
        self,
        target_cols,
        curve_types=("logistic", "gompertz", "richards", "hyperbolic"),
        output="best",
        date_col=None,
        on_stream_col=None,
        min_on_stream=12,
        min_rate=0,
        fit_start_strategy="all",
        smoothing_window=14,
        t_scale_days=365.25,
        clip_negative=True,
        add_slope=True,
        add_residual_lag=False,
        residual_lag_periods=1,
        feature_prefix_map=None,
        maxfev=50000,
    ):
        self.target_cols = target_cols
        self.curve_types = curve_types
        self.output = output
        self.date_col = date_col
        self.on_stream_col = on_stream_col
        self.min_on_stream = min_on_stream
        self.min_rate = min_rate
        self.fit_start_strategy = fit_start_strategy
        self.smoothing_window = smoothing_window
        self.t_scale_days = t_scale_days
        self.clip_negative = clip_negative
        self.add_slope = add_slope
        self.add_residual_lag = add_residual_lag
        self.residual_lag_periods = residual_lag_periods
        self.feature_prefix_map = feature_prefix_map
        self.maxfev = maxfev

    # ------------------------------------------------------------------
    # Curve equations
    # ------------------------------------------------------------------

    @staticmethod
    def _logistic(t, A, K, k, t0):
        """
        Generalized logistic.
        Can fit increasing or decreasing behavior depending on A and K.
        """
        z = np.clip(-k * (t - t0), -100, 100)
        return A + (K - A) / (1.0 + np.exp(z))

    @staticmethod
    def _gompertz(t, A, K, k, t0):
        """
        Generalized Gompertz.
        Can fit increasing or decreasing behavior depending on A and K.
        """
        z = np.clip(-k * (t - t0), -100, 100)
        return A + (K - A) * np.exp(-np.exp(z))

    @staticmethod
    def _richards(t, A, K, k, t0, nu):
        """
        Richards curve.
        More flexible version of logistic.
        """
        z = np.clip(-k * (t - t0), -100, 100)
        return A + (K - A) / ((1.0 + np.exp(z)) ** (1.0 / nu))

    @staticmethod
    def _hyperbolic(t, qi, Di, b):
        """
        Arps hyperbolic decline.

        q(t) = qi / (1 + b * Di * t) ** (1 / b)
        """
        base = 1.0 + b * Di * t
        base = np.maximum(base, 1e-12)
        return qi / (base ** (1.0 / b))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_datetime_values(self, X):
        if self.date_col is None:
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError(
                    "X must have a DatetimeIndex when date_col=None."
                )
            return pd.to_datetime(X.index)
        else:
            return pd.to_datetime(X[self.date_col])

    def _get_elapsed_days(self, X):
        dates = self._get_datetime_values(X)
        elapsed_days = (dates - self.global_start_date_).days.astype(float)
        return np.asarray(elapsed_days), dates

    def _resolve_curve_types(self, target_col):
        if isinstance(self.curve_types, dict):
            return self.curve_types.get(target_col, [])
        return list(self.curve_types)

    def _resolve_fit_start_strategy(self, target_col):
        if isinstance(self.fit_start_strategy, dict):
            return self.fit_start_strategy.get(target_col, "all")
        return self.fit_start_strategy

    def _prefix(self, target_col):
        if self.feature_prefix_map is not None:
            return self.feature_prefix_map.get(target_col, target_col)
        return target_col

    def _smooth_y(self, y, index):
        y_series = pd.Series(y, index=index).astype(float)

        if self.smoothing_window is None or self.smoothing_window <= 1:
            return y_series

        return (
            y_series
            .rolling(
                window=self.smoothing_window,
                min_periods=max(3, self.smoothing_window // 3),
                center=True,
            )
            .median()
            .bfill()
            .ffill()
        )

    def _initial_params_and_bounds(self, curve_type, t, y):
        y = np.asarray(y, dtype=float)
        t = np.asarray(t, dtype=float)

        y_min = max(0.0, np.nanmin(y))
        y_max = max(1.0, np.nanmax(y))

        n_edge = max(3, int(0.1 * len(y)))
        y_start = np.nanmedian(y[:n_edge])
        y_end = np.nanmedian(y[-n_edge:])

        upper_y = max(y_max * 5.0, 1.0)
        t_min = float(np.nanmin(t))
        t_max = float(np.nanmax(t))
        t_mid = float(np.nanmedian(t))

        # For logistic/Gompertz/Richards:
        # A = early asymptote
        # K = late asymptote
        A0 = max(0.0, y_start)
        K0 = max(0.0, y_end)

        if curve_type == "logistic":
            p0 = [A0, K0, 1.0, t_mid]
            lower = [0.0, 0.0, 1e-5, t_min - 5.0]
            upper = [upper_y, upper_y, 50.0, t_max + 5.0]
            return p0, (lower, upper)

        if curve_type == "gompertz":
            p0 = [A0, K0, 1.0, t_mid]
            lower = [0.0, 0.0, 1e-5, t_min - 5.0]
            upper = [upper_y, upper_y, 50.0, t_max + 5.0]
            return p0, (lower, upper)

        if curve_type == "richards":
            p0 = [A0, K0, 1.0, t_mid, 1.0]
            lower = [0.0, 0.0, 1e-5, t_min - 5.0, 0.1]
            upper = [upper_y, upper_y, 50.0, t_max + 5.0, 10.0]
            return p0, (lower, upper)

        if curve_type == "hyperbolic":
            qi0 = max(y_start, y_max)
            p0 = [qi0, 0.5, 0.7]

            lower = [0.0, 1e-6, 0.01]
            upper = [upper_y, 20.0, 3.0]
            return p0, (lower, upper)

        raise ValueError(f"Unknown curve_type: {curve_type}")

    def _get_curve_function(self, curve_type):
        if curve_type == "logistic":
            return self._logistic
        if curve_type == "gompertz":
            return self._gompertz
        if curve_type == "richards":
            return self._richards
        if curve_type == "hyperbolic":
            return self._hyperbolic
        raise ValueError(f"Unknown curve_type: {curve_type}")

    def _fit_single_curve(self, curve_type, t, y):
        func = self._get_curve_function(curve_type)
        p0, bounds = self._initial_params_and_bounds(curve_type, t, y)

        params, _ = curve_fit(
            func,
            t,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=self.maxfev,
        )

        y_hat = func(t, *params)

        if self.clip_negative:
            y_hat = np.clip(y_hat, 0, None)

        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
        mae = float(np.mean(np.abs(y - y_hat)))

        return {
            "curve_type": curve_type,
            "params": params,
            "rmse": rmse,
            "mae": mae,
            "success": True,
        }

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        X = X.copy()

        if self.output not in ["best", "all"]:
            raise ValueError("output must be 'best' or 'all'.")

        self.global_start_date_ = self._get_datetime_values(X).min()
        elapsed_days, dates = self._get_elapsed_days(X)

        self.fitted_curves_ = {}
        self.best_curve_ = {}
        self.curve_fit_origin_ = {}
        self.fit_summary_ = []

        for target_col in self.target_cols:
            if target_col not in X.columns:
                raise ValueError(f"{target_col} not found in X.")

            curve_types = self._resolve_curve_types(target_col)

            if len(curve_types) == 0:
                continue

            y_raw = X[target_col].astype(float).copy()

            fit_mask = np.isfinite(y_raw)
            fit_mask &= y_raw > self.min_rate

            if self.on_stream_col is not None and self.on_stream_col in X.columns:
                fit_mask &= X[self.on_stream_col].astype(float) >= self.min_on_stream

            fit_dates = pd.Series(dates[fit_mask], index=X.index[fit_mask])
            fit_y_raw = y_raw.loc[fit_mask]

            if len(fit_y_raw) < 10:
                self.fit_summary_.append(
                    {
                        "target": target_col,
                        "curve_type": None,
                        "success": False,
                        "reason": "not enough valid fitting points",
                    }
                )
                continue

            strategy = self._resolve_fit_start_strategy(target_col)

            if strategy == "peak_to_end":
                smoothed_for_peak = self._smooth_y(
                    fit_y_raw.values,
                    fit_y_raw.index,
                )
                peak_idx = smoothed_for_peak.idxmax()
                fit_y_raw = fit_y_raw.loc[fit_y_raw.index >= peak_idx]
                fit_dates = fit_dates.loc[fit_dates.index >= peak_idx]

            elif strategy == "all":
                pass

            else:
                raise ValueError(
                    "fit_start_strategy must be 'all', 'peak_to_end', "
                    "or a dict using those values."
                )

            if len(fit_y_raw) < 10:
                self.fit_summary_.append(
                    {
                        "target": target_col,
                        "curve_type": None,
                        "success": False,
                        "reason": "not enough valid points after fit_start_strategy",
                    }
                )
                continue

            y_smooth = self._smooth_y(fit_y_raw.values, fit_y_raw.index)

            curve_origin_date = fit_dates.min()
            t_fit_days = (fit_dates - curve_origin_date).dt.days.astype(float)
            t_fit = np.asarray(t_fit_days) / self.t_scale_days
            y_fit = np.asarray(y_smooth, dtype=float)

            finite_mask = np.isfinite(t_fit) & np.isfinite(y_fit)

            t_fit = t_fit[finite_mask]
            y_fit = y_fit[finite_mask]

            fitted_for_target = []

            for curve_type in curve_types:
                try:
                    result = self._fit_single_curve(curve_type, t_fit, y_fit)
                    result["target"] = target_col
                    result["fit_n"] = len(y_fit)
                    result["fit_origin_date"] = curve_origin_date
                    result["selected"] = False

                    fitted_for_target.append(result)

                    self.fit_summary_.append(
                        {
                            "target": target_col,
                            "curve_type": curve_type,
                            "success": True,
                            "rmse": result["rmse"],
                            "mae": result["mae"],
                            "fit_n": len(y_fit),
                            "fit_origin_date": curve_origin_date,
                            "params": result["params"],
                            "selected": False,
                        }
                    )

                except Exception as e:
                    self.fit_summary_.append(
                        {
                            "target": target_col,
                            "curve_type": curve_type,
                            "success": False,
                            "reason": str(e),
                        }
                    )

            if len(fitted_for_target) == 0:
                continue

            best = min(fitted_for_target, key=lambda d: d["rmse"])

            self.fitted_curves_[target_col] = {
                item["curve_type"]: item for item in fitted_for_target
            }

            self.best_curve_[target_col] = best["curve_type"]

            for item in fitted_for_target:
                curve_type = item["curve_type"]
                self.curve_fit_origin_[(target_col, curve_type)] = curve_origin_date

            # Update selected flag in summary
            for row in self.fit_summary_:
                if (
                    row.get("target") == target_col
                    and row.get("curve_type") == best["curve_type"]
                    and row.get("success") is True
                ):
                    row["selected"] = True

        self.fit_summary_ = pd.DataFrame(self.fit_summary_)

        return self

    def transform(self, X):
        check_is_fitted(
            self,
            ["fitted_curves_", "best_curve_", "curve_fit_origin_", "global_start_date_"],
        )

        X_out = X.copy()
        dates = self._get_datetime_values(X_out)

        for target_col in self.target_cols:
            if target_col not in self.fitted_curves_:
                continue

            prefix = self._prefix(target_col)

            if self.output == "best":
                curve_types_to_output = [self.best_curve_[target_col]]
            else:
                curve_types_to_output = list(self.fitted_curves_[target_col].keys())

            for curve_type in curve_types_to_output:
                model_info = self.fitted_curves_[target_col][curve_type]
                params = model_info["params"]
                func = self._get_curve_function(curve_type)

                origin_date = self.curve_fit_origin_[(target_col, curve_type)]
                t_days = (dates - origin_date).days.astype(float)
                t_days = np.maximum(t_days, 0.0)
                t = np.asarray(t_days) / self.t_scale_days

                trend = func(t, *params)

                if self.clip_negative:
                    trend = np.clip(trend, 0, None)

                if self.output == "best":
                    trend_col = f"{prefix}_curve_trend"
                    slope_col = f"{prefix}_curve_slope"
                    resid_lag_col = f"{prefix}_curve_residual_lag_{self.residual_lag_periods}"
                else:
                    trend_col = f"{prefix}_{curve_type}_curve_trend"
                    slope_col = f"{prefix}_{curve_type}_curve_slope"
                    resid_lag_col = (
                        f"{prefix}_{curve_type}_curve_residual_lag_{self.residual_lag_periods}"
                    )

                X_out[trend_col] = trend

                if self.add_slope:
                    if len(trend) > 1:
                        day_values = np.asarray((dates - dates.min()).days.astype(float))
                        day_values = np.maximum(day_values, 0.0)

                        if np.nanmax(day_values) > np.nanmin(day_values):
                            slope = np.gradient(trend, day_values)
                        else:
                            slope = np.zeros_like(trend)
                    else:
                        slope = np.zeros_like(trend)

                    X_out[slope_col] = slope

                if self.add_residual_lag:
                    if target_col in X_out.columns:
                        residual = X_out[target_col].astype(float) - trend
                        X_out[resid_lag_col] = residual.shift(self.residual_lag_periods)
                    else:
                        X_out[resid_lag_col] = np.nan

        return X_out