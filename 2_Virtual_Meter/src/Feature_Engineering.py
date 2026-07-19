
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


#------------------------------------------------------------------------------------------------------#
class OnStreamHoursCategorizer(BaseEstimator, TransformerMixin):
    """
    Converts ON_STREAM_HRS into categorical bins:

        <1hr      : ON_STREAM_HRS < 1
        1-12hr    : 1 <= ON_STREAM_HRS < 12
        12-24hr   : 12 <= ON_STREAM_HRS <= 24

    Values outside 0-24 or missing values become NaN.
    """

    def __init__(
        self,
        column="ON_STREAM_HRS",
        new_column="ON_STREAM_HRS_CAT",
        drop_original=False,
        as_category=True,
    ):
        self.column = column
        self.new_column = new_column
        self.drop_original = drop_original
        self.as_category = as_category

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame.")

        self.feature_names_in_ = X.columns.to_list()
        self.categories_ = ["<1hr", "1-12hr", "12-24hr"]

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        X = X.copy()

        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame.")

        hrs = pd.to_numeric(X[self.column], errors="coerce")

        cat = pd.Series(pd.NA, index=X.index, dtype="object")

        cat.loc[hrs < 1] = "<1hr"
        cat.loc[(hrs >= 1) & (hrs < 12)] = "1-12hr"
        cat.loc[(hrs >= 12) & (hrs <= 24)] = "12-24hr"

        if self.as_category:
            cat = pd.Categorical(
                cat,
                categories=self.categories_,
                ordered=True,
            )

        X[self.new_column] = cat

        if self.drop_original:
            X = X.drop(columns=[self.column])

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        output_features = list(input_features)

        if self.drop_original and self.column in output_features:
            output_features.remove(self.column)

        if self.new_column not in output_features:
            output_features.append(self.new_column)

        return output_features

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



class LassoSelectedLagFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Creates lag features using Feature-engine, uses Lasso only for lag-feature selection,
    and returns the original dataframe plus selected lag features.
    Also force-keeps specified ACF/PACF lags, for example 1D, 2D, 6D, 7D., or any other user defined lags 

    Important:
    - NaNs are dropped internally only for fitting Lasso.
    - transform() does NOT drop rows.
    - Drop missing rows later with a final cleaning transformer.

    Parameters
    -----------
    target_col : str
        Target column used for Lasso feature selection.

    variables : list or None
        Columns to create lags for. If None, numeric columns are selected automatically.

    max_lag_days : int
        Maximum lag range. For example, max_lag_days=45 creates lags 1D to 44D.

    targets_to_remove : list or None
        Target-related lag columns to exclude from Lasso candidates, usually gas/water
        targets when building an oil model.

    force_keep_lag_map : dict or None
        User-specified lags to keep regardless of Lasso selection.
        Example:
        {
            "BORE_OIL_VOL_STB_D": [1, 2, 6, 7],
            "GOR_MSCF_STB": [1, 7],
        }

    Attributes after fitting
    ------------------------
    original_columns_ : list
        Columns present before lag features are added.

    freqs_ : list
        Lag frequencies used by Feature-engine, such as ["1D", "2D", ..., "44D"].

    variables_ : list
        Final list of variables used to generate lag features.

    columns_to_drop_ : list
        Lag columns removed from Lasso candidates.

    candidate_lag_features_ : list
        Lag features considered by Lasso.

    best_alpha_ : float
        Best Lasso alpha selected by GridSearchCV.

    lasso_selected_lag_features_ : list
        Lag features selected by Lasso.

    force_keep_lag_features_ : list
        User-specified lag features kept regardless of Lasso.

    selected_lag_features_ : list
        Final lag features returned by the transformer.

    final_columns_ : list
        Final dataframe columns returned by transform().

    feature_summary_df_ : pd.DataFrame
        Summary of selected features and Lasso coefficients.
    """

    def __init__(
        self,
        target_col="BORE_OIL_VOL_STB_D",
        variables=None,
        max_lag_days=45,
        targets_to_remove=None,
        force_keep_lag_map=None,
        alphas=None,
        n_splits=5,
        max_iter=500000,
        tol=1e-5,
        min_features=5,
        verbose=True,
    ):
        self.target_col = target_col
        self.variables = variables
        self.max_lag_days = max_lag_days
        self.targets_to_remove = targets_to_remove
        self.force_keep_lag_map = force_keep_lag_map
        self.alphas = alphas
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.tol = tol
        self.min_features = min_features
        self.verbose = verbose

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        X = X.copy()

        self.original_columns_ = X.columns.tolist()

        if self.target_col not in X.columns and y is None:
            raise ValueError(
                f"target_col='{self.target_col}' is not in X, and y was not provided."
            )

        # Default alphas
        if self.alphas is None:
            self.alphas_ = [
                0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 1.0,
                5, 10, 20, 50, 100,
            ]
        else:
            self.alphas_ = self.alphas

        # Default target lags to remove from Lasso candidates
        if self.targets_to_remove is None:
            self.targets_to_remove_ = [
                "BORE_GAS_VOL_MSCF_D",
                "BORE_WAT_VOL_STB_D",
                "BORE_GAS_VOL",
                "BORE_WAT_VOL",
            ]
        else:
            self.targets_to_remove_ = self.targets_to_remove

        # Default force-keep lag map
        if self.force_keep_lag_map is None:
            self.force_keep_lag_map_ = {}
        else:
            self.force_keep_lag_map_ = self.force_keep_lag_map

        # Example: max_lag_days=45 gives 1D to 44D
        self.freqs_ = [f"{i}D" for i in range(1, self.max_lag_days)]

        # Variables to lag
        if self.variables is None:
            self.variables_ = [
                col for col in X.columns
                if pd.api.types.is_numeric_dtype(X[col]) and "_lag_" not in col
            ]
        else:
            self.variables_ = list(self.variables)

        # Create lag transformer
        self.lag_transformer_ = LagFeatures(
            variables=self.variables_,
            freq=self.freqs_,
        )

        self.lag_transformer_.fit(X)

        # Create all lag features for Lasso selection
        X_with_lags = self.lag_transformer_.transform(X)

        # Only lag columns generated by Feature-engine
        all_lag_features = [
            col for col in X_with_lags.columns
            if "_lag_" in col
        ]

        # Remove gas/water lags from candidate features
        self.columns_to_drop_ = [
            col for col in all_lag_features
            if any(tgt in col for tgt in self.targets_to_remove_)
        ]

        self.candidate_lag_features_ = [
            col for col in all_lag_features
            if col not in self.columns_to_drop_
        ]

        if len(self.candidate_lag_features_) == 0:
            raise ValueError("No candidate lag features found for Lasso selection.")

        X_lasso = X_with_lags[self.candidate_lag_features_].copy()

        # Target for Lasso
        if y is None:
            y_lasso = X_with_lags[self.target_col].copy()
        else:
            if isinstance(y, pd.Series):
                y_lasso = y.copy()
            else:
                y_lasso = pd.Series(y, index=X.index, name=self.target_col)

            y_lasso = y_lasso.loc[X_lasso.index]

        # ---------------------------------------------------
        # IMPORTANT:
        # Drop NaNs only inside Lasso fitting.
        # This does NOT affect the dataframe returned later.
        # ---------------------------------------------------
        valid_idx = X_lasso.dropna().index.intersection(y_lasso.dropna().index)

        X_lasso_fit = X_lasso.loc[valid_idx]
        y_lasso_fit = y_lasso.loc[valid_idx]

        if len(X_lasso_fit) <= self.n_splits:
            raise ValueError(
                f"Not enough rows for TimeSeriesSplit. "
                f"Rows after internal NaN removal: {len(X_lasso_fit)}, "
                f"n_splits: {self.n_splits}."
            )

        # Internal Lasso pipeline
        ts_cv = TimeSeriesSplit(n_splits=self.n_splits)

        lasso_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lasso", Lasso(max_iter=self.max_iter, tol=self.tol)),
            ]
        )

        param_grid = {
            "lasso__alpha": self.alphas_
        }

        grid_search = GridSearchCV(
            estimator=lasso_pipeline,
            param_grid=param_grid,
            cv=ts_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )

        grid_search.fit(X_lasso_fit, y_lasso_fit)

        # Store best Lasso info
        self.best_params_ = grid_search.best_params_
        self.best_alpha_ = grid_search.best_estimator_.named_steps["lasso"].alpha

        best_lasso = grid_search.best_estimator_.named_steps["lasso"]

        summary_df = pd.DataFrame(
            {
                "feature": self.candidate_lag_features_,
                "coefficient": best_lasso.coef_,
                "abs_coefficient": np.abs(best_lasso.coef_),
            }
        )

        selected_summary_df = (
            summary_df[summary_df["coefficient"] != 0]
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )

        # Fallback if Lasso selects zero features
        if len(selected_summary_df) == 0 and self.min_features > 0:
            selected_summary_df = (
                summary_df
                .sort_values("abs_coefficient", ascending=False)
                .head(self.min_features)
                .reset_index(drop=True)
            )

        self.lasso_selected_lag_features_ = selected_summary_df["feature"].tolist()

        # ---------------------------------------------------
        # Force-keep specific ACF/PACF lags by variable or other user defined variable
        # Example:
        # {
        #     "BORE_OIL_VOL_STB_D": [1, 2, 6, 7],
        #     "GOR_MSCF_STB": [1, 7],
        # }
        # ---------------------------------------------------
        self.force_keep_lag_features_ = []
        self.missing_force_keep_lag_features_ = []

        for var, lags in self.force_keep_lag_map_.items():
            for lag in lags:
                lag_col = f"{var}_lag_{lag}D"

                if lag_col in X_with_lags.columns and lag_col not in self.columns_to_drop_:
                    self.force_keep_lag_features_.append(lag_col)
                else:
                    self.missing_force_keep_lag_features_.append(lag_col)

        # Final selected lags = force-kept lags + Lasso-selected lags
        self.selected_lag_features_ = (
            self.force_keep_lag_features_
            + self.lasso_selected_lag_features_
        )

        # Remove duplicates while preserving order
        self.selected_lag_features_ = list(dict.fromkeys(self.selected_lag_features_))

        # Final output = original dataframe + selected lag features
        self.final_columns_ = self.original_columns_ + self.selected_lag_features_
        self.final_columns_ = list(dict.fromkeys(self.final_columns_))

        # Summary of final selected lag features
        self.feature_summary_df_ = summary_df[
            summary_df["feature"].isin(self.selected_lag_features_)
        ].copy()

        self.feature_summary_df_ = (
            self.feature_summary_df_
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )

        self.is_fitted_ = True

        if self.verbose:
            print(f"LassoSelectedLagFeatureAdder__Original columns after GOR/WCUT: {len(self.original_columns_)}")
            print(f"LassoSelectedLagFeatureAdder__All lag features created: {len(all_lag_features)}")
            print(f"LassoSelectedLagFeatureAdder__Lag features removed from Lasso candidates: {len(self.columns_to_drop_)}")
            print(f"LassoSelectedLagFeatureAdder__Candidate lag features used by Lasso: {len(self.candidate_lag_features_)}")
            print(f"LassoSelectedLagFeatureAdder__Rows used internally for Lasso fit: {len(X_lasso_fit)}")
            print(f"LassoSelectedLagFeatureAdder__Best alpha: {self.best_alpha_}")
            print(f"LassoSelectedLagFeatureAdder__Lasso-selected lag features: {len(self.lasso_selected_lag_features_)}")
            print(f"LassoSelectedLagFeatureAdder__Force-kept lag features: {len(self.force_keep_lag_features_)}")
            print(f"LassoSelectedLagFeatureAdder__Final selected lag features: {len(self.selected_lag_features_)}")
            print(f"LassoSelectedLagFeatureAdder__Final dataframe columns returned: {len(self.final_columns_)}")

            if len(self.missing_force_keep_lag_features_) > 0:
                print("Missing force-keep lag features:")
                print(self.missing_force_keep_lag_features_)

        return self

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        X = X.copy()

        # Create lags
        X_with_lags = self.lag_transformer_.transform(X)

        missing_selected_features = [
            col for col in self.selected_lag_features_
            if col not in X_with_lags.columns
        ]

        if missing_selected_features:
            raise ValueError(
                f"Missing selected lag features in transform data: "
                f"{missing_selected_features}"
            )

        available_final_columns = [
            col for col in self.final_columns_
            if col in X_with_lags.columns
        ]

        X_out = X_with_lags[available_final_columns].copy()

        # ---------------------------------------------------
        # IMPORTANT:
        # Do NOT drop NaNs here.
        # A later final transformer should drop NaNs once.
        # ---------------------------------------------------
        return X_out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "is_fitted_")
        return np.array(self.final_columns_)
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
class AddSelectedCalendarFourierFeatures(BaseEstimator, TransformerMixin):
    """
    Creates calendar Fourier features, uses Lasso internally to select useful
    Fourier features, and appends only the selected Fourier columns to the dataframe.

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
    - y is not transformed.
    - Lasso is used only to select Fourier columns.
    - transform() returns original dataframe + selected Fourier features.
    - Does not drop rows.
    """

    def __init__(
        self,
        target_col="BORE_OIL_VOL_STB_D",
        param_grid=None,
        date_col=None,
        keep_original=True,
        drop_date_col=False,
        min_features=1,
        n_splits=5,
        max_iter=800000,
        tol=1e-5,
        scoring="neg_root_mean_squared_error",
        verbose=True,
        feature_prefix=None,
    ):
        self.target_col = target_col
        self.param_grid = param_grid
        self.date_col = date_col
        self.keep_original = keep_original
        self.drop_date_col = drop_date_col
        self.min_features = min_features
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.tol = tol
        self.scoring = scoring
        self.verbose = verbose
        self.feature_prefix = feature_prefix

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        X = X.copy()
        self.original_columns_ = X.columns.tolist()

        if y is None:
            if self.target_col not in X.columns:
                raise ValueError(
                    f"target_col='{self.target_col}' is not in X, and y was not provided."
                )
            y_fit = X[self.target_col].copy()
        else:
            if isinstance(y, pd.Series):
                y_fit = y.copy()
            else:
                y_fit = pd.Series(y, index=X.index, name=self.target_col)

        if self.param_grid is None:
            alpha_grid = [
                0.001, 0.01, 0.1, 1.0, 10.0,
                20, 30, 50, 100,
            ]

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

            valid_idx = X_fourier.dropna().index.intersection(y_fit.dropna().index)

            X_fourier_fit = X_fourier.loc[valid_idx]
            y_lasso_fit = y_fit.loc[valid_idx]

            if len(X_fourier_fit) <= self.n_splits:
                continue

            internal_pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "lasso",
                        Lasso(
                            alpha=alpha,
                            max_iter=self.max_iter,
                            tol=self.tol,
                        ),
                    ),
                ]
            )

            ts_cv = TimeSeriesSplit(n_splits=self.n_splits)

            scores = cross_val_score(
                internal_pipe,
                X_fourier_fit,
                y_lasso_fit,
                cv=ts_cv,
                scoring=self.scoring,
                n_jobs=-1,
            )

            mean_score = scores.mean()

            internal_pipe.fit(X_fourier_fit, y_lasso_fit)

            if mean_score > best_score:
                best_score = mean_score

                best_result = {
                    "seasonality": seasonality,
                    "K": K,
                    "alpha": alpha,
                    "score": mean_score,
                    "best_estimator": internal_pipe,
                    "feature_names": X_fourier.columns.tolist(),
                }

        if best_result is None:
            raise ValueError(
                "No valid Fourier model was fitted. "
                "Check your data length, n_splits, and date index."
            )

        self.best_seasonality_ = best_result["seasonality"]
        self.best_K_ = best_result["K"]
        self.best_alpha_ = best_result["alpha"]
        self.best_score_ = best_result["score"]
        self.best_estimator_ = best_result["best_estimator"]

        self.best_params_ = {
            "fourier__seasonality": self.best_seasonality_,
            "fourier__K": self.best_K_,
            "model__alpha": self.best_alpha_,
        }

        self.candidate_fourier_features_ = best_result["feature_names"]

        best_lasso = self.best_estimator_.named_steps["lasso"]

        summary_df = pd.DataFrame(
            {
                "feature": self.candidate_fourier_features_,
                "coefficient": best_lasso.coef_,
                "abs_coefficient": np.abs(best_lasso.coef_),
            }
        )

        selected_summary_df = (
            summary_df[summary_df["coefficient"] != 0]
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )

        if len(selected_summary_df) == 0 and self.min_features > 0:
            selected_summary_df = (
                summary_df
                .sort_values("abs_coefficient", ascending=False)
                .head(self.min_features)
                .reset_index(drop=True)
            )

        self.feature_summary_df_ = selected_summary_df
        self.selected_fourier_features_ = self.feature_summary_df_["feature"].tolist()

        if self.keep_original:
            self.final_columns_ = self.original_columns_ + self.selected_fourier_features_

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
            print(f"Best seasonality: {self.best_seasonality_}")
            print(f"Best K: {self.best_K_}")
            print(f"Best alpha: {self.best_alpha_}")
            print(f"Best CV score: {self.best_score_}")
            print(f"Candidate Fourier features: {len(self.candidate_fourier_features_)}")
            print(f"Selected Fourier features: {len(self.selected_fourier_features_)}")
            print(f"Final dataframe columns returned: {len(self.final_columns_)}")

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

        selected_fourier_df = X_fourier[self.selected_fourier_features_].copy()

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
            sin_col = f"{prefix}_sin_{k}"
            cos_col = f"{prefix}_cos_{k}"
    
            if self.feature_prefix is not None:
                sin_col = f"{self.feature_prefix}_{sin_col}"
                cos_col = f"{self.feature_prefix}_{cos_col}"
    
            fourier_df[sin_col] = np.sin(
                2 * np.pi * k * t / period
            )
    
            fourier_df[cos_col] = np.cos(
                2 * np.pi * k * t / period
            )
    
        return fourier_df

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "is_fitted_")
        return np.array(self.final_columns_)
#---------------------------------------------------------------------------------------------------------

class RollingSlopeFeatures(BaseEstimator, TransformerMixin):
    """
    Adds rolling linear slope features for selected variables.

    For each date t, the slope is shifted so that the model only sees
    information available before t. This helps avoid leakage.

    Example output:
        BORE_WAT_VOL_STB_D_slope_7D
        BORE_WAT_VOL_STB_D_slope_14D
        WCUT_slope_7D
    """

    def __init__(
        self,
        variables,
        windows=(7, 14, 30),
        min_periods=None,
        periods=1,
        drop_original=False,
        sort_index=True,
    ):
        self.variables = variables
        self.windows = windows
        self.min_periods = min_periods
        self.periods = periods
        self.drop_original = drop_original
        self.sort_index = sort_index

    def fit(self, X, y=None):
        X = X.copy()

        missing = [col for col in self.variables if col not in X.columns]
        if missing:
            raise ValueError(f"These variables are missing from X: {missing}")

        self.variables_ = list(self.variables)
        self.feature_names_in_ = list(X.columns)

        return self

    @staticmethod
    def _slope(values):
        values = np.asarray(values, dtype=float)

        if len(values) < 2:
            return np.nan

        if not np.isfinite(values).all():
            return np.nan

        x = np.arange(len(values))

        return np.polyfit(x, values, 1)[0]

    def transform(self, X):
        X = X.copy()

        if self.sort_index:
            X = X.sort_index()

        new_cols = []

        for col in self.variables_:
            for window in self.windows:
                min_periods = self.min_periods if self.min_periods is not None else window

                new_col = f"{col}_slope_{window}D"

                X[new_col] = (
                    X[col]
                    .rolling(window=window, min_periods=min_periods)
                    .apply(self._slope, raw=True)
                    .shift(self.periods)
                )

                new_cols.append(new_col)

        if self.drop_original:
            X = X.drop(columns=self.variables_)

        return X
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