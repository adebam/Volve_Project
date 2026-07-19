# Volve Reservoir Virtual Meter

The Volve oil field is located in Block 15/9 of the central Norwegian North Sea. It is situated approximately 200 kilometers west of Stavanger, Norway,and in a water depth of around 80 meters. The Volve field produced oil and gas from 2008 to 2016 and is now decommissioned, but it remains as one of the most open-source subsurface and production datasets in the energy industry. The aim of this project is to come up with different ways to visualize the production data and perform decline curve analysis. 

<figure>
<img src="The-geographic-location-of-the-Volve-field.png" width="400" height="300">
<figcaption>Source: <a href="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT7ex3O5HJRM-omJ3P_GW9B6RS3xKcMWQPhJA&s">Picture of Volve field</a></figcaption>
</figure>


## Project Motivation
Accurate measurement of oil, gas, and water production is essential for production optimization, reservoir management, production allocation, and well-performance surveillance. However, installing dedicated multiphase flow meters on every well is often economically impractical, particularly in fields containing many producing wells.

In current operations, individual well production may be estimated using periodic separator tests, shared test facilities, and flowback-allocation methods. Measured facility-level production is allocated back to individual wells using the most recent well-test results, historical production ratios, operating conditions, or engineering assumptions. Although this approach is widely used, the allocated rates may become less representative between well tests, especially when well performance, choke settings, pressures, water cut, gas–oil ratio, or operating status change.

This project investigates the development of a machine-learning-based virtual flow meter that uses routinely available well measurements, including wellhead pressure, choke size, temperature, on-stream hours, and historical production data, to estimate daily oil, gas, and water rates.

The proposed system is intended to complement the existing flowback-allocation process by providing more frequent, well-specific phase-rate estimates between physical well tests. This could improve production-allocation accuracy, reduce reliance on frequent separator testing, identify changes in well behavior earlier, and support faster production-management decisions.

The project also evaluates the model’s ability to forecast future production, recognize changing operating regimes, and maintain reliable performance as well and reservoir conditions evolve.

## Executive Summary




## Modules

- DF__1-15-9-F-1_C_Recursive.ipynb: Recurvsive Model for Well 15-9-F-1_C
- DF__2-15_9-F-11_Recursive.ipynb: Recursive Model for Well 15_9-F-11
- DF__2-15_9-F-11_Direct.ipynb: Direct Moodel for Well 15_9-F-11
- DF__3-15-9-F-12_Recursive.ipynb: Recursive Moodel for Well 15_9-F-12
- DF__3-15-9-F-12_Direct.ipynb: Direct Model for Well 15-9-F-12
- DF__4-15-9-F-14_Recursive.ipynb: Recursive Moodel for Well 15_9-F-14
- DF__4-15-9-F-14_Direct.ipynb: Direct Model for Well 15-9-F-14
- 1_Load_split_export_make_plots: Testing functions to split input data into individual wells and make time plots
- 2_Plot_Clean_dfs.ipynb: Testing function to clean data

## Project Structure
```text
├── assets
│   └── style.css
├── new-workspace.jupyterlab-workspace
├── notebooks
│   ├── 1_Load_Clean_Production_data.ipynb
│   ├── 2_Find_Load_plot_X_Y_loc.ipynb
│   ├── 3_Grid_properties.ipynb
│   ├── 4_decline_curve_analysis.ipynb
│   ├── Dash_Decline_Curve.ipynb
│   ├── Dash_Production.ipynb
│   ├── Dash_Well_Comparison.ipynb
│   └── temp-plot.html
├── README.md
├── src
│   ├── Decline_curve.py
│   ├── Find_Load_plot_X_Y_loc.py
│   ├── Load_clean_production_data.py
│   └── Load_Petrophysical_data.py
└── The-geographic-location-of-the-Volve-field.png
```

## Overall Workflow
### Why Time-Series Analysis Diffent than other Machine Learning Applications

Time-series analysis is essential in virtual flow metering because oil, gas, and water production measurements are collected sequentially over time. In addition, the oil, gas and water are produced at the sametime A well’s current production rate is influenced by its previous production, pressure history, choke settings, operating hours, shutdowns, reservoir depletion, and other earlier operating conditions and reservoir conditions. Therefore, the order in which the observations occur contains valuable information that should not be ignored.

In many traditional machine-learning applications, observations are assumed to be largely independent and identically distributed. For example, when predicting house prices, the order in which houses appear in the dataset usually has no physical meaning. The data can often be randomly shuffled before being divided into training and testing sets.

Time-series data are different because observations are temporally dependent. Randomly shuffling production data would allow the model to learn from future well behavior when predicting the past, creating data leakage and producing unrealistically optimistic performance. Training, validation, and testing must therefore follow chronological order so that the model is evaluated under conditions that resemble real deployment.

Time-series analysis is also important because production data commonly contain several time-dependent patterns:

* **Trend:** Long-term increases or decreases caused by reservoir depletion, well cleanup, artificial-lift changes, or water breakthrough.
* **Seasonality or cyclic behavior:** Repeated patterns related to operating schedules, testing frequency, facility constraints, or periodic interventions.
* **Autocorrelation:** Current production is often strongly related to production during previous days.
* **Regime changes:** Shut-ins, choke changes, workovers, startup periods, unstable flow, and water breakthrough can cause the well to behave differently over time.
* **Nonstationarity:** The statistical distribution of production may change as the well and reservoir evolve.

These characteristics require specialized feature engineering, such as lagged production values, rolling averages, rolling slopes, cumulative production, time-since-start variables, and operating-regime indicators.

Time-series forecasting also introduces an additional challenge: the availability of future input variables. In a normal supervised-learning problem, all predictor variables are usually available when a prediction is made. In forecasting, future values of variables such as wellhead pressure, choke size, and on-stream hours may not yet be known. The model must therefore use historical values, planned operating conditions, separately forecasted variables, or recursive predictions.

Another important difference is how errors propagate. In a one-day-ahead prediction, the model can use the most recently measured production data. In a recursive 30-day forecast, the prediction for the first day may be used as an input for the second day, and subsequent predictions continue to depend on earlier predicted values. Small errors can therefore accumulate and cause the forecast to drift over longer horizons.

For virtual metering, time-series analysis makes it possible to:

1. Capture the natural production decline and changing behavior of the well.
2. Use recent production and operating history to improve phase-rate estimates.
3. Predict oil, gas, and water rates between separator tests.
4. Detect unusual behavior or operating conditions not represented in the historical data.
5. Evaluate the model realistically using chronological validation.
6. Update the model as new separator measurements and sensor data become available.

Consequently, virtual metering should not be treated as a conventional regression problem in which rows can be randomly shuffled. It is a dynamic forecasting problem in which time, operational history, data availability, and changes in well behavior must be explicitly considered.

### Model Type Architecture 
#### One Step (Day) Look Ahead
A one-step look-ahead model predicts the production rate for the next day using all information available up to the current day.

For example, at the end of Day 13, the model uses production history, sensor measurements, and operating conditions observed through Day 13 to predict the oil, gas, and water rates for Day 14.For day 15 prediction, the actual rates of day 14 is no avaliable, and the predicted rate for day 14 is now replaced by the actual real rates of day 14.

```mermaid
flowchart LR
    D10["Day 10<br/>Available data"]
    D11["Day 11<br/>Prediction target"]
    D12["Day 12<br/>Prediction target"]
    D13["Day 13<br/>Prediction target"]

    D10 -->|Predict| D11
    D11 -->|Predict| D12
    D12 -->|Predict| D13
```
This is commonly called walk-forward one-step forecasting:

1. Predict the next day.
2. Observe the actual next-day value.
3. Add the actual value to the available history.
4. Recalculate the lagged and rolling features.
5. Predict the following day.

A variable can only be used for Day (t+1) if its value is genuinely available when the forecast is generated.

For example, at the end of Day 13:

* Day 13 WHP can be used to predict Day 14;
* Day 13 choke size can be used;
* a planned Day 14 choke setting can be used if it is known in advance.
* the actual Day 14 WHP cannot be used because it has not yet been measured;
* the actual Day 14 oil rate cannot be used because it is the prediction target.

$$
\hat{y}_{t+1} = f\left( y_t,\, \text{WHP}_t,\, \text{choke}_t,\, \text{on-stream hours}_t \right)
$$

It should not use future information that is unavailable when the prediction is made:

$$
\hat{y}_{t+1} \neq f\left( \text{actual WHP}_{t+1},\, \text{actual oil rate}_{t+1} \right)
$$

One-step forecasting is generally easier than forecasting several days ahead because the model always receives the latest actual production measurements.

For example:

    Day 13 actual data → predict Day 14
    Day 14 actual data → predict Day 15
    Day 15 actual data → predict Day 16

The model does not have to use its Day 14 prediction as an input for predicting Day 15. This reduces cumulative forecast error and drift. This different than the recursive. In comparison, a recursive multi-day forecast may operate as follows:

    Day 13 actual data       → predict Day 14
    Day 14 predicted value   → predict Day 15
    Day 15 predicted value   → predict Day 16

Errors from earlier predictions can therefore propagate through the forecast.

In virtual meters, the one step look ahead model architecture is not really useful, because we are going to need the actual separator measurements every  other everyday.

However, when separator measurements are only obtained every 30 days, a true daily one-step forecasting system cannot continually use the actual previous-day phase rates. After the first forecasted day, it must either:

* use its own predicted phase rates recursively;
* use a direct multi-horizon model;


Consequently, one-step forecasting provides a strong benchmark for evaluating whether the latest well history contains enough information to predict the immediate next-day production behavior.

#### Recursive Model
For recursive forecasting, each new prediction is fed back into the model to generate the following prediction. Since the error might propagate from day to day, the model will need to be updated from real data at a particular interval.
## Recursive Multi-Day Forecasting

For the first forecast day, the model uses the latest available actual production values and operating measurements:

```markdown
$$
\hat{y}_{t+1}
=
f\left(
y_t,\,
y_{t-1},\,
\mathrm{WHP}_t,\,
\mathrm{choke}_t,\,
\mathrm{on\text{-}stream\ hours}_t
\right)
$$
```

For the second forecast day, the actual production rate at (t+1) is not yet available. Therefore, the model feeds its Day 1 prediction back into the feature set:

```markdown
$$
\hat{y}_{t+2}
=
f\left(
\hat{y}_{t+1},\,
y_t,\,
\widehat{\mathrm{WHP}}_{t+1},\,
\mathrm{choke}_{t+1}^{\mathrm{planned}},\,
\widehat{\mathrm{on\text{-}stream\ hours}}_{t+1}
\right)
$$
```

For the third forecast day, the model again uses its previous predictions:

```markdown
$$
\hat{y}_{t+3}
=
f\left(
\hat{y}_{t+2},\,
\hat{y}_{t+1},\,
\widehat{\mathrm{WHP}}_{t+2},\,
\mathrm{choke}_{t+2}^{\mathrm{planned}},\,
\widehat{\mathrm{on\text{-}stream\ hours}}_{t+2}
\right)
$$
```

More generally, the recursive forecast at horizon (h) can be represented as:

```markdown
$$
\hat{y}_{t+h}
=
f\left(
\hat{y}_{t+h-1},\,
\hat{y}_{t+h-2},\,
\widehat{X}_{t+h-1}
\right)
$$
```

where:

```markdown
$$
\widehat{X}_{t+h-1}
=
\left(
\widehat{\mathrm{WHP}}_{t+h-1},\,
\mathrm{choke}_{t+h-1}^{\mathrm{planned}},\,
\widehat{\mathrm{on\text{-}stream\ hours}}_{t+h-1}
\right)
$$
```

The vector (\widehat{X}_{t+h-1}) contains operating variables that are either:

* known in advance;
* based on an operational plan;
* held constant using the most recent observation; or
* predicted separately before generating the production forecast.

The recursive forecasting process is therefore:

```text
Actual history through time t
            |
            v
Predict y(t+1)
            |
            v
Use predicted y(t+1) to predict y(t+2)
            |
            v
Use predicted y(t+1) and y(t+2) to predict y(t+3)
            |
            v
Continue until the required forecast horizon is reached
```

The model must not use actual future production or sensor measurements that would be unavailable when the forecast is generated:

```markdown
$$
\hat{y}_{t+h}
\neq
f\left(
y_{t+h-1}^{\mathrm{actual}},\,
\mathrm{WHP}_{t+h-1}^{\mathrm{actual}}
\right)
$$
```

Instead, it must use previous predictions and future-known, planned, assumed, or separately forecasted operating conditions:

```markdown
$$
\hat{y}_{t+h}
=
f\left(
\hat{y}_{t+h-1},\,
\hat{y}_{t+h-2},\,
\widehat{X}_{t+h-1}
\right)
$$
```

Because each predicted production rate becomes an input for the next forecast step, errors can accumulate as the forecast horizon increases. This error propagation is one of the main limitations of recursive multi-day forecasting.



## Technologies

- Python
- Plotly Dash
- Pandas
- NumPy
- JupyterLab

## Installation
##### 1. Clone the Repository
```bash
git clone https://github.com/adebam/Volve_Project.git
cd 1_Visualization
```
##### 2. Install Dependencies
```bash
mamba env create -f environment.yml
```
Or using Conda:
```bash
conda env create -f environment.yml
```
##### 3. Activate the Environment
```bash
mamba activate volve_project_visualization
```

##### 4. Start JupyterLab
```bash
jupyter lab
```

## License
Copyright (c) 2026 Dayo Adebamiro

Permission is granted to use, copy, modify, and distribute this software
for personal, educational, and research purposes only.

Commercial use is prohibited without prior written permission.

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
