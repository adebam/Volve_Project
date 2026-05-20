from lmfit import Parameters, minimize, report_fit
import numpy as np
import pandas as pd

##############################
#Helper functions             #
##############################
def get_arps_initial_guess(t, q, n_early=360):
    """
    This is an helper function to get initial guesses for the decline curve analysis
    t is time (days on), array
    q  production, for this dataset stb/day or mscf/day, array
    n_early: the initial amount of days to use to calculate the initial guess for the Di, integer
    returns inital guesses for qi, Di(1/day), and b
    """
    t = np.asarray(t)
    q = np.asarray(q)

    mask = q > 0
    t = t[mask]
    q = q[mask]

    order = np.argsort(t)
    t = t[order]
    q = q[order]

    n = min(n_early, len(q))

    qi_guess = np.percentile(q[:n], 90)

    try:
        slope = np.polyfit(t[:n], np.log(q[:n]), 1)[0]   #Fits a straight line
        Di_guess = max(-slope, 1e-6)
    except:
        Di_guess = 0.001

    b_guess = 0.5

    return [qi_guess, Di_guess, b_guess]



def sort_remove_nan(df, rate):
    """
    This is an helper function to sort the dataframe and make sure no value less then zero is selected
    df: dataframe having time as in days_on prod and rate. rate could be  BORE_OIL_VOL_STB	BORE_GAS_VOL_MSCF	BORE_WAT_VOL_STB
    returns the cleaned dataframe
    """
    df = df.sort_values("Days_on_prod")
    df = df[df[rate] > 0]
    return df
    
##############################
# Exponential Decline Curve  #
##############################
def Exponential(t, qi, Di):
    """
    Exponential decline curve analysis
    t is time (days on), array
    qi initial production, array
    Di initial decline rate (per unit time this case 1/day), float
    returns qmodel at every time step, array
    """

    return  qi*np.exp(-Di*t)

def Exponential_residuals_log_lmfit(params, t, q):
    """
    Loss function. returns the difference between the actual data and the model
    params: object like array to take parameters to be optimized
    t is time (days on), array
    q production, array
    """
    qi = params["qi"].value
    Di = params["Di"].value
    
    q_model = Exponential(t, qi, Di)
    
    q_safe = np.maximum(q, 1e-6)
    q_model_safe = np.maximum(q_model, 1e-6)
    
    return np.log(q_model_safe) - np.log(q_safe)

def Exponential_minimizer(t,q):
    """
    This function calls the lmfit optimizer. 
    df: dataframe
    rate: production column name
    returns the fitted qi and Di
    """
    # clean dataframe
    #df = sort_remove_nan(df, rate)
    # extract arrays
    #t = np.asarray(t, dtype=float)
    #q = np.asarray(q, dtype=float)
    
    t = t.to_numpy(dtype=float)
    q = q.to_numpy(dtype=float)
    
    params_exponential = Parameters()      # parameters for the exponential distribution
    
    x0 = get_arps_initial_guess(t, q)    # get intial guesses
    
    params_exponential.add("qi", value=x0[0], min=0)
    params_exponential.add("Di", value=x0[1], min=0, max=1)
    
    result = minimize(
        Exponential_residuals_log_lmfit,
        params_exponential,
        args=(t, q),
        method="least_squares",
        loss="soft_l1"
        )
    
    #report_fit(result)
    
    qi_fit = result.params["qi"].value
    Di_fit = result.params["Di"].value
    return(qi_fit, Di_fit)


##############################
# Hyperbolic Decline Curve    #
##############################

def Hyperbolic(t, qi, Di,b):
    """
    hyperbolic decline curve analysis
    t is time (days on), array
    qi initial production, array
    Di initial decline rate (per unit time this case 1/day), float
    b decline curve exponent
    returns qmodel at every time step, array
    """
    t = np.asarray(t)
    

    return  qi/(1+b*Di*t)**(1/b)

def Hyperbolic_residuals_log_lmfit(params, t, q):
    """
    Loss function. returns the difference between the actual data and the model
    params: object like array to take parameters to be optimized
    t is time (days on), array
    q production, array
    """
    qi = params["qi"].value
    Di = params["Di"].value
    b = params["b"].value
    
    q_model = Hyperbolic(t, qi, Di,b)
    
    q_safe = np.maximum(q, 1e-6)
    q_model_safe = np.maximum(q_model, 1e-6)
    
    return np.log(q_model_safe) - np.log(q_safe)

def Hyperbolic_minimizer(t,q):
    """
    This function calls the lmfit optimizer. 
    t: time arrays in days
    q: rate array in stb/day or mscf/day
    returns the fitted qi and Di and b
    """
    # clean dataframe
    #df = sort_remove_nan(df, rate)
    # extract arrays
    t = t.to_numpy(dtype=float)
    q = q.to_numpy(dtype=float)
    
    #t = np.asarray(t, dtype=float)
    #q = np.asarray(q, dtype=float)
    
    params_Hyperbolic = Parameters()      # parameters for the hyperbolic distribution
    
    x0 = get_arps_initial_guess(t, q)    # get intial guesses
    
    params_Hyperbolic.add("qi", value=x0[0], min=0)
    params_Hyperbolic.add("Di", value=x0[1], min=0, max=1)
    params_Hyperbolic.add("b",  value=x0[2], min=0.1, max=0.9999999)
    
    result = minimize(
        Hyperbolic_residuals_log_lmfit,
        params_Hyperbolic,
        args=(t, q),
        method="least_squares",
        loss="soft_l1"
        )
    
    report_fit(result)
    
    qi_fit = result.params["qi"].value
    Di_fit = result.params["Di"].value
    b_fit = result.params["b"].value
    
    return(qi_fit, Di_fit,b_fit)
##############################
# Harmonic Decline Curve    #
##############################

def Harmonic(t, qi, Di,b=1):
    """
    Harmonic decline curve analysis
    t is time (days on), array
    qi initial production, array
    Di initial decline rate (per unit time this case 1/day), float
    returns qmodel at every time step, array
    """

    return  qi/(1+b*Di*t)**(1/b)

def Harmonic_residuals_log_lmfit(params, t, q):
    """
    Loss function. returns the difference between the actual data and the model
    params: object like array to take parameters to be optimized
    t is time (days on), array
    q production, array
    """
    qi = params["qi"].value
    Di = params["Di"].value
    
    q_model = Harmonic(t, qi, Di)
    
    q_safe = np.maximum(q, 1e-6)
    q_model_safe = np.maximum(q_model, 1e-6)
    
    return np.log(q_model_safe) - np.log(q_safe)

def Harmonic_minimizer(t,q):
    """
    This function calls the lmfit optimizer. 
    t: time arrays in days
    q: rate array in stb/day or mscf/day
    returns the fitted qi and Di and b
    """
    t = t.to_numpy(dtype=float)
    q = q.to_numpy(dtype=float)
    
    params_Harmonic = Parameters()      # parameters for the exponential distribution
    
    x0 = get_arps_initial_guess(t, q)    # get intial guesses
    
    params_Harmonic.add("qi", value=x0[0], min=0)
    params_Harmonic.add("Di", value=x0[1], min=0, max=1)
    
    result = minimize(
        Harmonic_residuals_log_lmfit,
        params_Harmonic,
        args=(t, q),
        method="least_squares",
        loss="soft_l1"
        )
    
    report_fit(result)
    
    qi_fit = result.params["qi"].value
    Di_fit = result.params["Di"].value
    return(qi_fit, Di_fit)


##############################
# Date  helper #
##############################
def forecast_dates(df):
    """
    The function takes a dataframe  calculate the last date of production as well as the last days on
    Calculates the calendar days
    Calculates the time production starts, t_forecast
    returns the fture calendar days as well as the future forecast dates as an array
    The maximum prediction days is 15 years this includes the time the well has been on production
    """
    
    total_predict_days=5475 # 15 years
    last_date = df["DATEPRD"].max()
    last_days_on = df["Days_on_prod"].max()
    
    calendar_days = (df["DATEPRD"].max()-df["DATEPRD"].min()).days # days on production including when the well is down
    calendar_days = max(calendar_days, 1)
    # forecast producing days
    uptime_fraction = last_days_on / calendar_days
    remaining_calendar_days = max(total_predict_days - calendar_days, 0)
    t_forecast = np.arange(last_days_on,last_days_on + remaining_calendar_days * uptime_fraction,30)  # I want to start the forecast after (5475-normalized_days) 5475 is 15 years
    
    # estimate uptime
    uptime_fraction = (last_days_on /calendar_days)
    
    # convert to calendar days
    future_calendar_days = (t_forecast - last_days_on) / uptime_fraction
    
    # forecast dates
    forecast_dates = (last_date+pd.to_timedelta(future_calendar_days, unit="D"))

    return (future_calendar_days, forecast_dates)