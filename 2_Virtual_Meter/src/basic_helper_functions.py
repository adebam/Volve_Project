import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle


def change_metric_to_field(production_df):
    # 1. Create a copy of the input 
    df_field = pd.DataFrame()
    
    # Helper list for columns that just need to be copied exactly as they are
    exact_copy_cols = [
        "DATEPRD", "WELL_BORE_CODE", "NPD_WELL_BORE_CODE", "NPD_WELL_BORE_NAME", 
        "NPD_FIELD_CODE", "NPD_FIELD_NAME", "NPD_FACILITY_CODE", "NPD_FACILITY_NAME", 
        "ON_STREAM_HRS", "AVG_CHOKE_SIZE_P", "AVG_CHOKE_UOM", 
         "FLOW_KIND", "WELL_TYPE"
    ]
    
    # 2. Safely copy exact columns if they exist
    for col in exact_copy_cols:
        if col in production_df.columns:
            df_field[col] = production_df[col]
            
    # 3. Safely apply unit conversions (only if the source column exists)
    if "AVG_DOWNHOLE_PRESSURE" in production_df.columns:
        df_field["AVG_DOWNHOLE_PRESSURE_PSI"] = production_df["AVG_DOWNHOLE_PRESSURE"] * 14.5038
        
    if "AVG_DOWNHOLE_TEMPERATURE" in production_df.columns:
        df_field["AVG_DOWNHOLE_TEMPERATURE_F"] = production_df["AVG_DOWNHOLE_TEMPERATURE"] * 9/5 + 32
        
    if "AVG_DP_TUBING" in production_df.columns:
        df_field["AVG_DP_TUBING_PSI"] = production_df["AVG_DP_TUBING"] * 14.5038
        
    if "AVG_ANNULUS_PRESS" in production_df.columns:
        df_field["AVG_ANNULUS_PRESS_PSI"] = production_df["AVG_ANNULUS_PRESS"] * 14.5038

    if "DP_CHOKE_SIZE" in production_df.columns:
        df_field["DP_CHOKE_SIZE_PSI"] = production_df["DP_CHOKE_SIZE"] * 14.5038

    if "AVG_WHP_P" in production_df.columns:
        df_field["AVG_WHP_P_PSI"] = production_df["AVG_WHP_P"] * 14.5038 

    if "BORE_WI_VOL" in production_df.columns:
        df_field["BORE_WI_VOL_STB_D"] = production_df["BORE_WI_VOL"] * 6.28981
        
    if "AVG_WHT_P" in production_df.columns:
        df_field["AVG_WHT_P"] = production_df["AVG_WHT_P"] * 9/5 + 32 
        
    if "BORE_OIL_VOL" in production_df.columns:
        df_field["BORE_OIL_VOL_STB_D"] = production_df["BORE_OIL_VOL"] * 6.28981
        
    if "BORE_GAS_VOL" in production_df.columns:
        df_field["BORE_GAS_VOL_MSCF_D"] = production_df["BORE_GAS_VOL"] * 35.3147 / 1000 
        
    if "BORE_WAT_VOL" in production_df.columns:
        df_field["BORE_WAT_VOL_STB_D"] = production_df["BORE_WAT_VOL"] * 6.28981

    df_field["GOR_MSCF_STB"]=df_field["BORE_GAS_VOL_MSCF_D"]/df_field["BORE_OIL_VOL_STB_D"]
    df_field["WCUT"]=df_field["BORE_WAT_VOL_STB_D"]/(df_field["BORE_OIL_VOL_STB_D"]+ df_field["BORE_WAT_VOL_STB_D"])

    return df_field

#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def combined_target_plot(df):
    '''
    This function creates  a combined plot for oil, gas and oil
    '''
    # Define my color mapping
    color_map = {
        'BORE_OIL_VOL_STB_D': 'darkgreen',
        'BORE_GAS_VOL_MSCF_D': 'crimson',
        'BORE_WAT_VOL_STB_D': 'blue'
    }
    
    plt.figure(figsize=(18, 9))
    
    for col in df.columns:
        # Use .get() so it falls back to 'black' if a column isn't in my map
        col_color = color_map.get(col, 'black') 
        
        plt.plot(df.index, df[col], label=col, color=col_color)
    
    plt.title('Well Production Volumes Over Time (Combined View)')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def seperate_target_plot(df):
    '''
    This function creates  a seperate plot for oil, gas and oil
    '''
    # Define my color mapping
    color_map = {
        'BORE_OIL_VOL_STB_D': 'darkgreen',
        'BORE_GAS_VOL_MSCF_D': 'crimson',
        'BORE_WAT_VOL_STB_D': 'blue'
    }
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 6), sharex=True)
    
    # Loop through columns and axes to plot them individually
    for i, col in enumerate(df.columns):
        #Get the assigned color, defaulting to black if it's missing from the map
        col_color = color_map.get(col, 'black')
        
        # Apply the custom color
        axes[i].plot(df.index, df[col], color=col_color, label=col)
        
        #axes[i].set_ylabel(col, fontsize=9)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].legend(loc='upper right')
    
    # Set the shared x-axis label on the bottom plot
    axes[-1].set_xlabel('Date')
    fig.suptitle('Individual Target Volume Time-Series', y=0.98)
    
    # Adjust layout so labels don't overlap
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def combined_predictor_plot(df, features=None):
    '''
    Creates a combined plot for specified predictors over time.
    
    Parameters:
    - df: The DataFrame (assumed to have a DatetimeIndex).
    - features: List of column names to plot. If None, it automatically
                plots all numeric columns available in the color map.
    '''
    # Define my color mapping
    color_map = {
        'ON_STREAM_HRS': "#7d7575",
        'AVG_CHOKE_SIZE_P': "#7d7733",
        'DP_CHOKE_SIZE_PSI': "#b3d411", 
        'AVG_DOWNHOLE_PRESSURE_PSI': "#22330c", 
        'AVG_DOWNHOLE_TEMPERATURE_F': "#ab7432",
        'AVG_DP_TUBING_PSI': "#470b0b", 
        'AVG_ANNULUS_PRESS_PSI': "#0b4047", 
        'AVG_WHP_P_PSI': "#360996",
        'BORE_WI_VOL_STB_D': "#02d9f5", 
        'AVG_WHT_P': "#fa0505"
    }
    # 2. Define an infinite cycle of backup colors
    

    
    # 1. Handle dynamic feature selection
    if features is None:
        # Automatically find columns that are both in the DataFrame and in my color map
        # This keeps it safe from text/ID columns crashing the plot
        features = [col for col in df.columns if col in color_map]
    else:
        # If user passed a list or a single string, ensure it matches actual columns
        if isinstance(features, str):
            features = [features]
        features = [col for col in features if col in df.columns]

    # If no valid features are found, alert the user and exit safely
    if not features:
        print("Warning: No matching numeric features found to plot.")
        return

# If a column isn't in my map, it will grab the next color in this list
    fallback_colors = cycle(['grey', 'dimgrey', 'darkgrey', 'silver', 'black'])
    
    plt.figure(figsize=(18, 9))
    
    for col in features:
        if col in color_map:
            col_color = color_map[col]
        else:
            # If the column isn't in my map, grab the next available fallback color
            col_color = next(fallback_colors)
        
        plt.plot(df.index, df[col], label=col, color=col_color)
    
    plt.title('Well Production Predictors Over Time (Combined View)')
    plt.xlabel('Date')
    #plt.ylabel('Value / Metric Units')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------------#

from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd

from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd

def seperate_predictor_plot(df, features=None):
    '''
    Creates separate subplots stacked vertically for specified predictors over time.
    Parameters:
    - df: The DataFrame (assumed to have a DatetimeIndex).
    - features: List of column names to plot. If None, it automatically plots all numeric columns available in the color map.
    '''
    # Define my specific color mapping
    color_map = {
        'ON_STREAM_HRS': "#7d7575",
        'AVG_CHOKE_SIZE_P': "#7d7733",
        'DP_CHOKE_SIZE_PSI': "#b3d411",
        'AVG_DOWNHOLE_PRESSURE_PSI': "#22330c",
        'AVG_DOWNHOLE_TEMPERATURE_F': "#ab7432",
        'AVG_DP_TUBING_PSI': "#470b0b",
        'AVG_ANNULUS_PRESS_PSI': "#0b4047",
        'AVG_WHP_P_PSI': "#360996",
        'BORE_WI_VOL_STB_D': "#02d9f5",
        'AVG_WHT_P': "#fa0505",
        'GOR_MSCF_STB': "#9400d3",  # Dark Violet/Purple for Gas-Oil Ratio
        'WCUT': "#008080"            # Teal for Water Cut
    }
    
    # 1. Define an infinite cycle with my exact custom backup colors
    fallback_colors = cycle(['grey', 'dimgrey', 'darkgrey', 'black'])
    
    # 2. Handle dynamic feature selection properly
    if features is None:
        # Default behavior: pick columns that match the colormap and exist in df
        features = [col for col in df.columns if col in color_map]
    else:
        if isinstance(features, str):
            features = [features]
        
        # Keep features that exist in the dataframe AND are numeric
        features = [
            col for col in features 
            if col in df.columns and pd.to_numeric(df[col], errors='coerce').notna().any()
        ]

    num_features = len(features)
    if num_features == 0:
        print("Warning: No matching numeric features found to plot.")
        return
        
    # 3. Dynamically scale the figure height based on the number of features
    dynamic_height = max(2.0 * num_features, 4)
    
    # squeeze=False ensures 'axes' is always a 2D array, even if nrows=1
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(18, dynamic_height), sharex=True, squeeze=False)
    
    # 4. Loop through columns and axes to plot them individually
    for i, col in enumerate(features):
        ax = axes[i, 0]  # Flatten to 1D index
        
        # Get custom color or pull sequentially from my backup cycle
        if col in color_map:
            col_color = color_map[col]
        else:
            col_color = next(fallback_colors)
            
        # Plot on the specific axis
        ax.plot(df.index, df[col], color=col_color, label=col)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        ax.set_ylabel(col, fontsize=9)
        
    # Set the shared x-axis label on the absolute bottom plot
    axes[-1, 0].set_xlabel('Date')
    fig.suptitle('Individual Predictor Time-Series', y=0.99)
    
    # Adjust layout so labels don't overlap
    plt.tight_layout()
    plt.show()
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def delete_nans_and_zeros(df):
    '''
    This function:
    1. Deletes any column with >50% zeros or NaNs.
    2. Identifies remaining NaNs, creates a binary missing indicator, and applies linear interpolation.
    3. Identifies negative values in numeric columns, creates a negative indicator, and clips values to 0.
    
    Returns:
        df_out: The modified DataFrame.
        dropped_nan_cols: List of column names dropped due to >50% NaNs.
        dropped_zero_cols: List of column names dropped due to >50% Zeros.
        missing_indicator_cols: List of newly created missing indicator column names.
        negative_indicator_cols: List of newly created negative indicator column names.
    '''    
    # Work on a copy to prevent SettingWithCopy or unintended mutations outside the function
    df_out = df.copy()
    
    print(f"The length of the dataframe before cleaning is {len(df_out)}")
    
    # 1. Identify and drop columns with more than 50% NaN values
    dropped_nan_cols = list(df_out.columns[df_out.isnull().mean() > 0.50])
    print("Deleted columns with nan greater than 50% of the total data:", dropped_nan_cols)
    df_out.drop(columns=dropped_nan_cols, inplace=True)

    # 2. Identify and drop columns with more than 50% zero values
    dropped_zero_cols = list(df_out.columns[(df_out == 0).mean() > 0.50])
    print("Deleted columns with zero greater than 50% of the total data:", dropped_zero_cols)
    df_out.drop(columns=dropped_zero_cols, inplace=True)
    
    # Lists to keep track of newly engineered indicator columns
    missing_indicator_cols = []
    negative_indicator_cols = []
    
    # Process remaining columns
    for col in df_out.columns:
        # Check the column for NaNs 
        if df_out[col].isna().sum() > 0:
            print(f"Before cleaning, {col} has {df_out[col].isna().sum()} NaNs")
            
            # Make another column to tell what part of the data was missing
            missing_col_name = f'{col}_WAS_MISSING'
            df_out[missing_col_name] = df_out[col].isna().astype(int)
            missing_indicator_cols.append(missing_col_name)
            
            # Fill NaN with linear interpolation
            df_out[col] = df_out[col].interpolate(method='linear', limit_direction='both')

        # Convert negative values to zero for numeric columns only
        if pd.api.types.is_numeric_dtype(df_out[col]):
            negative_count = (df_out[col] < 0).sum()

            if negative_count > 0:
                print(f"Before cleaning, {col} has {negative_count} negative values")
                
                # Make another column to mark values that were negative
                negative_col_name = f"{col}_WAS_NEGATIVE"
                df_out[negative_col_name] = (df_out[col] < 0).astype(int)
                negative_indicator_cols.append(negative_col_name)
                
                # Replace negative values with zero
                df_out[col] = df_out[col].clip(lower=0)
                
    return df_out, dropped_nan_cols, dropped_zero_cols, missing_indicator_cols, negative_indicator_cols
#-------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_clean_plot(df):
    '''
    This function is going to plot the data before cleaning, clean and then plot again. 
    '''
    print('\033[1;34m' + 'how many nans before cleaning' + '\033[0m')
    print(df.isna().sum())

    target_col=['BORE_OIL_VOL_STB_D', 'BORE_GAS_VOL_MSCF_D','BORE_WAT_VOL_STB_D']

    # plot before cleaning
    print('\033[1;34m' + 'Plot before cleaning' + '\033[0m')
    seperate_target_plot(df[target_col])
    seperate_predictor_plot(df)

    # cleaning the data
    print('\033[1;34m' + 'Cleaning data' + '\033[0m')
    df_out, dropped_nan_cols, dropped_zero_cols, missing_indicator_cols, negative_indicator_cols=delete_nans_and_zeros(df)
    
    # check nans after cleaning
    print('\033[1;34m' + 'how many nans after cleaning' + '\033[0m')
    print(df_out.isna().sum()) 

    # plot after cleaning
    print('\033[1;34m' + 'Plot after cleaning' + '\033[0m')
    seperate_target_plot(df_out[target_col])
    seperate_predictor_plot(df_out)
    return df_out, dropped_nan_cols, dropped_zero_cols, missing_indicator_cols, negative_indicator_cols