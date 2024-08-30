import pandas as pd

def load_data_from_config(config_file):
    """
    Load data from a CSV file using the URL specified in the config file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    
    # Extract the URL from the config
    data_url = config_file['paths']['data']
    
    # Column names for the DataFrame
    names = ['year', 'month', 'day', 'dec_date', 'sn_value',
             'sn_error', 'obs_num', 'unused1']
    
    # Column 1-3: Gregorian calendar date
    # - Year
    # - Month
    # - Day
    # Column 4: Date in fraction of year
    # Column 5: Daily total sunspot number. A value of -1 indicates that no number is available for that day (missing value).
    # Column 6: Daily standard deviation of the input sunspot numbers from individual stations.
    # Column 7: Number of observations used to compute the daily value.
    # Column 8: Definitive/provisional indicator. A blank indicates that the value is definitive. A '*' symbol indicates that the value is still provisional and is subject to a possible revision (Usually the last 3 to 6 months)
        
    # Load the data from the URL
    df = pd.read_csv(data_url, sep=';', header=None, names=names,
                     na_values=['-1'], index_col=False)
    
    return df

