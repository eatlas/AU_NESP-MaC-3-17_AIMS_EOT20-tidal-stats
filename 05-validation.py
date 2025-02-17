"""
This script is incomplete. This is just a test for processing the tide data from BOM for a single site.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_tide_gauge_data(filename):
    """
    Reads a tide gauge file and returns a DataFrame with the time series data.
    
    Each complete data row should have 8 columns:
      Month, Year, Gaps, Good, Minimum, Maximum, Mean, StDevn
    
    If any data row is missing the sea level data (fewer than 8 tokens)
    then this function returns None to indicate that the station has too many gaps.
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip blank lines
            if not line:
                continue

            # Process only lines that start with a digit (data rows)
            if line[0].isdigit():
                tokens = line.split()
                # Check if the row contains the full set of expected columns
                if len(tokens) < 8:
                    print(f"Data gap detected in line: {line}")
                    return None

                try:
                    month   = int(tokens[0])
                    year    = int(tokens[1])
                    gaps    = int(tokens[2])
                    good    = int(tokens[3])
                    minimum = float(tokens[4])
                    maximum = float(tokens[5])
                    mean    = float(tokens[6])
                    stdev   = float(tokens[7])
                except Exception as e:
                    print(f"Error parsing line: {line}\n{e}")
                    return None

                data.append({
                    'year': year,
                    'month': month,
                    'gaps': gaps,
                    'good': good,
                    'minimum': minimum,
                    'maximum': maximum,
                    'mean': mean,
                    'stdev': stdev
                })
    
    if not data:
        return None  # No valid data found

    return pd.DataFrame(data)

def calc_monthly_stats(df):
    """
    Calculate monthly statistics from the tide gauge DataFrame.
    
    For each month (1 to 12) we:
      - Remove samples where 'gaps' > 100.
      - Count the number of valid samples.
      - Estimate LAT using the 10th percentile of the 'minimum' values.
      - Estimate Mean Sea Level using the median of the 'mean' values.
      - Estimate HAT using the 90th percentile of the 'maximum' values.
      
    Returns a DataFrame with one row per month.
    """
    monthly_stats = []
    for month in range(1, 13):
        subset = df[df['month'] == month]
        # Remove samples where gaps are greater than 100.
        subset = subset[subset['gaps'] <= 100]
        sample_count = len(subset)
        if sample_count > 0:
            lat = subset['minimum'].quantile(0.10)
            median_mean = subset['mean'].median()
            hat = subset['maximum'].quantile(0.90)
        else:
            lat = np.nan
            median_mean = np.nan
            hat = np.nan
        
        monthly_stats.append({
            'month': month,
            'sample_count': sample_count,
            'LAT (10th pct min)': lat,
            'Median Mean': median_mean,
            'HAT (90th pct max)': hat
        })
    return pd.DataFrame(monthly_stats)

def plot_monthly_stats(monthly_df):
    """
    Plots the monthly statistics as three lines:
      - LAT (10th percentile of minimum)
      - Mean Sea Level (median of mean)
      - HAT (90th percentile of maximum)
    """
    month_numbers = monthly_df['month']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    plt.figure(figsize=(10, 6))
    plt.plot(month_numbers, monthly_df['LAT (10th pct min)'], marker='o', label='LAT (10th pct min)')
    plt.plot(month_numbers, monthly_df['Median Mean'], marker='o', label='Mean Sea Level (median)')
    plt.plot(month_numbers, monthly_df['HAT (90th pct max)'], marker='o', label='HAT (90th pct max)')
    
    plt.xticks(month_numbers, month_names)
    plt.xlabel('Month')
    plt.ylabel('Sea Level (m)')
    plt.title('Monthly Sea Level Statistics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Replace the filename with your actual file path.
    filename = "in-data-3p/AU_BOM_Tide-gauge/IDO70000_59250_SLD.txt"
    df = read_tide_gauge_data(filename)
    
    if df is None:
        print("Data not available (too many gaps or file format error).")
    else:
        # Calculate overall statistics (for reference)
        overall_min = df['minimum'].min()
        overall_max = df['maximum'].max()
        overall_mean = df['mean'].mean()
        print("Overall Statistics:")
        print(f"  Minimum sea level: {overall_min}")
        print(f"  Mean sea level:    {overall_mean}")
        print(f"  Maximum sea level: {overall_max}")
        
        # Calculate and report monthly statistics
        monthly_df = calc_monthly_stats(df)
        print("\nMonthly Statistics:")
        print(monthly_df)
        
        # Create a plot of the monthly statistics.
        plot_monthly_stats(monthly_df)
