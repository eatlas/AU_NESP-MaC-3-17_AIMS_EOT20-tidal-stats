import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd
from urllib.parse import urlparse

# Import our custom functions from tide_stats_module.py
from tide_stats_module import predict_tide, load_eot20_constants, read_tide_gauge_data

# ----------------------------
# Configuration and directories
# ----------------------------
# Path to the CSV file containing gauge details
gauges_csv = "data/in/BOM_tide-gauges.csv"
# Directory where the monthly gauge stats files are stored
monthly_data_dir = "data/in-3p/AU_BOM_Monthly-tide-stats"
# Directory where output plots will be saved
output_plots_dir = "data/validation/BOM-EOT20-monthly-time-series"
os.makedirs(output_plots_dir, exist_ok=True)

# Directory containing the EOT20 NetCDF files for tide modelling.
# (Make sure this directory contains the required *.nc files.)
eot20_dir = "data/in-3p/World_EOT20_2021/ocean_tides" 

print("Loading EOT20 tide model constants...")
eot20_consts = load_eot20_constants(eot20_dir)
print("EOT20 tide model constants loaded.\n")

# ----------------------------
# Load gauge station list
# ----------------------------
print(f"Loading gauge station details from {gauges_csv}...")
gauges_df = pd.read_csv(gauges_csv)
print(f"Found {len(gauges_df)} tide gauge stations.\n")

# ----------------------------
# Process each gauge station
# ----------------------------
for idx, row in gauges_df.iterrows():
    station_name = row['StationName']
    monthly_stats_url = row['MonthlyStatsURL']
    parsed_url = urlparse(monthly_stats_url)
    original_filename = os.path.basename(parsed_url.path)
    station_id = os.path.splitext(original_filename)[0]
    #station_id = row['ID']
    lat = row['Latitude']
    lon = row['Longitude']
    state = row['State']
    print(f"Processing station: {station_name} (ID: {station_id})")

    # Build path to monthly stats file for this station
    monthly_file = os.path.join(monthly_data_dir, f"{station_id}.txt")
    if not os.path.exists(monthly_file):
        print(f"  -> Monthly stats file not found for station {station_id}. Skipping...\n")
        continue

    # Read the gauge station monthly tide data
    df = read_tide_gauge_data(monthly_file)
    if df is None:
        print(f"  -> Data gaps too high in file for station {station_id}. Skipping...\n")
        continue

    # ------
    # Filter the gauge data:
    # - Exclude any month where gaps > 200 samples.
    # - Create a datetime column from 'year' and 'month'
    # - Limit the time series to the 19 years from the most recent observation.
    # ------
    df = df[df['gaps'] <= 200]
    if df.empty:
        print(f"  -> No valid gauge data after filtering gaps > 200. Skipping...\n")
        continue

    # Create a datetime column (using day=1 for each month)
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    most_recent = df["date"].max()
    start_date = most_recent - pd.DateOffset(years=19)
    df = df[df["date"] >= start_date]
    if df.empty:
        print(f"  -> No gauge data within the most recent 19-year window. Skipping...\n")
        continue

    # Determine which month/year periods are valid in the gauge data
    valid_periods = set(df["date"].dt.to_period("M").unique())

    # ------
    # Generate tide model predictions
    # ------
    print(f"  -> Generating tide predictions for period {start_date.date()} to {most_recent.date()}")
    # Create a continuous time series with a 0.5hr time step
    # _most_recent corresponds to the first day of the last month. We then shift it to
    # the last day of the month with MonthEnd(0), then add 23.5 hours to reach the last 30 min
    adjusted_most_recent = most_recent + MonthEnd(0) + pd.Timedelta(hours=23, minutes=30)

    full_time_range = pd.date_range(start=start_date, end=adjusted_most_recent, freq="30min")

    # Keep only times that fall in months where the gauge has valid data.
    predicted_times = [t for t in full_time_range if t.to_period("M") in valid_periods]
    if not predicted_times:
        print(f"  -> No prediction times match the gauge data periods. Skipping station...\n")
        continue
    predicted_times = pd.to_datetime(predicted_times)

    # Predict tide series (in metres) at the gauge location over these times
    tide_pred = predict_tide(lat, lon, predicted_times, eot20_consts)
    tide_pred = np.array(tide_pred)  # Ensure it's a numpy array

    # Build a DataFrame for the predicted tide time series
    pred_df = pd.DataFrame({"tide": tide_pred}, index=predicted_times)
    pred_df["year"] = pred_df.index.year
    pred_df["month"] = pred_df.index.month
    # Use the first day of the month as the representative date for grouping
    pred_df["date"] = pred_df.index.to_period("M").to_timestamp()

    # Aggregate the tide model predictions by month (min, mean, max)
    monthly_pred = (
        pred_df.groupby("date")["tide"]
        .agg(["min", "mean", "max"])
        .reset_index()
        .sort_values("date")
    )

    # ------
    # Prepare the gauge monthly data for comparison
    # ------
    # We already have one row per month in the gauge file.
    gauge_monthly = df[["date", "minimum", "mean", "maximum"]].copy()
    gauge_monthly = gauge_monthly.sort_values("date")

    # Merge gauge and model monthly data on the date (month/year)
    merged = pd.merge(
        gauge_monthly,
        monthly_pred,
        on="date",
        how="inner",
        suffixes=("_gauge", "_model"),
    )
    if merged.empty:
        print("  -> No overlapping months between gauge and model predictions. Skipping station...\n")
        continue

    # Align gauge data to modelled data:
    # Compute an offset so that the mean of the gauge monthly means aligns with the model monthly means.
    # Note: gauge 'mean' is now 'mean_gauge' and model 'mean' is 'mean_model'
    gauge_mean_avg = merged["mean_gauge"].mean()
    model_mean_avg = merged["mean_model"].mean()
    offset = model_mean_avg - gauge_mean_avg
    print(f"  -> Calculated offset = {offset:.3f} m (to be added to gauge data)")

    # Apply the offset to all gauge stats.
    # The gauge columns for min and max did not conflict so they remain as "minimum" and "maximum"
    merged["minimum_adj"] = merged["minimum"] + offset
    merged["mean_gauge_adj"] = merged["mean_gauge"] + offset
    merged["maximum_adj"] = merged["maximum"] + offset

    # Compute RMS errors between the adjusted gauge and model monthly statistics.
    rms_min = np.sqrt(np.mean((merged["min"] - merged["minimum_adj"]) ** 2))
    rms_mean = np.sqrt(np.mean((merged["mean_model"] - merged["mean_gauge_adj"]) ** 2))
    rms_max = np.sqrt(np.mean((merged["max"] - merged["maximum_adj"]) ** 2))
    rms_overall = np.sqrt(
        np.mean(
            (
                (merged["min"] - merged["minimum_adj"]) ** 2 +
                (merged["mean_model"] - merged["mean_gauge_adj"]) ** 2 +
                (merged["max"] - merged["maximum_adj"]) ** 2
            ) / 3
        )
    )
    print(
        f"  -> RMS errors (m): min = {rms_min:.3f}, mean = {rms_mean:.3f}, max = {rms_max:.3f}, overall = {rms_overall:.3f}"
    )

    # ------
    # Create a comparison plot
    # ------
    fig, ax = plt.subplots(figsize=(10, 7), dpi = 200)
    # Plot gauge (adjusted) data with solid lines (no markers) in darker colors
    ax.plot(merged["date"], merged["minimum_adj"], "-", label="Gauge Min (adj)", color="#37b4fa")
    ax.plot(merged["date"], merged["mean_gauge_adj"], "-", label="Gauge Mean (adj)", color="#077ae0")
    ax.plot(merged["date"], merged["maximum_adj"], "-", label="Gauge Max (adj)", color="#205191")

    # Plot model predictions with dashed lines in standard (brighter) colors
    ax.plot(merged["date"], merged["min"], "--", label="Model Min", color="#eb7705")
    ax.plot(merged["date"], merged["mean_model"], "--", label="Model Mean", color="#cf4400")
    ax.plot(merged["date"], merged["max"], "--", label="Model Max", color="#a10212")

    # Update the title with station name, state, GPS coordinates, and note that data are monthly aggregates
    
    # Set the main title as a suptitle (larger font)
    fig.suptitle(f"Monthly Tidal Range Time Series for {station_name}, {state}",
                 fontsize=14, y=0.98)
                 
    # Create a subtitle (smaller font) that shows gauge data period and offset/RMS info.
    subtitle_text = (
        f"Tide Gauge vs EOT20 Tidal Model, ID: {station_id}, GPS: ({lat}, {lon})\n" 
        f"Tide Gauge Zero to EOT20 offset  = {offset:.3f} m  |  RMS (m): min = {rms_min:.3f}, mean = {rms_mean:.3f}, max = {rms_max:.3f}"
    )
    ax.set_title(subtitle_text, fontsize=10)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Sea Level (m) - EOT20 MSL")
    ax.legend()
    # Adjust layout to leave extra space at the bottom for attribution
    #fig.subplots_adjust(bottom=0.5)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    # Add footer text (attribution) with increased gap below the graph
    fig.text(0.5, 0.01, (
        "Derived from EOT20 Tidal Model and BOM Tide Gauge Data. Graph is licensed under CC BY 4.0\n"
        "Processing: AIMS, https://doi.org/10.26274/z8b6-zx94, Tidal model EOT20: https://doi.org/10.17882/79489\n"
        "Tide Gauge data: http://www.bom.gov.au/oceanography/projects/ntc/monthly/"),
             ha="center", fontsize=8, color='grey')
             
    ax.grid(True)
    plt.xticks(rotation=45)
    

    # Save the plot
    plot_filename = os.path.join(output_plots_dir, f"{station_id}_comparison.png")
    fig.savefig(plot_filename)
    plt.close(fig)
    print(f"  -> Plot saved to {plot_filename}\n")

print("Processing complete.")
 