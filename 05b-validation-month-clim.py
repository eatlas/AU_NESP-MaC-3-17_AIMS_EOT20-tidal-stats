import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.tseries.offsets import MonthEnd
from urllib.parse import urlparse

# Import our custom functions from tide_stats_module.py
from tide_stats_module import predict_tide, load_eot20_constants, read_tide_gauge_data

# ----------------------------------------------------
# Unified function to compute monthly climatology
# ----------------------------------------------------
def calc_monthly_stats(df, filter_gaps=False, gap_threshold=100, col_min="minimum", col_mean="mean", col_max="maximum"):
    """
    Calculate monthly statistics from a DataFrame.
    
    For each month (1 to 12) we compute:
      - The median of the column 'col_min' (median min)
      - The median of the column 'col_mean' (median mean)
      - The median of the column 'col_max' (median max)
      
    Optionally, if filter_gaps is True and the 'gaps' column exists,
    rows where gaps > gap_threshold are removed.
    
    Returns a DataFrame with one row per month.
    """
    monthly_stats = []
    for month in range(1, 13):
        subset = df[df['month'] == month]
        if filter_gaps and "gaps" in subset.columns:
            subset = subset[subset["gaps"] <= gap_threshold]
        sample_count = len(subset)
        if sample_count > 0:
            med_min = subset[col_min].median()
            med_mean = subset[col_mean].median()
            med_max = subset[col_max].median()
        else:
            med_min = med_mean = med_max = np.nan
        monthly_stats.append({
            "month": month,
            "sample_count": sample_count,
            "median_min": med_min,
            "median_mean": med_mean,
            "median_max": med_max
        })
    return pd.DataFrame(monthly_stats)

# ----------------------------
# Configuration and directories
# ----------------------------
gauges_csv = "data/in/BOM_tide-gauges.csv"
monthly_data_dir = "data/in-3p/AU_BOM_Monthly-tide-stats"
output_plots_dir = "data/validation/BOM-EOT20-tide-monthly-climatology"
os.makedirs(output_plots_dir, exist_ok=True)

# Directory containing the EOT20 NetCDF files for tide modelling.
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
# Define model prediction period (19 years ending on 31 Dec 2024)
# ----------------------------
model_start = pd.Timestamp("2006-01-01 00:00")
model_end = pd.Timestamp("2024-12-31 23:30")
print(f"Model prediction period: {model_start.date()} to {model_end.date()}\n")

# List to store RMS values for each station
rms_records = []

# ----------------------------
# Process each gauge station
# ----------------------------
for idx, row in gauges_df.iterrows():
    station_name = row['StationName']

    monthly_stats_url = row['MonthlyStatsURL']
    parsed_url = urlparse(monthly_stats_url)
    original_filename = os.path.basename(parsed_url.path)
    station_id = os.path.splitext(original_filename)[0]
    lat = row['Latitude']
    lon = row['Longitude']
    state = row['State']
    print(f"Processing station: {station_name} (ID: {station_id})")
    
    # Build path to monthly gauge stats file for this station
    monthly_file = os.path.join(monthly_data_dir, f"{station_id}.txt")
    if not os.path.exists(monthly_file):
        print(f"  -> Monthly stats file not found for station {station_id}. Skipping...\n")
        continue

    # Read the gauge station monthly tide data (all available years)
    gauge_df = read_tide_gauge_data(monthly_file)
    if gauge_df is None:
        print(f"  -> Data gaps too high in file for station {station_id}. Skipping...\n")
        continue

    # Create a datetime column (using day=1) and extract the month
    gauge_df["date"] = pd.to_datetime(dict(year=gauge_df.year, month=gauge_df.month, day=1))
    gauge_df["month"] = gauge_df["date"].dt.month

    # Calculate gauge climatology (using median on min, mean, max; filter out rows with gaps > 100)
    gauge_clim = calc_monthly_stats(gauge_df, filter_gaps=True, gap_threshold=100)
    gauge_data_start = gauge_df["date"].min().date()
    gauge_data_end = gauge_df["date"].max().date()
    print(f"  -> Gauge climatology computed over {gauge_data_start} to {gauge_data_end}")

    # ----------------------------
    # Generate tide model predictions for the fixed 19-year period
    # ----------------------------
    full_time_range = pd.date_range(start=model_start, end=model_end, freq="30min")
    tide_pred = predict_tide(lat, lon, full_time_range, eot20_consts)
    tide_pred = np.array(tide_pred)

    # Build a DataFrame for the predicted tide time series
    model_df = pd.DataFrame({"tide": tide_pred}, index=full_time_range)
    model_df["year"] = model_df.index.year
    model_df["month"] = model_df.index.month
    # Use the first day of each month as the representative date
    model_df["date"] = model_df.index.to_period("M").to_timestamp()

    # Aggregate the tide model predictions by each month (min, mean, max)
    monthly_model = (
        model_df.groupby("date")["tide"]
        .agg(["min", "mean", "max"])
        .reset_index()
        .sort_values("date")
    )
    monthly_model["month"] = monthly_model["date"].dt.month
    # Note: monthly_model has columns "min", "mean", "max" which we map to the function parameters

    # Calculate model climatology (using the median on min, mean, max)
    model_clim = calc_monthly_stats(monthly_model, filter_gaps=False, col_min="min", col_mean="mean", col_max="max")
    print("  -> Model climatology computed from tide model predictions.")

    # ----------------------------
    # Standardize gauge data to mean sea level
    # ----------------------------
    # Compute overall mean values from the climatologies (using median_mean)
    gauge_mean_avg = gauge_clim["median_mean"].mean()
    model_mean_avg = model_clim["median_mean"].mean()
    offset = model_mean_avg - gauge_mean_avg

    # Adjust gauge climatology so its mean matches the model
    gauge_clim_adj = gauge_clim.copy()
    gauge_clim_adj["median_min"] += offset
    gauge_clim_adj["median_mean"] += offset
    gauge_clim_adj["median_max"] += offset

    # ----------------------------
    # Compute RMS errors between adjusted gauge and model climatology
    # ----------------------------
    merged = pd.merge(gauge_clim_adj, model_clim, on="month", suffixes=("_gauge", "_model"))
    rms_min = np.sqrt(np.nanmean((merged["median_min_model"] - merged["median_min_gauge"])**2))
    rms_mean = np.sqrt(np.nanmean((merged["median_mean_model"] - merged["median_mean_gauge"])**2))
    rms_max = np.sqrt(np.nanmean((merged["median_max_model"] - merged["median_max_gauge"])**2))

    # Record RMS values for the station
    rms_records.append({
        "StationName": station_name,
        "StationID": station_id,
        "Latitude": lat,
        "Longitude": lon,
        "RMS_Min": rms_min,
        "RMS_Mean": rms_mean,
        "RMS_Max": rms_max
    })

    # ----------------------------
    # Create a comparison plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create month labels (e.g. Jan, Feb, …)
    months = np.arange(1, 13)
    month_labels = [datetime(2000, m, 1).strftime("%b") for m in months]
    
    # Plot adjusted gauge climatology (solid lines with markers)
    ax.plot(months, gauge_clim_adj["median_min"], "o-", color="navy", label="Gauge Median Min (adj)")
    ax.plot(months, gauge_clim_adj["median_mean"], "o-", color="darkgreen", label="Gauge Median Mean (adj)")
    ax.plot(months, gauge_clim_adj["median_max"], "o-", color="darkred", label="Gauge Median Max (adj)")
    
    # Plot model climatology (dashed lines with markers)
    ax.plot(months, model_clim["median_min"], "s--", color="blue", label="Model Median Min")
    ax.plot(months, model_clim["median_mean"], "s--", color="green", label="Model Median Mean")
    ax.plot(months, model_clim["median_max"], "s--", color="red", label="Model Median Max")
    
    # Set plot title including offset and RMS errors
    ax.set_title(
        f"{station_name} (ID: {station_id}) - {state}\n"
        f"Gauge Data: {gauge_data_start} to {gauge_data_end}\n"
        f"Offset = {offset:.3f} m  |  RMS (m): min = {rms_min:.3f}, mean = {rms_mean:.3f}, max = {rms_max:.3f}"
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Sea Level (m)")
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_plots_dir, f"{station_id}_climatology.png")
    fig.savefig(plot_filename)
    plt.close(fig)
    print(f"  -> Climatology plot saved to {plot_filename}\n")

# Save RMS values to a CSV file
rms_df = pd.DataFrame(rms_records)
rms_output_csv = os.path.join(output_plots_dir, "EOT20-Tide-gauge-clim-diff-rms.csv")
rms_df.to_csv(rms_output_csv, index=False)
print(f"RMS values saved to {rms_output_csv}")

print("Processing complete.")
