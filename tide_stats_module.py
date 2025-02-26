"""
This is just a common place for the main processing functions. They are placed here to 
allow easier reuse and testing.
"""
import stat
import yaml
import pathlib
import xarray as xr

import pyTMD.io.FES
import pyTMD.predict
import numpy as np
from datetime import datetime, timedelta
import os
import glob
import rasterio
import pandas as pd

# -----------------------------------------------------------------------------
# LOAD YAML CONFIGURATION
# -----------------------------------------------------------------------------
def load_config(config_path: str, required_params: list[str]) -> dict:
    """
    Load configuration parameters from a YAML file and ensure that all expected parameters are present.
    If any required parameter is missing, an error is raised indicating which variable is missing.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Configuration file does not contain a valid YAML mapping.")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration file {config_path}: {e}")
    
    missing = [param for param in required_params if param not in config]
    if missing:
        raise ValueError(
            f"Missing configuration parameter(s): {', '.join(missing)}. "
            "Please check the configuration file and refer to the documentation for valid options."
        )
    
    return config

def print_config(config: dict):
    """Print the configuration parameters as a formatted list."""
    print("Configuration parameters:")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    print()

# -----------------------------------------------------------------------------
# TIDE MODEL CLIPPING
# -----------------------------------------------------------------------------
def _clip_eot20_model(ds: xr.Dataset, min_lon: float, min_lat: float, max_lon: float, max_lat: float):
    """
    Clip an EOT20 tide model xarray.Dataset to a specified bounding box.

    Assumes that ds.coords["lon"] is in the range [0, 360] and ds.coords["lat"]
    is in descending order.
    """
    lon_vals = ds["lon"].compute()
    lat_vals = ds["lat"].compute()

    def wrap_to_360(x):
        return x % 360

    left_360 = wrap_to_360(min_lon)
    right_360 = wrap_to_360(max_lon)
    lat_slice = slice(max_lat, min_lat) if lat_vals[0] > lat_vals[-1] else slice(min_lat, max_lat)

    if left_360 <= right_360:
        ds_clipped = ds.sel(lon=slice(left_360, right_360), lat=lat_slice)
    else:
        ds_left = ds.sel(lon=slice(left_360, 360), lat=lat_slice)
        ds_right = ds.sel(lon=slice(0, right_360), lat=lat_slice)
        ds_clipped = xr.concat([ds_left, ds_right], dim="lon")

    return ds_clipped


def clip_eot20_files(tide_model_path: str, clipped_tide_model_path: str,
                     min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                     clipping_buffer_deg: float = 5.0):
    """
    Clips EOT20 NetCDF files from the tide model constituent folder to the bounding box
    (with an added buffer) and saves them to the specified output folder.
    """
    in_dir = pathlib.Path(tide_model_path)
    out_dir = pathlib.Path(clipped_tide_model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply buffer to bounding box
    min_lon_b = min_lon - clipping_buffer_deg
    max_lon_b = max_lon + clipping_buffer_deg
    min_lat_b = min_lat - clipping_buffer_deg
    max_lat_b = max_lat + clipping_buffer_deg

    nc_files = list(in_dir.glob("*.nc"))
    if not nc_files:
        print(f"No NetCDF (.nc) files found in {in_dir} to clip.")
        return

    print(f"\nClipping tide model files from {in_dir} to bounding box "
          f"({min_lon_b}, {min_lat_b}, {max_lon_b}, {max_lat_b})...")

    for nc_file in nc_files:
        out_file = out_dir / nc_file.relative_to(in_dir)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with xr.open_dataset(nc_file, engine="netcdf4") as ds:
                ds_clipped = _clip_eot20_model(ds, min_lon_b, min_lat_b, max_lon_b, max_lat_b)
                ds_clipped.to_netcdf(out_file, mode="w")
            print(f"  - Clipped {nc_file.name} -> {out_file}")
        except Exception as e:
            print(f"Error processing file {nc_file}: {e}")

    print("Finished clipping tide model files.\n")

# -----------------------------------------------------------------------------
# LOAD TIDE MODEL DATA
# -----------------------------------------------------------------------------
def load_eot20_constants(base_path):
    """
    Load the EOT20 NetCDF files from the specified directory.
    Returns a 'constituents' object with amplitude/phase grids.
    """
    model_files = sorted(glob.glob(os.path.join(base_path, "*.nc")))
    if len(model_files) == 0:
        raise FileNotFoundError(
            "No EOT20 netCDF found in '*.nc' under:\n"
            f"  {base_path}"
        )

    
    # read_constants loads amplitude/phase for each constituent
    eot20_consts = pyTMD.io.FES.read_constants(
        model_files=model_files,
        type="z",           # 'z' = vertical displacement
        version="EOT20",
        compressed=False    # set True if they are .gz
    )
    
    
    return eot20_consts

def split_grid_columns_by_active_pixels(grid, num_splits):
    """
    Split the grid into approximately equal vertical strips based on active pixel counts.
    
    Parameters:
        grid (numpy.ndarray): A 2D array where active (tide modelling) pixels are marked with 1
                              and inactive pixels with 0.
        num_splits (int): The number of vertical splits (processes) desired.
    
    Returns:
        List[Tuple[int, int]]: A list of (start_col, end_col) indices for each vertical strip.
    
    The function calculates the total active pixels, computes an ideal number of pixels per split,
    and then iterates over each column summing the active pixels until the ideal is reached. The
    remaining columns after allocating num_splits-1 splits are assigned to the final split.
    """
    if num_splits < 1:
        raise ValueError("num_splits must be at least 1")
    
    height, width = grid.shape
    # Count active pixels in each column (assuming active pixels are marked with 1)
    active_counts = np.sum(grid == 1, axis=0)
    total_active = np.sum(active_counts)
    
    # If no active pixels, fall back to equal column splits
    if total_active == 0:
        cols_per_split = width // num_splits
        splits = []
        for i in range(num_splits):
            start = i * cols_per_split
            end = (i + 1) * cols_per_split if i < num_splits - 1 else width
            splits.append((start, end))
        return splits
    
    ideal_pixels_per_split = total_active / num_splits
    
    splits = []
    current_sum = 0
    start_col = 0
    current_split = 0
    
    for col in range(width):
        current_sum += active_counts[col]
        # Create a new split if the current sum meets or exceeds the ideal,
        # but leave the final split to take the remainder.
        if current_sum >= ideal_pixels_per_split and current_split < num_splits - 1:
            end_col = col + 1  # end index is exclusive
            splits.append((start_col, end_col))
            start_col = end_col  # start next split from the next column
            current_sum = 0
            current_split += 1
    
    # The remaining columns form the last split
    splits.append((start_col, width))
    return splits

# -----------------------------------------------------------------------------
# TIDE-PREDICTION FUNCTION
# -----------------------------------------------------------------------------
def predict_tide(lat, lon, times, eot20_consts):
    """
    Predict tidal elevations for a single lat/lon over a series of times
    using the older pyTMD FES/EOT approach and `predict.time_series`.
    
    Parameters
    ----------
    lat : float
        Latitude of the point (degrees)
    lon : float
        Longitude of the point (degrees)
    times : pandas.DatetimeIndex (length N)
        Array of time steps (datetimes) over which to predict tide
    eot20_consts : pyTMD.constituents
        Output of pyTMD.io.FES.read_constants(...) for EOT20
    
    Returns
    -------
    tide_series : np.ndarray, shape (N,)
        Tidal elevations in meters for each of the N time steps.
    """
    # 1) Interpolate amplitude/phase at this location
    amp, ph = pyTMD.io.FES.interpolate_constants(
        np.atleast_1d(lon),
        np.atleast_1d(lat),
        eot20_consts,
        method="bilinear",
        extrapolate=True,
        cutoff=np.inf,
        scale=0.01  # Convert from cm to metres. EOT tidal consitutents are in cm.
                    # are in units of cm.
    )
    # shape of amp, ph = (1, nConstituents)

    # 2) Convert amplitude/phase to complex harmonic coefficients
    #    ph in degrees => radians = ph*(pi/180) => multiply by -1j
    cph = -1j * ph * (np.pi / 180.0)
    hc = amp * np.exp(cph)  # shape: (1, nConstituents)

    # 3) Convert `times` to "days since 1992-01-01"
    epoch_1992 = datetime(1992, 1, 1)
    time_days = np.array(
        [(t - epoch_1992).total_seconds() / 86400.0 for t in times],
        dtype=np.float64
    )

    # 4) Predict tide series (shape: (Ntimes,)) at this single location
    #    eot20_consts.fields is the list of constituent names (e.g. ['m2','s2',...])
    #    'time_series' will compute the entire tide across all times in one call.
    tide_ts = pyTMD.predict.time_series(
        t=time_days,
        hc=hc,
        constituents=eot20_consts.fields,
        deltat=0.0,       # if no ET -> TT correction is needed
        corrections="FES" # FES-based or EOT-based nodal corrections
    )
    # tide_ts is a masked array, shape (Ntimes,)

    # If you just want a standard numpy array
    # (mask == False everywhere unless the location is invalid)
    tide_series = tide_ts.filled(np.nan)
    return tide_series

# -----------------------------------------------------------------------------
# Helper functions for tidal statistics
# -----------------------------------------------------------------------------
def compute_moon_phases(start, end):
    """
    Estimate new and full moon dates between 'start' and 'end',
    supporting both past and future years (at least 19 years back).

    Uses a reference new moon date (2000-01-06) and calculates cycles forward
    and backward using the synodic month (29.53 days).
    """
    ref_new_moon = datetime(2000, 1, 6)  # A known reference new moon date
    synodic = 29.53  # Average synodic month in days
    phases = []

    # Determine the closest previous new moon before `start`
    elapsed_days = (start - ref_new_moon).days
    cycles_since_ref = elapsed_days / synodic
    closest_new_moon = ref_new_moon + timedelta(days=round(cycles_since_ref) * synodic)

    # Iterate backwards to ensure we cover any missing past new moons
    current_new = closest_new_moon
    while current_new >= start:
        full_moon = current_new + timedelta(days=synodic / 2)
        if current_new <= end:
            phases.append(("new", current_new))
        if start <= full_moon <= end:
            phases.append(("full", full_moon))
        current_new -= timedelta(days=synodic)  # Move one cycle backward

    # Iterate forward to fill new moons in the given range
    current_new = closest_new_moon + timedelta(days=synodic)
    while current_new <= end:
        full_moon = current_new + timedelta(days=synodic / 2)
        if current_new >= start:
            phases.append(("new", current_new))
        if start <= full_moon <= end:
            phases.append(("full", full_moon))
        current_new += timedelta(days=synodic)  # Move one cycle forward

    phases.sort(key=lambda x: x[1])  # Ensure phases are ordered by date
    return phases

def compute_tidal_stats(time_series, tide_series, start_dt, end_dt, percentile_values: list[int]):
    """
    Compute overall tidal stats (LPT, HPT, MLWS, MHWS), percentiles, and monthly stats.
    
    Parameters
    ----------
    time_series : pandas.DatetimeIndex
        The timestamps corresponding to the tide_series.
    tide_series : np.ndarray
        Array of tide predictions.
    start_dt : datetime
        Start of the period for which tidal statistics are computed.
    end_dt : datetime
        End of the period for which tidal statistics are computed.
    percentile_values : list of int
        List of percentiles to compute.

    Returns
    -------
    result : dict
        Dictionary containing tidal statistics.
    """
    result = {}
    # Overall stats
    result['lpt'] = np.min(tide_series)
    result['hpt'] = np.max(tide_series)
    
    phases = compute_moon_phases(start_dt, end_dt)
    low_tides = []
    high_tides = []
    for phase, phase_time in phases:
        time_deltas = (time_series - phase_time).total_seconds()
        window_mask = (time_deltas >= -12 * 3600) & (time_deltas <= 4 * 86400)
        if window_mask.any():
            window_vals = tide_series[window_mask]
            low_tides.append(np.min(window_vals))
            high_tides.append(np.max(window_vals))
    result['mlws'] = np.mean(low_tides) if low_tides else np.nan
    result['mhws'] = np.mean(high_tides) if high_tides else np.nan

    percentiles = {}
    for p in percentile_values:
        percentiles[f"p{p:02d}"] = np.percentile(tide_series, p)
    result['percentiles'] = percentiles

    # Monthly stats
    ts_index = pd.DatetimeIndex(time_series)
    ts_series = pd.Series(tide_series, index=ts_index)
    monthly_group = ts_series.groupby(ts_series.index.to_series().dt.month)
    monthly_stats = monthly_group.agg(['min', 'mean', 'max'])
    monthly_result = {}
    for month in range(1, 13):
        if month in monthly_stats.index:
            row = monthly_stats.loc[month]
            monthly_result[month] = {
                'lpt': row['min'],
                'mean': row['mean'],
                'hpt': row['max']
            }
        else:
            monthly_result[month] = {'lpt': np.nan, 'mean': np.nan, 'hpt': np.nan}
    result['monthly'] = monthly_result

    return result


def dep_compute_tidal_stats(time_series, tide_series, start_dt, end_dt):
    """
    Compute overall tidal stats (LPT, HPT, MLWS, MHWS), percentiles, and monthly stats.
    
    Returns a dictionary with the following keys:
      - 'lpt': lowest predicted tide (min value)
      - 'hpt': highest predicted tide (max value)
      - 'mlws': mean low water spring (derived from a window around new/full moons)
      - 'mhws': mean high water spring (derived from a window around new/full moons)
      - 'percentiles': dict of tide series percentiles with keys 'p01', 'p02', 'p05', 'p10',
                       'p20', 'p50', 'p70', 'p90', 'p95', 'p98', 'p99'
      - 'monthly': dict mapping month numbers (1-12) to a dict with keys 'lpt', 'mean', and 'hpt'
    """
    result = {}
    # Overall stats
    result['lpt'] = np.min(tide_series)
    result['hpt'] = np.max(tide_series)
    
    phases = compute_moon_phases(start_dt, end_dt)
    low_tides = []
    high_tides = []
    for phase, phase_time in phases:
        time_deltas = (time_series - phase_time).total_seconds()
        window_mask = (time_deltas >= -12 * 3600) & (time_deltas <= 4 * 86400)
        if window_mask.any():
            window_vals = tide_series[window_mask]
            low_tides.append(np.min(window_vals))
            high_tides.append(np.max(window_vals))
    result['mlws'] = np.mean(low_tides) if low_tides else np.nan
    result['mhws'] = np.mean(high_tides) if high_tides else np.nan

    # Percentiles
    perc_values = [1, 2, 5, 10, 20, 50, 70, 90, 95, 98, 99]
    percentiles = {}
    for p in perc_values:
        percentiles[f"p{p:02d}"] = np.percentile(tide_series, p)
    result['percentiles'] = percentiles

    # Monthly stats
    # Explicitly ensure the index is a DatetimeIndex
    ts_index = pd.DatetimeIndex(time_series)
    ts_series = pd.Series(tide_series, index=ts_index)
    # Use the dt accessor on the index to group by month
    monthly_group = ts_series.groupby(ts_series.index.to_series().dt.month)
    monthly_stats = monthly_group.agg(['min', 'mean', 'max'])
    monthly_result = {}
    for month in range(1, 13):
        if month in monthly_stats.index:
            row = monthly_stats.loc[month]
            monthly_result[month] = {
                'lpt': row['min'],
                'mean': row['mean'],
                'hpt': row['max']
            }
        else:
            monthly_result[month] = {'lpt': np.nan, 'mean': np.nan, 'hpt': np.nan}
    result['monthly'] = monthly_result

    return result

def save_geotiff(file_path, data_array, profile, metadata):
    """
    Saves a 2D NumPy array as a GeoTIFF with specified metadata.

    Parameters:
    - file_path (str): Path to the output GeoTIFF file.
    - data_array (np.ndarray): 2D array containing tidal statistics.
    - profile (dict): Rasterio profile containing metadata about the GeoTIFF.
    - metadata (dict): Additional metadata tags to include in the file.
    """
    with rasterio.open(file_path, "w", **profile) as dst:
        # If the file is single-band, ensure we write a 2D array.
        if profile.get("count", 1) == 1:
            # If data_array is 3D with a singleton first dimension, squeeze it.
            if data_array.ndim == 3 and data_array.shape[0] == 1:
                data_array = data_array[0]
            dst.write(data_array, 1)
        else:
            # For multi-band data, ensure data_array has shape (bands, height, width)
            # Sometimes an extra dimension might sneak in (e.g. shape (1, bands, height, width))
            if data_array.ndim == 4 and data_array.shape[0] == 1:
                data_array = data_array[0]
            dst.write(data_array)
        dst.update_tags(**metadata)


def add_product_metadata(config: dict, product_type: str) -> dict:
    """
    Returns a copy of the base metadata updated with product-specific details
    and CF-compliant fields.
    """
    # Helper to get ordinal suffix (e.g., 1 -> "1st", 2 -> "2nd", etc.)
    def ordinal(n):
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
    
    base = {}
    stat_desc = ""
    stat_notes = ""
    if product_type == "Percentiles":
        # Use percentiles from config or a default list if not provided
        percentile_values = config.get("percentiles")
        band_descriptions = []
        for i, p in enumerate(percentile_values):
            label = f"p{p:02d}"
            # Optionally, treat 50th percentile specially (as median)
            if p == 50:
                desc = f"{label} (median)"
            else:
                desc = f"{label} ({ordinal(p)} percentile)"
            band_descriptions.append(f"Band {i+1}: {desc}")
        base["bands_description"] = ", ".join(band_descriptions)
        stat_desc = "Percentiles"
        stat_notes = "Percentiles are calculated from the entire tide series."
    elif product_type in ["Monthly_LPT", "Monthly_Mean", "Monthly_HPT"]:
        stat_lookup = {"Monthly_LPT": "Lowest Predicted Tide - By month",
                       "Monthly_Mean": "Mean Tide - By month",
                       "Monthly_HPT": "Highest Predicted Tide  - By month"}
        stat_desc = stat_lookup[product_type]
        month_names = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        base["bands_description"] = ", ".join(
            [f"Band {i+1}: {month}" for i, month in enumerate(month_names)]
        )
        stat_note_lookup = {"Monthly_LPT": "Lowest tide value for each month, across all years in the simulation period.",
                            "Monthly_Mean": "Mean tide value for each month, across all years in the simulation period.",
                            "Monthly_HPT": "Highest predicted tide for each month, across all years in the simulation period."}
        stat_notes = stat_note_lookup[product_type]
    elif product_type in ["LPT", "HPT", "LAT", "HAT"]:
        lat = product_type=='LPT' or product_type=='LAT'
        base["bands_description"] = f"{'Lowest' if lat else 'Highest'} Predicted Tide"
        stat_desc = base["bands_description"]
        stat_notes = f"{stat_desc} is calculated as the overall {'minimum' if lat else 'maximum'} tide over the simulation period."
    elif product_type in ["MLWS", "MHWS"]:
        base["bands_description"] = f"Mean {'Low' if product_type=='MLWS' else 'High'} Water Spring"
        stat_desc = base["bands_description"]
        stat_notes = (f"{stat_desc} is computed by averaging the {'low' if product_type=='MLWS' else 'high'} tide values "
                      "within a window 12 hours before to 4 days after new and full moon phases.")
    elif product_type == "Tidal_Range":
        base["bands_description"] = "Tidal Range (HPT - LPT)"
        stat_desc = "Tidal Range"
        stat_notes = "Tidal Range is calculated as the difference between the Highest Predicted Tide and Lowest Predicted Tide."
    extra = {
        "Conventions": "CF-1.8",
        "title": f"{stat_desc} - Tidal Statistics derived from EOT20",
        "summary": ("This dataset contains tidal statistics (e.g., lowest/highest "
                    "predicted tides, monthly statistics, and percentiles) derived from "
                    "the EOT20 tide model. Tide predictions are computed over a user-specified "
                    f"time period {config.get('start_date')} to {config.get('end_date')}."
                    "This data is from the following dataset: Tidal Statistics for Australia "
                    "Tidal range, LAT, HAT, MLWS, MHWS, Percentiles) derived from the EOT20 tidal model "
                    "(NESP MaC 3.17, AIMS) (V1) [Data set]. eAtlas. https://doi.org/10.26274/z8b6-zx94."
                    "The EOT20 model is a global Empirical Ocean Tide model described in "
                    "Hart-Davis Michael, Piccioni Gaia, Dettmering Denise, "
                    "Schwatke Christian, Passaro Marcello, Seitz Florian (2021). "
                    "EOT20 - A global Empirical Ocean Tide model from multi-mission satellite altimetry. "
                    "SEANOE. https://doi.org/10.17882/79489"),
        "author": config.get("author"),
        "description": config.get("description"),
        "processing": stat_notes,
        "institution": config.get("organization", "Unknown Institution"),
        "source": "EOT20 tide model (satellite altimetry) processed with pyTMD.",
        "history": f"Generated on {datetime.now().isoformat()} using 03-tidal_stats.py from https://github.com/eatlas/AU_NESP-MaC-3-17_AIMS_EOT20-tidal-stats",
        "start_date": config.get("start_date"),
        "end_date": config.get("end_date"),
        "time_step_hours": config.get("time_step"),
        "reference": config.get("reference", "No reference provided"),
        "metadata_link": config.get("metadata_link"),
        "license": config.get("license")
    }
    
    merged = base.copy()
    merged.update(extra)
    return merged


def read_tide_gauge_data(filename):
    """
    Reads a tide gauge file and returns a DataFrame with the time series data.
    
    Each complete data row should have 8 columns:
      Month, Year, Gaps, Good, Minimum, Maximum, Mean, StDevn
    
    Rows with missing sea level data (i.e., fewer than 8 tokens) are skipped.
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
                    print(f"Incomplete data row detected, skipping: {line}")
                    continue

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