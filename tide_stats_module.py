"""
This is just a common place for the main processing functions. They are placed here to 
allow easier reuse and testing.
"""
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

    print("Reading in consts")
    # read_constants loads amplitude/phase for each constituent
    eot20_consts = pyTMD.io.FES.read_constants(
        model_files=model_files,
        type="z",           # 'z' = vertical displacement
        version="EOT20",
        compressed=False    # set True if they are .gz
    )
    
    print(eot20_consts)
    return eot20_consts


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


def compute_tidal_stats(time_series, tide_series, start_dt, end_dt):
    """
    LPT = min tide
    HPT = max tide
    MLWS = mean of min tide in (-12 hours to +4 days) window around new moon & full moon
    MHWS = mean of max tide in (-12 hours to +4 days) window around new moon & full moon
    """
    lpt = np.min(tide_series)
    hpt = np.max(tide_series)

    phases = compute_moon_phases(start_dt, end_dt)
    low_tides = []
    high_tides = []

    for phase, phase_time in phases:
        # Time differences in seconds
        time_deltas = (time_series - phase_time).total_seconds()

        # Apply the (-12 hours to +4 days) time window
        window_mask = (time_deltas >= -12 * 3600) & (time_deltas <= 4 * 86400)
        if window_mask.any():
            window_vals = tide_series[window_mask]

            # Collect min and max tides within this window for both phases
            low_tides.append(np.min(window_vals))
            high_tides.append(np.max(window_vals))

    # Compute MLWS and MHWS from the collected tide values
    mlws = np.mean(low_tides) if len(low_tides) > 0 else np.nan
    mhws = np.mean(high_tides) if len(high_tides) > 0 else np.nan

    return lpt, hpt, mlws, mhws

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
        dst.write(data_array, 1)
        dst.update_tags(**metadata)

def get_metadata_tags(config, software):
    """
    This function generates metadata suitable for the generated Geotiff files.
    It pulls from the YAML configuration file to populate the metadata.
    Parameters:
    - config (dict): Dictionary containing configuration parameters.
    - software (str): Name of the script used to generate the data.
    """
    metadata_tags = {
        "start_date": config.get("start_date"),
        "end_date": config.get("end_date"),
        "time_step_hours": config.get("time_step"),
        "tide_model": "Hart-Davis Michael, Piccioni Gaia, Dettmering Denise, Schwatke Christian, Passaro Marcello, Seitz Florian (2021). EOT20 - A global Empirical Ocean Tide model from multi-mission satellite altimetry. SEANOE. https://doi.org/10.17882/79489",
        "description": "Tidal statistics derived from EOT20 using pyTMD.",
        "units": "metres",
        "author": config.get("author"),
        "organization": config.get("organization"),
        "date_created": datetime.now().strftime("%Y-%m-%d"),
        "software": software,
        "reference": config.get("reference"),
        "metadata_link": config.get("metadata_link"),
        "license": config.get("license")
    }
    return metadata_tags

