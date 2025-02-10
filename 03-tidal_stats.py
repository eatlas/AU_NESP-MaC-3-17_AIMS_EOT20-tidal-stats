#!/usr/bin/env python
"""
03-tidal_stats.py

This script computes tidal statistics (Lowest/Highest Predicted Tide and
Mean Low/High Water Springs) for each grid cell indicated for processing
in the tide-model grid. It uses pyTMD to read EOT20 tidal constituents
(via pyTMD.io.FES.read_constants and interpolate_constants) and then
simulates tidal elevations over a user-specified time period with a default
timestep of 0.5 hours. The work is split into vertical slices for parallel
processing. Intermediate results are saved every 100 processed pixels to
allow for script restart.

Quick start example:
    python 03-tidal_stats.py --config config/king-sound-quick-test.yaml
    
Example parallel processing:
    python 03-tidal_stats.py --config config/king-sound-quick-test.yaml --split 2 --index 0
    python 03-tidal_stats.py --config config/king-sound-quick-test.yaml --split 2 --index 1

"""
import argparse
import os
import glob
import numpy as np
import rasterio
from rasterio.transform import xy
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

import pyTMD.io.FES
import pyTMD.predict
import util as util

import signal
import sys
import threading

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global flag to allow graceful termination
stop_processing = False

def quit_listener():
    """
    Waits for the user to type 'q' and press Enter to quit.
    """
    global stop_processing
    print("\nPress 'q' and Enter at any time to stop the script gracefully...\n")
    while True:
        user_input = input().strip().lower()
        if user_input == "q":
            stop_processing = True
            print("\nStopping after current process...\n")
            break

# Start quit listener thread
quit_thread = threading.Thread(target=quit_listener, daemon=True)
quit_thread.start()



# -----------------------------------------------------------------------------
# 1. READ EOT20 CONSTANTS 
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
# 2. TIDE-PREDICTION FUNCTION
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
        scale=1.0  # if the EOT20 is already in meters
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

# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Calculate tidal statistics from tide-model grid (EOT20) with older pyTMD API."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file containing model run parameters."
    )
    parser.add_argument(
        "--split", type=int, default=1,
        help="Number of vertical slices"
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="0-based index of the slice to process"
    )
    
    parser.add_argument(
    "--debug", action="store_true",
    help="Enable debug plotting for each processed pixel."
)

    args = parser.parse_args()

    # List of required configuration parameters.
    required_params = [
        "clipped_tide_model_path",
        "grid_path",
        "start_date",
        "end_date",
        "time_step",
        "working_path",
        "lat_label",
        "hat_label"
    ]

    # Load model run parameters from the YAML config file.
    config = util.load_config(args.config, required_params)

    print("Started script with the following configuration:")
    util.print_config(config)

    # Unpack configuration values from YAML.
    clipped_tide_model_path = config.get("clipped_tide_model_path")
    grid_path = config.get("grid_path")
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    time_step = config.get("time_step")
    working_path = config.get("working_path")
    lat_label = config.get("lat_label")
    hat_label = config.get("hat_label")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    times = pd.date_range(start=start_dt, end=end_dt, freq=f"{time_step}H")

    # Read the tide-model grid mask
    with rasterio.open(grid_path) as src:
        grid = src.read(1)  # 1=process, 0=skip
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

    # Determine columns for this slice
    split = args.split
    index = args.index
    cols_per_slice = width // split
    extra = width % split
    start_col = index * cols_per_slice
    if index == (split - 1):
        end_col = width
    else:
        end_col = start_col + cols_per_slice

    slice_width = end_col - start_col

    # Prepare empty arrays
    lpt_arr = np.full((height, slice_width), np.nan, dtype=np.float32)
    hpt_arr = np.full((height, slice_width), np.nan, dtype=np.float32)
    mlws_arr = np.full((height, slice_width), np.nan, dtype=np.float32)
    mhws_arr = np.full((height, slice_width), np.nan, dtype=np.float32)

    # Build output filenames
    sim_years = f"{start_dt.strftime('%Y-%m')}_{end_dt.strftime('%Y-%m')}"
    base = f"EOT20_{sim_years}_strip_{index}.tif"
    lpt_file = os.path.join(working_path, lat_label+"_" + base)
    hpt_file = os.path.join(working_path, hat_label+"_" + base)
    mlws_file = os.path.join(working_path, "MLWS_" + base)
    mhws_file = os.path.join(working_path, "MHWS_" + base)

    if not os.path.exists(working_path):
        os.makedirs(working_path)

    # Resume from existing partial outputs if present
    for fname, arr_ref in zip(
        [lpt_file, hpt_file, mlws_file, mhws_file],
        [lpt_arr, hpt_arr, mlws_arr, mhws_arr]
    ):
        if os.path.exists(fname):
            with rasterio.open(fname) as src_in:
                data = src_in.read(1)
            arr_ref[:] = data

    # Load EOT20 constants (older pyTMD approach)
    eot20_consts = load_eot20_constants(clipped_tide_model_path)

    # List cells to process
    process_indices = []
    for row in range(height):
        for col in range(start_col, end_col):
            if grid[row, col] == 1:
                if np.isnan(lpt_arr[row, col - start_col]):
                    process_indices.append((row, col))

    total = len(process_indices)
    print(f"Processing {total} grid cells in slice {index} (cols={start_col}:{end_col})")

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


    # ---------------------------------------------------------------------------
    # Main loop for processing grid cells
    # ---------------------------------------------------------------------------
    count = 0
    metadata_tags = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "time_step_hours": str(time_step),
        "tide_model": "Hart-Davis Michael, Piccioni Gaia, Dettmering Denise, Schwatke Christian, Passaro Marcello, Seitz Florian (2021). EOT20 - A global Empirical Ocean Tide model from multi-mission satellite altimetry. SEANOE. https://doi.org/10.17882/79489",
        "description": f"Tidal statistics derived from EOT20 using pyTMD.",
        "author": "Eric Lawrey",
        "organization": "Australian Institute of Marine Science",
        "date_created": datetime.now().strftime("%Y-%m-%d"),
        "software": "03-tidal_stats.py using Rasterio",
        "reference": "Tidal Statistics for Australia (Tidal range, LAT, HAT, MLWS, MHWS) derived from the EOT20 tidal model (NESP MaC 3.17, AIMS) (V1) [Data set]. eAtlas. https://doi.org/10.26274/z8b6-zx94",
        "metadata_link": "https://doi.org/10.26274/z8b6-zx94",
        "license": "CC-BY 4.0"
    }
    

    for row, col in tqdm(process_indices, total=total):
        if stop_processing:  # If 'q' was pressed, exit cleanly
            print("\nGraceful shutdown initiated...\n")
            break
        lon, lat = xy(transform, row, col, offset="center")

        # Predict tide
        tide_series = predict_tide(lat, lon, times, eot20_consts)

        # Compute stats
        lpt, hpt, mlws, mhws = compute_tidal_stats(times, tide_series, start_dt, end_dt)
        j = col - start_col
        lpt_arr[row, j] = lpt
        hpt_arr[row, j] = hpt
        mlws_arr[row, j] = mlws
        mhws_arr[row, j] = mhws

        # Debug plotting (only if --debug flag is set)
        if args.debug:
            phases = compute_moon_phases(start_dt, end_dt)
            plt.figure(figsize=(10, 5))
            plt.plot(times, tide_series, label="Tide Series", color="black")

            for phase, phase_time in phases:
                color = "blue" if phase == "new" else "red"
                linestyle = "--" if phase == "new" else "-."
                plt.axvline(x=phase_time, color=color, linestyle=linestyle, alpha=0.7,
                            label="New Moon" if phase == "new" else "Full Moon")

            plt.title(f"Tide Prediction at (row={row}, col={col})\n"
                      f"LPT={lpt:.3f}, HPT={hpt:.3f}, MLWS={mlws:.3f}, MHWS={mhws:.3f}")

            plt.xlabel("Time")
            plt.ylabel("Tide Elevation (m)")
            plt.grid(True)
            plt.legend()
            plt.show(block=True)

            print(f"\nDebug Info:\n"
                  f"  row={row}, col={col}\n"
                  f"  LPT = {lpt:.3f}\n"
                  f"  HPT = {hpt:.3f}\n"
                  f"  MLWS = {mlws:.3f}\n"
                  f"  MHWS = {mhws:.3f}\n")

            input("Press Enter to continue to the next pixel...")
            plt.close()

        # Save partial results every 10 pixels
        count += 1
        if (count % 100) == 0:
            profile = {
                "driver": "GTiff",
                "height": height,
                "width": slice_width,
                "count": 1,
                "dtype": "float32",
                "crs": crs,
                "transform": transform,
                "nodata": np.nan
            }
            for arr_, f_ in [
                (lpt_arr, lpt_file),
                (hpt_arr, hpt_file),
                (mlws_arr, mlws_file),
                (mhws_arr, mhws_file)
            ]:
                save_geotiff(f_, arr_, profile, metadata_tags)


    # ---------------------------------------------------------------------------
    # Final save
    # ---------------------------------------------------------------------------
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": slice_width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan
    }

    for arr_, f_ in [
        (lpt_arr, lpt_file),
        (hpt_arr, hpt_file),
        (mlws_arr, mlws_file),
        (mhws_arr, mhws_file)
    ]:
        save_geotiff(f_, arr_, profile, metadata_tags)

    if args.split>1:
        print("To merge the multiple grids into the final grids run:")
        print(f"python 04-merge_strips.py --split {args.split} --config {args.config}")
    print(f"Tidal statistics complete for slice {args.index}. ")
    print("Exiting program safely.")
    sys.exit(0)  # Ensures a clean exit with status code 0

if __name__ == "__main__":
    main()
