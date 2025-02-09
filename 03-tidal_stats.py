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

**Important**: We manually convert Python datetimes to "days since 1992-01-01"
to avoid the recent timescale/time conflict in pyTMD.
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

# Older pyTMD API calls
import pyTMD.io.FES
import pyTMD.predict

warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------------------------------------------------------------
# 1. READ EOT20 CONSTANTS (older pyTMD API)
# -----------------------------------------------------------------------------
def load_eot20_constants(base_path):
    """
    Load the EOT20 NetCDF files from the specified directory.
    Returns a 'constituents' object with amplitude/phase grids.
    """
    model_files = sorted(glob.glob(os.path.join(base_path, "ocean_tides", "*.nc")))
    if len(model_files) == 0:
        raise FileNotFoundError(
            "No EOT20 netCDF found in 'EOT20/ocean_tides/*.nc' under:\n"
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
        method="spline",
        extrapolate=True,
        cutoff=10.0,
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
    Estimate new and full moon dates between 'start' and 'end'.
    """
    ref_new_moon = datetime(2024, 1, 11)
    synodic = 29.53  # days
    phases = []
    current_new = ref_new_moon
    while current_new < end:
        if current_new >= start:
            phases.append(("new", current_new))
        full_moon = current_new + timedelta(days=synodic / 2)
        if (full_moon >= start) and (full_moon < end):
            phases.append(("full", full_moon))
        current_new += timedelta(days=synodic)
    phases.sort(key=lambda x: x[1])
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
        "--path_to_tide_models",
        default="in-data-3p/World_EOT20_2021",
        help="Path to parent folder containing EOT20/ocean_tides/*.nc"
    )
    parser.add_argument(
        "--path_input_grid",
        default="working/AU_AIMS_EOT20-model-grid.tif",
        help="Path to the input grid (GeoTiff) to specify where the tide modelling should be done. Created using 02-tide_model_grid.py"
    )
    parser.add_argument(
        "--start-date", default="2024-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default="2024-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--time_step", type=float, default=0.5,
        help="Time step in hours"
    )
    parser.add_argument(
        "--working_path", default="working",
        help="Folder for intermediate outputs"
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
        "--name_switch", action="store_true",
        help="If set, switch output to LAT/HAT naming."
    )
    
    parser.add_argument(
    "--debug", action="store_true",
    help="Enable debug plotting for each processed pixel."
)

    args = parser.parse_args()

    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    times = pd.date_range(start=start_dt, end=end_dt, freq=f"{args.time_step}H")

    # Read the tide-model grid mask
    with rasterio.open(args.tide_model_grid) as src:
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
    sim_years = f"{start_dt.year}-{start_dt.month}_{end_dt.year}-{end_dt.month}"
    lpt_tide_label = "LPT" if args.name_switch else "LAT"
    hpt_tide_label = "HPT" if args.name_switch else "HAT"
    base = f"EOT20_{sim_years}_strip_{index}.tif"
    lpt_file = os.path.join(args.working_path, lpt_tide_label+"_" + base)
    hpt_file = os.path.join(args.working_path, hpt_tide_label+"_" + base)
    mlws_file = os.path.join(args.working_path, "MLWS_" + base)
    mhws_file = os.path.join(args.working_path, "MHWS_" + base)

    if not os.path.exists(args.working_path):
        os.makedirs(args.working_path)

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
    eot20_consts = load_eot20_constants(args.path_to_tide_models)

    # List cells to process
    process_indices = []
    for row in range(height):
        for col in range(start_col, end_col):
            if grid[row, col] == 1:
                if np.isnan(lpt_arr[row, col - start_col]):
                    process_indices.append((row, col))

    total = len(process_indices)
    print(f"Processing {total} grid cells in slice {index} (cols={start_col}:{end_col})")

    # Main loop
    count = 0
    for row, col in tqdm(process_indices, total=total):
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

        # ==================
        # DEBUG PLOTTING BLOCK
        # ==================
        if args.debug:
            phases = compute_moon_phases(start_dt, end_dt)

            plt.figure(figsize=(10, 5))
            plt.plot(times, tide_series, label="Tide Series", color="black")

            # Add moon phase markers
            for phase, phase_time in phases:
                color = "blue" if phase == "new" else "red"
                linestyle = "--" if phase == "new" else "-."
                plt.axvline(
                    x=phase_time, color=color, linestyle=linestyle, alpha=0.7,
                    label="New Moon" if phase == "new" else "Full Moon"
                )

            # Add title with computed stats
            plt.title(
                f"Tide Prediction at (row={row}, col={col})\n"
                f"LPT={lpt:.3f}, HPT={hpt:.3f}, MLWS={mlws:.3f}, MHWS={mhws:.3f}"
            )

            plt.xlabel("Time")
            plt.ylabel("Tide Elevation (m)")
            plt.grid(True)
            plt.legend()
            plt.show(block=True)  # blocks execution until the plot window is closed

            print(
                f"\nDebug Info:\n"
                f"  row={row}, col={col}\n"
                f"  LPT (Lowest Predicted Tide) = {lpt:.3f}\n"
                f"  HPT (Highest Predicted Tide) = {hpt:.3f}\n"
                f"  MLWS (Mean Low Water Springs) = {mlws:.3f}\n"
                f"  MHWS (Mean High Water Springs) = {mhws:.3f}\n"
            )

            input("Press Enter to continue to the next pixel...")

            plt.close()  # close the figure to free memory
            # ==================
            # END DEBUG BLOCK
            # ==================


        count += 1
        if (count % 10) == 0:
            # save partial results
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
                with rasterio.open(f_, "w", **profile) as dst:
                    dst.write(arr_, 1)
                    dst.update_tags(
                        start_date=str(args.start_date),
                        end_date=str(args.end_date),
                        time_step_hours=str(args.time_step),
                        tide_model="EOT20_ManualTime",
                    )

    # Final save
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
        with rasterio.open(f_, "w", **profile) as dst:
            dst.write(arr_, 1)
            dst.update_tags(
                start_date=str(args.start_date),
                end_date=str(args.end_date),
                time_step_hours=str(args.time_step),
                tide_model="EOT20_ManualTime",
            )

    print(f"Tidal statistics complete for slice {index}.")

if __name__ == "__main__":
    main()
