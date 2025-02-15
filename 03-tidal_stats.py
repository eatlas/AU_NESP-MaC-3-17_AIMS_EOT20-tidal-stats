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

A log file is used to allow monitoring of progress for long runs.

Quick start example:
    python 03-tidal_stats.py --config config/king-sound-quick-test.yaml
    
Example parallel processing:
    python 03-tidal_stats.py --config config/king-sound-quick-test.yaml --split 2 --index 0
    python 03-tidal_stats.py --config config/king-sound-quick-test.yaml --split 2 --index 1

"""
import argparse
import os
import glob
import re
import numpy as np
import rasterio
from rasterio.transform import xy
import pandas as pd
from datetime import datetime
import logging
import warnings
import matplotlib.pyplot as plt
import sys
import threading
import time

import tide_stats_module as tide_stats_module



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
        "hat_label",
        # Metadata tags
        "author",
        "organization",
        "description",
        "reference",
        "metadata_link",
        "license"
    ]

    # Load model run parameters from the YAML config file.
    config = tide_stats_module.load_config(args.config, required_params)

    # Unpack configuration values from YAML.
    clipped_tide_model_path = config.get("clipped_tide_model_path")
    grid_path = config.get("grid_path")
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    time_step = config.get("time_step")
    working_path = config.get("working_path")
    lat_label = config.get("lat_label")
    hat_label = config.get("hat_label")

    # Configure logging
    log_file = f"{working_path}/tidal_stats_{args.index}.log"

    # Make the working directory exists for the log file
    if not os.path.exists(working_path):
        os.makedirs(working_path)

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True  # Needed because no logs are generated without this.
                    # basicConfig will not overwrite existing loggers. 
                    # This will force it to.
    )
    logging.info("Script started.")

    print(f"Logging to {log_file}")  # Print initial log info

    print("Started script with the following configuration:")
    tide_stats_module.print_config(config)

    # --- Check for resumption files with mismatched split counts ---
    partial_files = glob.glob(os.path.join(working_path, "*strip_*.tif"))
    if partial_files:
        # Extract strip indices from filenames
        existing_indices = []
        for fname in partial_files:
            m = re.search(r"strip_(\d+)", fname)
            if m:
                existing_indices.append(int(m.group(1)))
        # If any file has an index that is not valid for the current split, warn the user
        if any(idx >= args.split for idx in existing_indices):
            msg = (f"Detected resumption files with strip indices up to {max(existing_indices)} "
                f"but current '--split' value is {args.split}. "
                "This indicates that the resumption files were generated with a different split. "
                "Please delete the existing partial files or run the script with '--split' set to "
                f"{max(existing_indices)+1}.")
            raise ValueError(msg)

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

    # Adjust the transform for the slice
    from affine import Affine
    slice_transform = transform * Affine.translation(start_col, 0)

    # Prepare empty arrays for the slice
    lpt_arr = np.full((height, slice_width), np.nan, dtype=np.float32)
    hpt_arr = np.full((height, slice_width), np.nan, dtype=np.float32)
    mlws_arr = np.full((height, slice_width), np.nan, dtype=np.float32)
    mhws_arr = np.full((height, slice_width), np.nan, dtype=np.float32)

    # Later when saving the GeoTIFF files, use slice_transform:
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": slice_width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": slice_transform,
        "nodata": np.nan
    }

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
            msg = f"Resuming from existing file: {fname}"
            logging.info(msg)
            print(msg)
            with rasterio.open(fname) as src_in:
                data = src_in.read(1)
            arr_ref[:] = data

    # Load EOT20 constants
    eot20_consts = tide_stats_module.load_eot20_constants(clipped_tide_model_path)

    # List cells to process
    process_indices = []
    for row in range(height):
        for col in range(start_col, end_col):
            if grid[row, col] == 1:
                if np.isnan(lpt_arr[row, col - start_col]):
                    process_indices.append((row, col))

    total = len(process_indices)
    msg = f"Processing {total} grid cells in slice {index} (cols={start_col}:{end_col})"
    logging.info(msg)
    print(msg)

    # --------------------------------------------------------------------------
    # Timing setup: we capture the time right before we begin pixel processing
    # --------------------------------------------------------------------------
    start_time = time.time()


    # ---------------------------------------------------------------------------
    # Main loop for processing grid cells
    # ---------------------------------------------------------------------------
    count = 0
    metadata_tags = tide_stats_module.get_metadata_tags(config, "03-tidal_stats.py")

    for count, (row, col) in enumerate(process_indices, start=1):
        if stop_processing:  # If 'q' was pressed, exit cleanly
            logging.info("\nGraceful shutdown initiated...\n")
            break
        lon, lat = xy(transform, row, col, offset="center")

        # Predict tide
        tide_series = tide_stats_module.predict_tide(lat, lon, times, eot20_consts)

        # Compute stats
        lpt, hpt, mlws, mhws = tide_stats_module.compute_tidal_stats(times, tide_series, start_dt, end_dt)
        j = col - start_col
        lpt_arr[row, j] = lpt
        hpt_arr[row, j] = hpt
        mlws_arr[row, j] = mlws
        mhws_arr[row, j] = mhws

        # Debug plotting (only if --debug flag is set)
        if args.debug:
            phases = tide_stats_module.compute_moon_phases(start_dt, end_dt)
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

        # Save partial results every 50 pixels
        count += 1
        if (count % 10) == 0:
            # ---------------------------------------------
            # Timing logic for progress and estimated finish
            # ---------------------------------------------
            elapsed_seconds = time.time() - start_time
            fraction_done = count / total
            if fraction_done > 0:
                estimated_total_time = elapsed_seconds / fraction_done
                remaining_seconds = estimated_total_time - elapsed_seconds
            else:
                remaining_seconds = 0

            # Create a simple HH:MM:SS string
            def format_hms(sec):
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = int(sec % 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            msg_progress = (f"Processed {count}/{total} pixels, "
                            f"elapsed={format_hms(elapsed_seconds)}, "
                            f"remaining≈{format_hms(remaining_seconds)}")
            print(msg_progress)
            logging.info(msg_progress)

            for arr_, f_ in [
                (lpt_arr, lpt_file),
                (hpt_arr, hpt_file),
                (mlws_arr, mlws_file),
                (mhws_arr, mhws_file)
            ]:
                tide_stats_module.save_geotiff(f_, arr_, profile, metadata_tags)


    # ---------------------------------------------------------------------------
    # Final save
    # ---------------------------------------------------------------------------
    for arr_, f_ in [
        (lpt_arr, lpt_file),
        (hpt_arr, hpt_file),
        (mlws_arr, mlws_file),
        (mhws_arr, mhws_file)
    ]:
        tide_stats_module.save_geotiff(f_, arr_, profile, metadata_tags)

    if args.split>1:
        logging.info("To merge the multiple grids into the final grids run:")
        logging.info(f"python 04-merge_strips.py --split {args.split} --config {args.config}")
    logging.info(f"Tidal statistics complete for slice {args.index}. ")

    logging.shutdown()
    print("Exiting program safely.")
    sys.exit(0)  # Ensures a clean exit with status code 0

if __name__ == "__main__":
    main()
