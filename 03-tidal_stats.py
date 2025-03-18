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

REPORT_RATE = 25 # Report progress every N pixels
DEFAULT_PERCENTILES = [2, 5, 10, 20, 50, 70, 90, 95, 98]

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global flag to allow graceful termination
stop_processing = False

def quit_listener():
    """
    Waits for the user to type 'q' and press Enter to quit.
    """
    global stop_processing
    print("\nPress 'q' and Enter at any time to stop the script gracefully. Ctrl+C will sometimes lockup...\n")
    while True:
        user_input = input().strip().lower()
        if user_input == "q":
            stop_processing = True
            print("\nStopping after current process...\n")
            break



# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Calculate tidal statistics from tide-model grid (EOT20) with extended stats."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file containing model run parameters.")
    parser.add_argument("--split", type=int, default=1,
                        help="Number of vertical slices")
    parser.add_argument("--index", type=int, default=0,
                        help="0-based index of the slice to process")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug plotting for each processed pixel.")
    parser.add_argument("--enable_quit_listner", action="store_true",
                        help=("If set then script can be gracefully stopped with a key input rather than Ctrl+C, "
                        "which can cause terminal lockups. Don't use this option on HPC as there is no key input."))
    args = parser.parse_args()

    required_params = [
        "clipped_tide_model_path",
        "grid_path",
        "start_date",
        "end_date",
        "time_step",
        "working_path",
        "lat_label",
        "hat_label",
        "author",
        "organization",
        "description",
        "reference",
        "metadata_link",
        "license",
        "percentiles"
    ]
    config = tide_stats_module.load_config(args.config, required_params)
    clipped_tide_model_path = config.get("clipped_tide_model_path")
    grid_path = config.get("grid_path")
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    time_step = config.get("time_step")
    working_path = config.get("working_path")
    lat_label = config.get("lat_label")
    hat_label = config.get("hat_label")
    index = args.index

    if args.enable_quit_listner:
        # Start quit listener thread
        # The quit listener does not work in a HPC environment.
        quit_thread = threading.Thread(target=quit_listener, daemon=True)
        quit_thread.start()
    else:
        print("Use Ctrl-C to stop the script, however this might fail due to GDAL library issues.")

    log_file = f"{working_path}/tidal_stats_{args.index}.log"
    if not os.path.exists(working_path):
        os.makedirs(working_path)
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    logging.info("Script started.")
    print(f"Index {index} - Logging to {log_file}")
    # print("Started script with the following configuration:")
    # tide_stats_module.print_config(config)

    # Read time series and check if all months are covered
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    times = pd.date_range(start=start_dt, end=end_dt, freq=f"{time_step}h")
    all_months = set(range(1, 13))
    simulated_months = set(times.month)

    # Check that simulation covers all months
    all_months = set(range(1, 13))
    simulated_months = set(times.month)
    missing_months_warning = ""
    if all_months != simulated_months:
        missing = sorted(list(all_months - simulated_months))
        missing_months_warning = (
            f"WARNING: Simulation period does not cover full year. "
            f"Missing months: {missing}. Monthly statistics will be incomplete."
        )
        logging.warning(missing_months_warning)
        print(missing_months_warning)

    with rasterio.open(grid_path) as src:
        grid = src.read(1)
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

    # Determine columns for this slice
    # split = args.split
    # index = args.index
    # cols_per_slice = width // split
    # extra = width % split
    # start_col = index * cols_per_slice
    # if index == (split - 1):
    #     end_col = width
    # else:
    #     end_col = start_col + cols_per_slice
    # slice_width = end_col - start_col
    # ---- New integration: Determine column splits based on active pixels ----
    splits = tide_stats_module.split_grid_columns_by_active_pixels(
        grid, args.split)
    if index < 0 or index >= len(splits):
        raise ValueError(f"Invalid index {index}. It must be in the range 0 to {len(splits)-1}")
    start_col, end_col = splits[index]
    slice_width = end_col - start_col
    print(f"Process {index} will handle columns {start_col} to {end_col} (width {slice_width}).")
    logging.info(f"Process {index} assigned columns {start_col} to {end_col} based on active pixel counts.")

    # Prepare empty arrays for original stats
    lpt_arr = np.full((height, slice_width), np.nan, dtype=np.float32)
    hpt_arr = np.full((height, slice_width), np.nan, dtype=np.float32)
    mlws_arr = np.full((height, slice_width), np.nan, dtype=np.float32)
    mhws_arr = np.full((height, slice_width), np.nan, dtype=np.float32)

    # Prepare empty arrays for extended stats:
    # Percentiles (11 bands) and monthly stats (3 files: each 12 bands)
    
    config_percentiles = config.get("percentiles", DEFAULT_PERCENTILES)
    percentile_labels = [f"p{p:02d}" for p in config_percentiles]
    percentiles_arr = np.full((len(percentile_labels), height, slice_width), np.nan, dtype=np.float32)
    monthly_lpt_arr = np.full((12, height, slice_width), np.nan, dtype=np.float32)
    monthly_mean_arr = np.full((12, height, slice_width), np.nan, dtype=np.float32)
    monthly_hpt_arr = np.full((12, height, slice_width), np.nan, dtype=np.float32)

    from affine import Affine
    slice_transform = transform * Affine.translation(start_col, 0)

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

    sim_years = f"{start_dt.strftime('%Y-%m')}-{end_dt.strftime('%Y-%m')}"
    base = f"{sim_years}_strip_{index}.tif"
    lpt_file = os.path.join(working_path, lat_label + "_" + base)
    hpt_file = os.path.join(working_path, hat_label + "_" + base)
    mlws_file = os.path.join(working_path, "MLWS_" + base)
    mhws_file = os.path.join(working_path, "MHWS_" + base)
    # Additional output files:
    percentiles_file = os.path.join(working_path, "Percentiles_" + base)
    monthly_lpt_file = os.path.join(working_path, "Monthly_LPT_" + base)
    monthly_mean_file = os.path.join(working_path, "Monthly_Mean_" + base)
    monthly_hpt_file = os.path.join(working_path, "Monthly_HPT_" + base)

    if not os.path.exists(working_path):
        os.makedirs(working_path)

    # Resume from existing partial outputs if present
    for fname, arr_ref in zip(
        [lpt_file, hpt_file, mlws_file, mhws_file, percentiles_file, monthly_lpt_file, monthly_mean_file, monthly_hpt_file],
        [lpt_arr, hpt_arr, mlws_arr, mhws_arr, percentiles_arr, monthly_lpt_arr, monthly_mean_arr, monthly_hpt_arr]
    ):
        if os.path.exists(fname):
            msg = f"Index {index} - Resuming from existing file: {fname}"
            logging.info(msg)
            print(msg)
            with rasterio.open(fname) as src:
                data = src.read()
                # For single-band files, squeeze the band dimension
                if data.ndim == 3 and data.shape[0] == 1:
                    data = data[0]
            arr_ref[:] = data

    eot20_consts = tide_stats_module.load_eot20_constants(clipped_tide_model_path)

    process_indices = []
    for row in range(height):
        for col in range(start_col, end_col):
            if grid[row, col] == 1:
                if np.isnan(lpt_arr[row, col - start_col]):
                    process_indices.append((row, col))
    total = len(process_indices)
    msg = f"Index {index} - Processing {total} grid cells in slice {index} (cols={start_col}:{end_col})"
    logging.info(msg)
    print(msg)

    start_time = time.time()
    #metadata_tags = tide_stats_module.get_metadata_tags(config, "03-tidal_stats.py")

    for count, (row, col) in enumerate(process_indices, start=1):
        if stop_processing:
            logging.info(f"Index {index} - Graceful shutdown initiated...")
            break
        lon, lat = xy(transform, row, col, offset="center")
        tide_series = tide_stats_module.predict_tide(lat, lon, times, eot20_consts)

        # Compute extended statistics
        stats = tide_stats_module.compute_tidal_stats(
            times, tide_series, start_dt, end_dt,
            percentile_values=config.get("percentiles")
        )

        lpt = stats['lpt']
        hpt = stats['hpt']
        mlws = stats['mlws']
        mhws = stats['mhws']
        percentiles_dict = stats['percentiles']
        monthly_dict = stats['monthly']

        j = col - start_col
        lpt_arr[row, j] = lpt
        hpt_arr[row, j] = hpt
        mlws_arr[row, j] = mlws
        mhws_arr[row, j] = mhws

        for i, label in enumerate(percentile_labels):
            percentiles_arr[i, row, j] = percentiles_dict[label]

        for month in range(1, 13):
            monthly_lpt_arr[month-1, row, j] = monthly_dict[month]['lpt']
            monthly_mean_arr[month-1, row, j] = monthly_dict[month]['mean']
            monthly_hpt_arr[month-1, row, j] = monthly_dict[month]['hpt']

        if args.debug:
            plt.figure(figsize=(10, 5))
            plt.plot(times, tide_series, label="Tide Series", color="black")
            phases = tide_stats_module.compute_moon_phases(start_dt, end_dt)
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
            print(f"\nDebug Info at (row={row}, col={col}):")
            print(f"  LPT = {lpt:.3f}")
            print(f"  HPT = {hpt:.3f}")
            print(f"  MLWS = {mlws:.3f}")
            print(f"  MHWS = {mhws:.3f}")
            print("\nExtended Stats:")
            print("Percentiles:")
            for key, value in percentiles_dict.items():
                print(f"  {key}: {value:.3f}")
            print("Monthly Stats:")
            for month in range(1, 13):
                mstats = monthly_dict[month]
                print(f"  Month {month}: LPT={mstats['lpt']:.3f}, Mean={mstats['mean']:.3f}, HPT={mstats['hpt']:.3f}")
            input("Press Enter to continue to the next pixel...")
            plt.close()

        if (count % REPORT_RATE) == 0:
            elapsed_seconds = time.time() - start_time
            fraction_done = count / total
            estimated_total_time = elapsed_seconds / fraction_done if fraction_done > 0 else 0
            remaining_seconds = estimated_total_time - elapsed_seconds
            def format_hms(sec):
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = int(sec % 60)
                return f"{h:02d}:{m:02d}:{s:02d}"
            msg_progress = (f"Index {index} - Processed {count}/{total} pixels, "
                            f"elapsed={format_hms(elapsed_seconds)}, "
                            f"remaining≈{format_hms(remaining_seconds)}")
            print(msg_progress)
            logging.info(msg_progress)
            # Define a mapping for each output file to a product type
            output_products = [
                (lpt_arr, lpt_file, 1, "LPT"),
                (hpt_arr, hpt_file, 1, "HPT"),
                (mlws_arr, mlws_file, 1, "MLWS"),
                (mhws_arr, mhws_file, 1, "MHWS"),
                (percentiles_arr, percentiles_file, len(percentile_labels), "Percentiles"),
                (monthly_lpt_arr, monthly_lpt_file, 12, "Monthly_LPT"),
                (monthly_mean_arr, monthly_mean_file, 12, "Monthly_Mean"),
                (monthly_hpt_arr, monthly_hpt_file, 12, "Monthly_HPT")
            ]

            for arr_, file_path, cnt, product_type in output_products:
                profile_copy = profile.copy()
                profile_copy["count"] = cnt
                # Merge in product-specific metadata
                metadata_tags = tide_stats_module.add_product_metadata(config, product_type)
                tide_stats_module.save_geotiff(file_path, arr_, profile_copy, metadata_tags)

    # Get the common metadata tags
    #base_metadata = tide_stats_module.get_metadata_tags(config, "03-tidal_stats.py")

    # Define a mapping for each output file to a product type
    output_products = [
        (lpt_arr, lpt_file, 1, "LPT"),
        (hpt_arr, hpt_file, 1, "HPT"),
        (mlws_arr, mlws_file, 1, "MLWS"),
        (mhws_arr, mhws_file, 1, "MHWS"),
        (percentiles_arr, percentiles_file, len(percentile_labels), "Percentiles"),
        (monthly_lpt_arr, monthly_lpt_file, 12, "Monthly_LPT"),
        (monthly_mean_arr, monthly_mean_file, 12, "Monthly_Mean"),
        (monthly_hpt_arr, monthly_hpt_file, 12, "Monthly_HPT")
    ]

    for arr_, file_path, cnt, product_type in output_products:
        profile_copy = profile.copy()
        profile_copy["count"] = cnt
        # Merge in product-specific metadata
        metadata_tags = tide_stats_module.add_product_metadata(config, product_type)
        tide_stats_module.save_geotiff(file_path, arr_, profile_copy, metadata_tags)

    if args.split > 1:
        logging.info("To merge the multiple grids into the final grids run:")
        logging.info(f"python 04-merge_strips.py --split {args.split} --config {args.config}")
    logging.info(f"Tidal statistics complete for slice {args.index}.")
    
    # At the very end, before clean exit, reprint the warning (if any) in bold.
    if missing_months_warning:
        bold_warning = "*** " + missing_months_warning + " ***"
        print(bold_warning)
        logging.warning("Reprinted warning at the end: " + missing_months_warning)

    logging.shutdown()
    print(f"Index {index} - Exiting program safely.")
    sys.exit(0)


if __name__ == "__main__":
    main()
