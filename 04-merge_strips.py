#!/usr/bin/env python
"""
04-merge_strips.py

This script merges the vertical slices produced by the tidal_stats.py script into complete raster
datasets for each tidal statistic. It expects that all partial GeoTIFFs (for LPT, HPT, MLWS, and MHWS)
share the same coordinate reference system, resolution, and alignment.

Quick start example:
    python 04-merge_strips.py --config config/king-sound-quick-test.yaml
"""

import argparse
import os
import glob
import rasterio
from rasterio.merge import merge
from datetime import datetime
import util as util

def main():
    parser = argparse.ArgumentParser(
        description="Merge partial tidal statistics GeoTIFFs into complete grids."
    )

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file containing model run parameters."
    )

    parser.add_argument(
        "--split",
        type=int,
        required=False,
        default=1,
        help="Number of vertical slices that were produced",
    )

    args = parser.parse_args()

    # List of required configuration parameters.
    required_params = [
        "start_date",
        "end_date",
        "working_path",
        "output_path_prefix",
        "lat_label",
        "hat_label"
    ]

    # Load model run parameters from the YAML config file.
    config = util.load_config(args.config, required_params)

    print("Started script with the following configuration:")
    util.print_config(config)

    # Unpack configuration values from YAML.
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    working_path = config.get("working_path")
    output_path_prefix = config.get("output_path_prefix")
    lat_label = config.get("lat_label")
    hat_label = config.get("hat_label")


    # Construct simulation-year string based on provided dates.
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    sim_years = f"{start_dt.strftime('%Y-%m')}_{end_dt.strftime('%Y-%m')}"
    # In this example, we assume filenames are of the form:
    #   LPT_EOT20_<tide_label>_<sim_years>_strip_{i}.tif

    stats = [lat_label, hat_label, "MLWS", "MHWS"]

    for stat in stats:
        file_list = []
        # For each slice, look for a file matching the expected pattern.
        for i in range(args.split):
            pattern = f"{stat}_EOT20_{sim_years}_strip_{i}.tif"
            full_pattern = os.path.join(working_path, pattern)
            matches = glob.glob(full_pattern)
            if not matches:
                raise FileNotFoundError(f"Could not find file for {stat} slice {i} using pattern: {full_pattern}")
            file_list.append(matches[0])
        print(f"Merging {len(file_list)} files for {stat}...")

        # Open each file and add to a list for merging.
        src_files = [rasterio.open(f) for f in file_list]
        mosaic, out_trans = merge(src_files, method="first")
        profile = src_files[0].profile.copy()
        profile.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })
        output_file = f"{output_path_prefix}{stat}_{sim_years}.tif"
        with rasterio.open(output_file, "w", **profile) as dest:
            dest.write(mosaic)
        for src in src_files:
            src.close()
        print(f"Merged {stat} saved to {output_file}")

if __name__ == "__main__":
    main()
