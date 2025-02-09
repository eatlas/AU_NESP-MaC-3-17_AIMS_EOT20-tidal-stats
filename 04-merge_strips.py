#!/usr/bin/env python
"""
merge_strips.py

This script merges the vertical slices produced by the tidal_stats.py script into complete raster
datasets for each tidal statistic. It expects that all partial GeoTIFFs (for LPT, HPT, MLWS, and MHWS)
share the same coordinate reference system, resolution, and alignment.
"""

import argparse
import os
import glob
import rasterio
from rasterio.merge import merge

def main():
    parser = argparse.ArgumentParser(
        description="Merge partial tidal statistics GeoTIFFs into complete grids."
    )
    parser.add_argument(
        "--split",
        type=int,
        required=True,
        help="Number of vertical slices that were produced",
    )
    parser.add_argument(
        "--working_path",
        type=str,
        default="working",
        help="Path to the input partial GeoTIFF files",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Simulation start date used for tidal_stats (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Simulation end date used for tidal_stats (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="public/AU_AIMS_EOT20-Tide-Stats",
        help="Output file name prefix",
    )
    args = parser.parse_args()

    # Construct simulation-year string based on provided dates.
    sim_years = f"{args.start_date[:4]}_{args.end-date[:4]}" if len(args.start-date) >= 4 else f"{args.start_date}_{args.end-date}"
    # In this example, we assume filenames are of the form:
    #   LPT_EOT20_<tide_label>_<sim_years>_strip_{i}.tif
    stats = ["LPT", "HPT", "MLWS", "MHWS"]

    for stat in stats:
        file_list = []
        # For each slice, look for a file matching the expected pattern.
        for i in range(args.split):
            pattern = f"{stat}_EOT20_*_{sim_years}_strip_{i}.tif"
            full_pattern = os.path.join(args.working_path, pattern)
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
        output_file = os.path.join(args.working_path, f"{args.output}_{stat}_{sim_years}.tif")
        with rasterio.open(output_file, "w", **profile) as dest:
            dest.write(mosaic)
        for src in src_files:
            src.close()
        print(f"Merged {stat} saved to {output_file}")

if __name__ == "__main__":
    main()
