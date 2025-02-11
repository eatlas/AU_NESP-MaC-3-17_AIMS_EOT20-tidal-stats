#!/usr/bin/env python
"""
04-merge_strips.py

This script merges the vertical slices produced by the 03-tidal_stats.py script into complete raster
datasets for each tidal statistic. It expects that all partial GeoTIFFs (for LPT, HPT, MLWS, and MHWS)
share the same coordinate reference system, resolution, and alignment.

Model-run parameters are read from a YAML configuration file, but the split parameter is required 
to know how many vertical slices were produced.

This script doesn't have any parallelisation because it is unnecessary because the processing time 
is minimal and the algorithm only works on a single process. 

Quick start example:
    python 04-merge_strips.py --config config/king-sound-quick-test.yaml
"""

import argparse
import os
import glob
import rasterio
from rasterio.merge import merge
from datetime import datetime
import tide_stats_module as tide_stats_module

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

    print("Started script with the following configuration:")
    tide_stats_module.print_config(config)

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

    # Craft the metadata for the final GeoTiff images
    metadata_tags = tide_stats_module.get_metadata_tags(config, "04-merge_strips.py")

    # Generate a GeoTiff for each statistic
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

        output_file = f"{output_path_prefix}{stat}_{sim_years}.tif"
        # If only one slice exists, directly copy the file without merging
        if len(file_list) == 1:
            single_file = file_list[0]
            print(f"Only one slice found for {stat}, copying {single_file} to {output_file}")

            with rasterio.open(single_file) as src:
                data = src.read(1)  # Read the first band
                profile = src.profile.copy()
            tide_stats_module.save_geotiff(output_file, data, profile, metadata_tags)
        else:
            # Open each file and add to a list for merging.
            src_files = [rasterio.open(f) for f in file_list]
            print(src_files)
            # Print details of each source file
            for idx, src in enumerate(src_files):
                print(f"Source File {idx+1}: {file_list[idx]}")
                print(f"  Width: {src.width}, Height: {src.height}")
                print(f"  Bounding Box: {src.bounds}")  # Bounding box of the raster
                print(f"  CRS: {src.crs}")  # Coordinate Reference System
            mosaic, out_trans = merge(src_files, method="first")
            profile = src_files[0].profile.copy()
            profile.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw"
            })
            tide_stats_module.save_geotiff(output_file, mosaic[0], profile, metadata_tags)

            for src in src_files:
                src.close()
        print(f"Merged {stat} saved to {output_file}")

if __name__ == "__main__":
    main()
