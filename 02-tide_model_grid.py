#!/usr/bin/env python
"""
02-tide_model_grid.py

This script creates a processing grid to perform the tidal modelling on for a specified region
and clips the tide model constituent NetCDF files to the same bounding box with a buffer.
All model-run parameters are read from a YAML configuration file.

Quick start example:
    python 02-tide_model_grid.py --config config/king-sound-test.yaml
"""

import argparse
import os
import warnings

import tide_stats_module as tide_stats_module

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import geopandas as gpd

from shapely.ops import unary_union


def main():
    # Set up the argument parser with a config file option.
    parser = argparse.ArgumentParser(
        description="Create Tide-Model-Grid from a YAML configuration file and clip tide model files. See config directory for examples."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file containing model run parameters."
    )

    args = parser.parse_args()

    # List of required configuration parameters.
    required_params = [
        "land_mask_path",
        "grid_bbox",
        "grid_cell_size",
        "land_overlap_px",
        "grid_path",
        "tide_model_path",
        "clipped_tide_model_path",
        "clipping_buffer_deg",
    ]

    # Load model run parameters from the YAML config file.
    config = tide_stats_module.load_config(args.config, required_params)

    print("Started script with the following configuration:")
    tide_stats_module.print_config(config)

    # Unpack configuration values from YAML.
    land_mask_path = config.get("land_mask_path")
    grid_bbox = config.get("grid_bbox")
    grid_cell_size = config.get("grid_cell_size")
    land_overlap_px = config.get("land_overlap_px")
    grid_path = config.get("grid_path")
    tide_model_path = config.get("tide_model_path")
    clipped_tide_model_path = config.get("clipped_tide_model_path")
    clipping_buffer_deg = config.get("clipping_buffer_deg")

    # Ensure the output directory for the grid exists.
    out_dir = os.path.dirname(grid_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    min_lon, min_lat, max_lon, max_lat = grid_bbox

    # ---------------------------------------------------------------------
    # 1) Create the tide-model grid GeoTIFF
    # ---------------------------------------------------------------------
    print("Creating base grid...")
    width = int(np.ceil((max_lon - min_lon) / grid_cell_size))
    height = int(np.ceil((max_lat - min_lat) / grid_cell_size))
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Calculated width={width} or height={height} is non-positive. "
            f"Check that min_lat < max_lat and min_lon < max_lon."
        )

    transform = from_origin(min_lon, max_lat, grid_cell_size, grid_cell_size)

    # Read the land mask shapefile (reproject if necessary).
    print("Reading land mask...")
    gdf = gpd.read_file(land_mask_path)
    if gdf.crs.to_string() != "EPSG:4326":
        print("Reprojecting shapefile to EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")

    # Remove features that are significantly smaller than the grid size.
    # Both the grid and the shapefile are in geographic CRS, which has
    # units of degrees. Normally area in degrees^2 makes no sense, but in 
    # this case it is what we want to ensure we are comparing with the 
    # grid sizes.
    # Suppress geopandas warnings about area calculation and buffer in a geographic CRS
    # Suppress only the specific warning related to geographic CRS area calculations
    print("Removing features much smaller than the grid")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")
        gdf = gdf[gdf.geometry.area >= grid_cell_size ** 2 / 1e3]
        simp_tolerance = grid_cell_size / 100
        print(f"Simplifying geometries with tolerance = {simp_tolerance} degrees.")
        gdf["geometry"] = gdf["geometry"].simplify(simp_tolerance)

        # Apply negative buffer to expand the water area. We do this to ensure that
        # river mouths and inshore intertidal areas get covered by the tidal modelling.
        buffer_distance = -grid_cell_size * land_overlap_px
        print(f"Applying negative buffer of {buffer_distance} degrees to land geometries...")
        gdf["geometry"] = gdf["geometry"].buffer(buffer_distance)

    # Merge geometries and rasterize the buffered land mask.
    print("Merging land geometries...")
    land_geom = unary_union(gdf["geometry"])
    print("Rasterizing the land mask...")
    land_mask = rasterize(
        [(land_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # Invert mask: water (0) becomes 1 in the processing grid.
    tide_model_grid = 1 - land_mask

    # Save the tide-model grid as a GeoTIFF.
    print(f"Saving grid to {grid_path}...")
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(grid_path, "w", **profile) as dst:
        dst.write(tide_model_grid, 1)
    print(f"Tide-model grid saved to {grid_path}")

    # ---------------------------------------------------------------------
    # 2) Clip the tide model constituent files.
    # ---------------------------------------------------------------------
    tide_stats_module.clip_eot20_files(
        tide_model_path=tide_model_path,
        clipped_tide_model_path=clipped_tide_model_path,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        clipping_buffer_deg=clipping_buffer_deg
    )


if __name__ == "__main__":
    main()
