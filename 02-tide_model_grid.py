#!/usr/bin/env python
"""
02-tide_model_grid.py

This script creates a processing grid to perform the tidal modelling on for a specified region
and clips the tide model constituent NetCDF files to the same bounding box with a buffer.
All model-run parameters are read from a YAML configuration file.
"""

import argparse
import os
import re
import warnings
import pathlib
from sympy import Array
import util as util

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import geopandas as gpd
import xarray as xr
from shapely.ops import unary_union


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
    config = util.load_config(args.config, required_params)

    print("Started script with the following configuration:")
    util.print_config(config)

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

    # Remove small features and simplify geometries.
    print("Removing features smaller than 1e-5 deg^2...")
    gdf = gdf[gdf.geometry.area >= 1e-5]
    simp_tolerance = grid_cell_size / 100
    print(f"Simplifying geometries with tolerance = {simp_tolerance} degrees.")
    gdf["geometry"] = gdf["geometry"].simplify(simp_tolerance)

    # Apply negative buffer to expand the water area.
    buffer_distance = -grid_cell_size * land_overlap_px
    print(f"Applying negative buffer of {buffer_distance} degrees to land geometries...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry is in a geographic CRS.*")
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
    clip_eot20_files(
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
