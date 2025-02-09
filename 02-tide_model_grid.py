#!/usr/bin/env python
"""
02-tide_model_grid.py

This script creates a processing grid (the “tide‐model grid”) for the Australian region
and also clips the EOT20 tidal constituents NetCDF files to the same bounding box.

It:
- Creates a grid GeoTIFF mask indicating which cells to process for tidal modelling.
- Optionally buffers over land by adding an overlap in grid cells.
- Clips the EOT20 model NetCDF files to the resulting bounding box + an optional buffer
  to handle edge effects and extrapolation.

Adapted in part from:
Bishop-Taylor, R., Sagar, S., Phillips, C., & Newey, V. (2024). eo-tides: Tide modelling 
tools for large-scale satellite earth observation analysis. 
https://github.com/GeoscienceAustralia/eo-tides

Example:
python 03-tidal_stats.py --start-date 2025-01-01 --end-date 2025-02-01 --time_step 0.5 --tide_model_grid working/EOT20-King-sound/grid.tif --tide_model_grid working/EOT20-King-sound
"""

import argparse
import os
import warnings
import pathlib

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import geopandas as gpd
import xarray as xr
from shapely.ops import unary_union


def _clip_eot20_model(
    ds: xr.Dataset,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
):
    """
    Clip an EOT20 tide model xarray.Dataset to a specified bounding box.

    Assumes that the dataset has dimensions and coords:
        - ds.coords["lon"] in [0..360]
        - ds.coords["lat"] in descending order (typical for global tide models).

    This approach is adapted from logic used in the eo-tides library for
    clipping NetCDF data that may span 0..360 longitudes.  
    """

    # Convert the data arrays into memory
    lon_vals = ds["lon"].compute()
    lat_vals = ds["lat"].compute()

    # Convert the input bounding box into 0..360 range if needed:
    # If the bounding box is negative or crosses 180/360, handle appropriately
    def wrap_to_360(x):
        return x % 360

    left_360 = wrap_to_360(min_lon)
    right_360 = wrap_to_360(max_lon)

    # Because the model is in 0..360, we must handle 2 cases:
    # 1) If left <= right, we can slice directly in that range
    # 2) If left > right, the bounding box crosses 0 meridian in the 360 sense,
    #    so we combine two slices
    lat_slice = slice(max_lat, min_lat) if lat_vals[0] > lat_vals[-1] else slice(min_lat, max_lat)

    if left_360 <= right_360:
        # Direct clip
        ds_clipped = ds.sel(
            lon=slice(left_360, right_360),
            lat=lat_slice
        )
    else:
        # BBox crosses 0 in 360 space, so break into two slices and then merge
        ds_left = ds.sel(
            lon=slice(left_360, 360),
            lat=lat_slice
        )
        ds_right = ds.sel(
            lon=slice(0, right_360),
            lat=lat_slice
        )
        ds_clipped = xr.concat([ds_left, ds_right], dim="lon")

    return ds_clipped


def clip_eot20_files(
    input_directory: str,
    output_directory: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    buffer: float = 5.0,
    overwrite: bool = False,
):
    """
    Clips EOT20 NetCDF files found in `{input_directory}/ocean_tides` to the bounding box
    [min_lon, min_lat, max_lon, max_lat], adding `buffer` degrees on each side to minimize
    boundary effects. Files are saved to `output_directory`.

    Parameters:
    ----------
    input_directory: str
        Path to the parent directory containing the `ocean_tides` folder.
    output_directory: str
        Path to the output directory where clipped NetCDF files will be saved.
    min_lon, min_lat, max_lon, max_lat: float
        Bounding box for clipping in degrees.
    buffer: float, optional
        Additional buffer (in degrees) applied to the bounding box.
    overwrite: bool, optional
        If True, overwrite existing files in the output directory.
    """
    in_dir = pathlib.Path(input_directory) / "ocean_tides"
    out_dir = pathlib.Path(output_directory) / "ocean_tides"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply buffer to bounding box
    min_lon_b = min_lon - buffer
    max_lon_b = max_lon + buffer
    min_lat_b = min_lat - buffer
    max_lat_b = max_lat + buffer

    # Identify all .nc files in the "ocean_tides" subfolder
    nc_files = list(in_dir.glob("*.nc"))

    if not nc_files:
        print(f"No NetCDF (.nc) files found in {in_dir} to clip.")
        return

    print(f"\nClipping EOT20 model files from {in_dir} to bounding box "
          f"({min_lon_b}, {min_lat_b}, {max_lon_b}, {max_lat_b})...")

    for nc_file in nc_files:
        out_file = out_dir / nc_file.relative_to(in_dir)
        if out_file.exists() and not overwrite:
            print(f"Skipping existing file: {out_file}")
            continue

        # Ensure parent directory for out_file
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Load data, clip, and save
        try:
            with xr.open_dataset(nc_file, engine="netcdf4") as ds:
                ds_clipped = _clip_eot20_model(ds, min_lon_b, min_lat_b, max_lon_b, max_lat_b)
                ds_clipped.to_netcdf(out_file, mode="w")
            print(f"  - Clipped {nc_file.name} -> {out_file}")
        except Exception as e:
            print(f"Error processing file {nc_file}: {e}")

    print("Finished clipping EOT20 files.\n")



def main():
    DEFAULT_GRID = 1 / 8
    parser = argparse.ArgumentParser(
        description="Create Tide-Model-Grid from bounding box and land mask, "
                    "and clip EOT20 model files."
    )
    parser.add_argument(
        "--land_mask_path",
        type=str,
        default="in-data-3p/ne_10m_land/ne_10m_land.shp",
        help="Path to the land mask shapefile",
    )
    parser.add_argument(
        "--grid_bbox",
        type=float,
        nargs=4,
        default=[
            96 - (DEFAULT_GRID / 2),
            -45 - (DEFAULT_GRID / 2),
            169 - (DEFAULT_GRID / 2),
            -8 - (DEFAULT_GRID / 2),
        ],  # (min_lon, min_lat, max_lon, max_lat)
        metavar=("min_lon", "min_lat", "max_lon", "max_lat"),
        help="Bounding box of the model grid in degrees (min_lon min_lat max_lon max_lat)",
    )
    parser.add_argument(
        "--grid_cell_size",
        type=float,
        default=DEFAULT_GRID,
        help="Spacing of grid cells in degrees (default 1/8 degree)",
    )
    parser.add_argument(
        "--land_overlap",
        type=int,
        default=1,
        help="Number of grid cells to use for negative buffering of the land mask",
    )
    parser.add_argument(
        "--output_tide_model_grid",
        type=str,
        default="working/EOT20-clipped/AU-grid.tif",
        help="Output filename for the tide model grid GeoTIFF",
    )
    parser.add_argument(
        "--path_to_tide_models",
        default="in-data-3p/World_EOT20_2021",
        help="Path to parent folder containing EOT20/ocean_tides/*.nc",
    )
    parser.add_argument(
        "--path_to_clipped_tide_models",
        default="working/EOT20-clipped",
        help="Path to folder where clipped tidal constituents will be saved",
    )
    parser.add_argument(
        "--clip_buffer",
        type=float,
        default=5.0,
        help="Buffer (in degrees) added around bounding box when clipping EOT20 NetCDF files",
    )
    parser.add_argument(
        "--overwrite_clips",
        action="store_true",
        help="If set, overwrite existing clipped netCDF files",
    )

    args = parser.parse_args()
    print("Started script")

    # Ensure the output directory exists
    out_dir = os.path.dirname(args.output_tide_model_grid)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    min_lon, min_lat, max_lon, max_lat = args.grid_bbox
    cell_size = args.grid_cell_size

    # ---------------------------------------------------------------------
    # 1) Create the tide-model grid GeoTIFF
    # ---------------------------------------------------------------------
    print("Creating base grid...")
    width = int(np.ceil((max_lon - min_lon) / cell_size))
    height = int(np.ceil((max_lat - min_lat) / cell_size))
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Calculated width={width} or height={height} is non-positive. "
            f"Check that min_lat < max_lat and min_lon < max_lon."
        )

    transform = from_origin(min_lon, max_lat, cell_size, cell_size)

    # Read the land mask shapefile (reproject if necessary)
    print("Reading land mask...")
    gdf = gpd.read_file(args.land_mask_path)
    if gdf.crs.to_string() != "EPSG:4326":
        print("Reprojecting shapefile to EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")

    # 1) Remove features whose area is < 1e-5 degrees^2
    print("Removing features smaller than 1e-5 deg^2...")
    gdf = gdf[gdf.geometry.area >= 1e-5]

    # 2) Simplify before buffering
    simp_tolerance = cell_size / 100
    print(f"Simplifying geometries with tolerance = {simp_tolerance} degrees.")
    gdf["geometry"] = gdf["geometry"].simplify(simp_tolerance)

    # 3) Apply negative buffer (in degrees)
    buffer_distance = -cell_size * args.land_overlap
    print(f"Applying negative buffer of {buffer_distance} degrees to land geometries...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry is in a geographic CRS.*")
        gdf["geometry"] = gdf["geometry"].buffer(buffer_distance)

    # Merge all buffered geometries into one
    print("Merging land geometries...")
    land_geom = unary_union(gdf["geometry"])

    # Rasterize the buffered land geometry
    print("Rasterizing the land mask...")
    land_mask = rasterize(
        [(land_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # Invert the mask: 1 => land => skip, 0 => water => process
    tide_model_grid = 1 - land_mask

    # Save the tide-model grid as a GeoTIFF
    print(f"Saving grid to {args.output_tide_model_grid}...")
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
    with rasterio.open(args.output_tide_model_grid, "w", **profile) as dst:
        dst.write(tide_model_grid, 1)

    print(f"Tide-model grid saved to {args.output_tide_model_grid}")

    # ---------------------------------------------------------------------
    # 2) Clip the EOT20 NetCDF files
    # ---------------------------------------------------------------------
    clip_eot20_files(
        input_directory=args.path_to_tide_models,
        output_directory=args.path_to_clipped_tide_models,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        buffer=args.clip_buffer,
        overwrite=args.overwrite_clips,
    )


if __name__ == "__main__":
    main()
