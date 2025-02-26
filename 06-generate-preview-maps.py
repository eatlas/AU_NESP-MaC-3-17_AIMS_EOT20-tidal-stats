#!/usr/bin/env python
"""
06-generate-preview-maps.py

This script reads a configuration file (YAML) to determine file paths and other
parameters, then generates preview maps for tidal statistics products.

This script is not complete and is just a draft. The maps need to be extended to 
provide far more metadata on them and the map styling improved.
...
"""

import argparse
import os
import math
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import imageio
from datetime import datetime
import yaml
import geopandas as gpd
from shapely.geometry import box  # New import for clipping

# -------------------------------------------------------------------------
# Utility Functions

def load_config(config_path, required_params):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
    return config

def load_raster_data(file_path, all_bands=False):
    """Load raster data, bounds, and CRS from a file.
    
    Parameters:
        file_path (str): Path to the raster file.
        all_bands (bool): If True, reads all bands; otherwise, reads the first band.
    
    Returns:
        data (numpy.ndarray): The raster data.
        bounds (rasterio.coords.BoundingBox): The spatial extent.
        crs: The coordinate reference system.
        If the file does not exist, returns (None, None, None).
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None, None
    with rasterio.open(file_path) as src:
        data = src.read() if all_bands else src.read(1)
        bounds = src.bounds
        crs = src.crs
    return data, bounds, crs

def reproject_land_mask(land_mask_gdf, target_crs):
    """Reproject land mask GeoDataFrame to the target CRS if needed."""
    if land_mask_gdf.crs != target_crs:
        return land_mask_gdf.to_crs(target_crs)
    return land_mask_gdf

def clip_land_mask(land_mask_gdf, bounds):
    """Clip the land mask to the given bounds.
    
    Parameters:
        land_mask_gdf (GeoDataFrame): The reprojected land mask.
        bounds (tuple): (left, bottom, right, top) spatial extent.
    
    Returns:
        GeoDataFrame: The clipped land mask.
    """
    left, bottom, right, top = bounds
    bbox = box(left, bottom, right, top)
    return gpd.clip(land_mask_gdf, bbox)

def add_attribution(fig, config):
    attribution_text = (
        f"Derived from EOT20 Tidal Model and BOM Tide Gauge Data. Graph is licensed under {config.get('license')}\n"
        f"Processing: AIMS, {config.get('metadata_link')}, Tidal model EOT20: https://doi.org/10.17882/79489\n"
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(0.5, 0.01, attribution_text, ha="center", fontsize=8, color='grey')

def ensure_preview_dir(working_path):
    preview_dir = os.path.join(working_path, "preview")
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    return preview_dir

def overlay_land(ax, land_mask_gdf):
    """Overlay the land mask on the provided axis."""
    land_mask_gdf.plot(
        ax=ax,
        facecolor='#e5e0d7',
        edgecolor='#5a6b88',
        linewidth=1,
        alpha=0.7
    )

# -------------------------------------------------------------------------
# Define colour ramps

pos_colors = ['#fffdfc', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
              '#ef3b2c', '#cb181d', '#940d38', '#5b002d']
neg_colors = ['#081f6b', '#083e9c', '#2163b5', '#4286c6', '#6babd6',
              '#9ecae1', '#c6e4ef', '#def2f7', '#f7feff']
diverging_colors = neg_colors[1:] + ['#ffffff'] + pos_colors[1:]

pos_cmap = LinearSegmentedColormap.from_list("pos_cmap", pos_colors)
neg_cmap = LinearSegmentedColormap.from_list("neg_cmap", neg_colors)
diverging_cmap = LinearSegmentedColormap.from_list("diverging_cmap", diverging_colors)

# -------------------------------------------------------------------------
# Plot Generation Functions

def generate_single_map(file_path, title, cmap, symmetric, preview_dir, config, land_mask_gdf):
    data, bounds, raster_crs = load_raster_data(file_path)
    if data is None:
        print(f"File for {title} not found: {file_path}")
        return
    left, bottom, right, top = bounds
    land_mask_local = reproject_land_mask(land_mask_gdf, raster_crs)
    # Crop the land mask to the raster's spatial extent
    land_mask_local = clip_land_mask(land_mask_local, bounds)

    fig, ax = plt.subplots(figsize=(12, 6))
    if symmetric:
        max_abs = np.nanmax(np.abs(data))
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    else:
        norm = None

    im = ax.imshow(
        data,
        cmap=cmap,
        norm=norm,
        extent=(left, right, bottom, top),
        origin='upper'
    )
    overlay_land(ax, land_mask_local)
    ax.set_title(f"{title} Preview Map")
    # Adjust colorbar: shrink it and add units label.
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.5)
    cbar.set_label("m - MSL EOT20")
    ax.set_aspect('equal')
    ax.axis("on")
    add_attribution(fig, config)
    
    output_file = os.path.join(preview_dir, f"{title}_map.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved {output_file}")

def generate_percentiles_panel(file_path, preview_dir, config, land_mask_gdf, percentile_labels):
    title = "Percentiles"
    data, bounds, raster_crs = load_raster_data(file_path, all_bands=True)
    if data is None:
        print(f"File for {title} not found: {file_path}")
        return
    left, bottom, right, top = bounds
    land_mask_local = reproject_land_mask(land_mask_gdf, raster_crs)
    # Crop the land mask to the raster's spatial extent
    land_mask_local = clip_land_mask(land_mask_local, bounds)
    
    num_bands = data.shape[0]
    # Force layout: 3 columns by 4 rows.
    ncols = 3
    nrows = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.atleast_2d(axes)
    
    abs_max = np.nanmax(np.abs(data))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    
    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < num_bands:
            band_data = data[i, :, :]
            im = ax.imshow(
                band_data,
                cmap=diverging_cmap,
                norm=norm,
                extent=(left, right, bottom, top),
                origin='upper'
            )
            # Use the corresponding percentile label for the title.
            ax.set_title(f"Percentile {percentile_labels[i]}")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.5)
            cbar.set_label("m - MSL EOT20")
            overlay_land(ax, land_mask_local)
            ax.set_aspect('equal')
            ax.axis("on")
        else:
            ax.axis("off")

    fig.suptitle("Percentiles Preview Map", fontsize=16)
    add_attribution(fig, config)
    
    output_file = os.path.join(preview_dir, f"{title}_map.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved {output_file}")

def generate_monthly_maps(file_path, statistic, cmap, symmetric, preview_dir, config, land_mask_gdf, month_names):
    data, bounds, raster_crs = load_raster_data(file_path, all_bands=True)
    if data is None:
        print(f"File for {statistic} not found: {file_path}")
        return
    left, bottom, right, top = bounds
    land_mask_local = reproject_land_mask(land_mask_gdf, raster_crs)
    # Crop the land mask to the raster's spatial extent
    land_mask_local = clip_land_mask(land_mask_local, bounds)
    
    frames = []
    # Loop over each of the 12 bands (months)
    for month in range(12):
        month_data = data[month, :, :]
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if symmetric:
            max_abs = np.nanmax(np.abs(month_data))
            norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
        else:
            norm = None
        
        im = ax.imshow(
            month_data,
            cmap=cmap,
            norm=norm,
            extent=(left, right, bottom, top),
            origin='upper'
        )
        overlay_land(ax, land_mask_local)
        # Use full month name for title.
        month_name = month_names[month]
        ax.set_title(f"{statistic} - {month_name}")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.5)
        cbar.set_label("m - MSL EOT20")
        ax.set_aspect('equal')
        ax.axis("on")
        add_attribution(fig, config)
        
        png_filename = os.path.join(preview_dir, f"{statistic}_Month{month+1}.png")
        plt.savefig(png_filename, dpi=300)
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf, dtype=np.uint8)[..., :3]
        frames.append(image)
        
        plt.close(fig)
        print(f"Saved {png_filename}")
    
    gif_filename = os.path.join(preview_dir, f"{statistic}_map.gif")
    imageio.mimsave(gif_filename, frames, fps=2)
    print(f"Saved animated GIF {gif_filename}")

# -------------------------------------------------------------------------
# Main routine

def main():
    parser = argparse.ArgumentParser(
        description="Generate preview maps for tidal statistics products."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file containing model run parameters.")
    parser.add_argument("--index", type=int, default=0,
                        help="Index of the strip (default: 0)")
    args = parser.parse_args()
    
    required_params = [
        "start_date",
        "end_date",
        "time_step",
        "working_path",
        "output_path_prefix",
        "lat_label",
        "hat_label",
        "metadata_link",
        "license",
        "percentiles",
        "land_mask_path"
    ]
    
    config = load_config(args.config, required_params)
    working_path = config.get("working_path")
    lat_label = config.get("lat_label")
    hat_label = config.get("hat_label")
    output_path_prefix = config.get("output_path_prefix")
    config_percentiles = config.get("percentiles")
    # Create percentile labels, e.g. p05, p10, etc.
    percentile_labels = [f"p{p:02d}" for p in config_percentiles]
    # Full month names for monthly maps.
    month_names = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]

    # Read the land mask shapefile once (GeoDataFrame)
    land_mask_path = config.get("land_mask_path")
    if not os.path.exists(land_mask_path):
        raise ValueError(f"Land mask shapefile not found: {land_mask_path}")
    land_mask_gdf = gpd.read_file(land_mask_path)

    start_date = config.get("start_date")
    end_date = config.get("end_date")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    sim_years = f"{start_dt.strftime('%Y-%m')}-{end_dt.strftime('%Y-%m')}"
    
    lpt_file = f"{output_path_prefix}{lat_label}_{sim_years}.tif"
    hpt_file = f"{output_path_prefix}{hat_label}_{sim_years}.tif"
    mlws_file = f"{output_path_prefix}MLWS_{sim_years}.tif"
    mhws_file = f"{output_path_prefix}MHWS_{sim_years}.tif" 
    percentiles_file = f"{output_path_prefix}Percentiles_{sim_years}.tif"
    monthly_lpt_file = f"{output_path_prefix}Monthly_LPT_{sim_years}.tif"
    monthly_mean_file = f"{output_path_prefix}Monthly_Mean_{sim_years}.tif"
    monthly_hpt_file = f"{output_path_prefix}Monthly_HPT_{sim_years}.tif"
    
    preview_dir = ensure_preview_dir(working_path)
    
    # Generate single map previews.
    single_products = [
        {"title": "LPT", "file": lpt_file, "cmap": neg_cmap, "symmetric": False},
        {"title": "HPT", "file": hpt_file, "cmap": pos_cmap, "symmetric": False},
        {"title": "MLWS", "file": mlws_file, "cmap": neg_cmap, "symmetric": False},
        {"title": "MHWS", "file": mhws_file, "cmap": pos_cmap, "symmetric": False},
    ]
    for prod in single_products:
        generate_single_map(
            file_path=prod["file"],
            title=prod["title"],
            cmap=prod["cmap"],
            symmetric=prod["symmetric"],
            preview_dir=preview_dir,
            config=config,
            land_mask_gdf=land_mask_gdf
        )
    
    # Generate multi-panel preview for Percentiles.
    generate_percentiles_panel(
        file_path=percentiles_file,
        preview_dir=preview_dir,
        config=config,
        land_mask_gdf=land_mask_gdf,
        percentile_labels=percentile_labels
    )
    
    # Generate monthly maps and animated GIFs.
    monthly_products = [
        {"statistic": "Monthly_LPT",  "file": monthly_lpt_file,  "cmap": neg_cmap,       "symmetric": False},
        {"statistic": "Monthly_Mean", "file": monthly_mean_file, "cmap": diverging_cmap, "symmetric": True},
        {"statistic": "Monthly_HPT",  "file": monthly_hpt_file,  "cmap": pos_cmap,       "symmetric": False},
    ]
    for prod in monthly_products:
        generate_monthly_maps(
            file_path=prod["file"],
            statistic=prod["statistic"],
            cmap=prod["cmap"],
            symmetric=prod["symmetric"],
            preview_dir=preview_dir,
            config=config,
            land_mask_gdf=land_mask_gdf,
            month_names=month_names
        )

if __name__ == "__main__":
    main()
