Tides play a crucial role in shaping coastal and marine environments, influencing processes such as sediment transport, water clarity, and the distribution of ecosystems. In particular, the tidal range—the difference between high and low tides—has significant implications for the physical and biological dynamics of intertidal regions. For coral reefs, tidal fluctuations determine the exposure and submersion of reef structures, influencing reef zonation, biodiversity, and habitat accessibility. Accurate and detailed tidal information is therefore essential for understanding, monitoring, and managing these critical ecosystems.

Despite the importance of tidal dynamics, datasets that map key tidal statistics, such as Lowest Astronomical Tide (LAT), Highest Astronomical Tide (HAT), Mean Low Spring Water (MLSW), Mean High Spring Water (MHSW), and tidal range, remain limited in spatial resolution and regional specificity. This gap is particularly evident for Australian waters, which host diverse and globally significant coral reef systems, such as the Great Barrier Reef, and extensive intertidal regions. High-quality tidal data are essential for mapping and monitoring these environments, particularly in applications such as delineating LAT boundaries for navigation and habitat mapping, and assessing the impacts of tidal range on nearshore turbidity and satellite imagery interpretation.

To address this need, we developed a high-resolution dataset derived from the EOT20 global tidal model, tailored specifically to Australian waters. This dataset provides spatially explicit information on LAT, HAT, MLSW, MHSW, and tidal range, offering a detailed understanding of tidal dynamics across the region.

The primary driver for the development of this dataset was to assist in the mapping of coral reefs and intertidal zones, particularly in areas where tidal range plays a significant role in modulating visibility and habitat exposure. For instance, understanding the relationship between tidal range and satellite-derived waterline positions can guide the use of all-tide or low-tide composite imagery for detecting LAT. This distinction is critical in regions with high tidal ranges, where the disparity between Mean Sea Level (MSL) and LAT can introduce significant errors in mapping efforts.

Methods:

The high‐resolution tidal dataset was generated using a suite of Python scripts that integrate the EOT20 global tidal model with robust geospatial processing routines. The overall methodology consists of three primary stages: the creation of a processing grid (Tide‐Model-Grid), the computation of tidal statistics (Tidal-Stats), and the subsequent merging of spatial subsets (Merge-Strips). These stages work in concert to produce spatially explicit estimates of tidal parameters—including estimates of the Lowest and Highest Astronomical Tides (LAT/HAT) or their predicted counterparts (LPT/HPT), as well as Mean Low Water Springs (MLWS) and Mean High Water Springs (MHWS)—tailored to the coastal and intertidal environments of Australian waters.

Overview and Workflow

The overall goal of the workflow is to generate high-resolution tidal datasets for Australian coastal and intertidal zones using the EOT20 global tidal model. The process is broken down into three modular scripts that each play a specific role in the workflow:

Tide-Model-Grid Script (02-tide_model_grid.py):
This script sets the stage by generating a spatial grid (output as a GeoTIFF) that identifies which grid cells require tidal processing. Using a user-defined bounding box and cell resolution (default 1/8°). This grid acts as a mask for where the tidal modelling should be performed. The EOT20 tidal constituents do not provides over land and due to the large grid size a significant portion of intertidal areas have no tidal constituents available. To overcome this limitation we ensure that the Tide-Model-Grid includes one extra pixel overlap with land and in the Tidal-Stats script we use the extrapolation capabilities of the pyTMD library to calculate the tides in these locations. 

To improve the performance of the modelling this script also creates a clipped version of the EOT20 tidal constituents to match the user-defined bounding box.

Tidal-Stats Script (03-tidal_stats.py):
Building on the grid created by the first script, this script performs the core tidal simulations and computes statistics for each grid cell that is flagged for processing. Using the pyTMD library (or a placeholder function in our example), the script simulates tidal elevations over a specified time period (with a default time step of 0.5 hours) and extracts key statistics such as the lowest and highest predicted tides (LPT/HPT) and mean low and high water springs (MLWS/MHWS). To handle high-resolution data efficiently, the work is divided into vertical slices (parallelizable via user-specified split and index parameters), and the script supports restart capability by periodically saving intermediate results.

Merge-Strips Script (04-merge_strips.py):
After the tidal statistics have been computed for each vertical slice, this final script merges the separate GeoTIFF outputs into complete, contiguous raster datasets for each tidal statistic. The merging process assumes that all partial outputs share the same coordinate reference system, resolution, and alignment, and the final merged files include all the necessary metadata for further analysis or dissemination.

How They Connect

Sequential Processing:
The workflow starts with tide_model_grid.py to define the processing area. The generated grid informs the tidal_stats.py script about which cells to compute tidal statistics for. Finally, once all grid slices are processed, merge_strips.py combines the partial outputs into a unified dataset.

Modularity for Flexibility:
Each script is designed to operate independently yet collectively contributes to the overall task. This modular approach makes it easier to update or enhance individual components (for example, by replacing the placeholder tidal prediction function with a full pyTMD implementation) without having to rework the entire workflow.

Parallel Processing and Restart Capability:
The tidal simulation and computation script splits the workload into vertical slices, enabling parallel processing across multiple processes or machines. Intermediate saving ensures that if a process is interrupted, it can be restarted without losing significant progress, which is crucial when dealing with high-resolution, computationally intensive datasets.




# Initial Prompts
The following is a record of the development of the Python scripts

Create a series of Python scripts that generates a set of high-resolution raster datasets for the Australian region based on the EOT20 global tidal model. The goal is to calculate an approximation to the Lowest Astronomical Tide (LAT), Highest Astronomical Tide (HAT), Mean Low Spring Water (MLSW), and Mean High Spring Water (MHSW) using the EOT20 model for a given bounding box and grid resolution. 

Coordinate Reference System (CRS) and gridding issues:
The EOT20 dataset uses EPSG:4326 with an 1/8 grid with grid centres aligned with multiples of 1/8 degree. The EOT20 tidal constituents grid does not have values in locations where the grid cell overlaps land. This means that parts of complex coastlines will have nodata values for the tidal constituents. In these areas we will rely on the extrapolation available through the pyTMD.

Tide-Model-Grid Script
This script should initially create a processing grid, (if it has not been previous made by a previous run of the script). This grid (land_grid) will serve two purposes: show where the tidal statistics need to be processed, and to act as a mask for locations, such as land, where the processing should be skipped. 
This part of the processing should use the Australian land and islands shapefile, to exclude grid cells that overlap with land, with one cavet. We want to ensure that the calculated tidal statistics completely overlap the intertidal zone and so we wish to calculate one pixel of tidal values over land to ensure the nearshore land areas are covered. To achieve this the land shapefile should have a negative buffer (equal to the grid resolution) applied to it, prior to being used to mask out land areas that shouldn't be calculated. The Tide-Model-Grid should be saved as a GeoTiff (with LZW compression) to allow it to be checked. It will also be input for the Tidal-Stats script. By having this grid calculated at the start saves it from needed to be recomputed by the Tidal-Stats process. 


This should support the following command line arguments:
--land_mask_path: Path to the land mask shapefile (default: in-data-3p/AU_AIMS_Coastline_50k_2024/AU_NESP-MaC-3-17_AIMS_Aus-Coastline-50k_2024_V1-1_simp.shp)
--grid_bbox min_lon min_lat max_lon max_lat: Bounding box of the model grid (default: 96 E,-8 N to 169 E,-45 N)
--grid_cell_size: spacing of the grid cells in degrees. (Default 1/8 degree)
--land_overlap: number of pixels to extrapolate over land areas to ensure intertidals areas are covered. (integer number of grid cells). This is used as the negative buffer applied to the land_mask shapefile.
--output_tide_model_grid: path and filename of the output GeoTiff grid. (default: working/AU_AIMS_EOT20-model-grid.tif)


Tidal-Stats script
This script will calculate tidal statistics for locations specified by the grid created by the Tide-Model-Grid.
The script should:
1. Use the pyTMD library along with the EOT20 dataset constituents to predict tidal elevations (user specifiable start and end date, with a default of 1 Jan 2024, 31 Dec 2024) to calculate the required statistics. The script should use a time step of 0.5 hours (also specifiable by user).
2. Generate raster datasets for LAT (Lowest Astronomical Tide), HAT (Highest Astronomical Tide), Mean Low Water Springs (MLWS), and Mean High Water Springs (MHWS), where each pixel represents the calculated value for the respective statistic at the resolution of the EOT20 model. Given that we will not necessarily simulate for 19 years to capture the full lunar nodal cycle, the LAT and HAT will only be estimates over the simulation period. For this reason the names should probably be Lowest Predicted Tide (LPT) and Highest Predicted Tide (HPT). If we simulate over a 19 year period these should be called LAT and HAT. Use LPT and HPT for internal variable names. 
3. The calculations should be for the grid locations from the Tide-Model-Grid.
4. Save the final raster datasets in a standard geospatial format (e.g., GeoTIFF) with appropriate metadata, including the coordinate reference system (CRS), spatial resolution, and data source. Each statistic should be stored in a separate file.

The script should provide feedback to the user to allow an understanding of progress. This might involve indicating the progress such as processing 243 of 40800 pixels, then when the pixel has been calculate it should output the lat, long and summary stats.

Command line arguments and parallel processing
The script should allow the path to the EOT20 model, and output directory to be specified. These should all have defaults based on information provided. 

Ensure the script is designed for efficiency, as the high resolution and extended time simulation may require significant computational resources. To allow for simple parallelism the script should provide two command line arguments:
--split N - Split the work into N parts
--index M - Which part (1 of N) should this process run.
To allow the processes to run independently the progressing grid should be split into N equal slices (vertical strips). So for example if split is 4 then the Tide-Model-Grid will be split into 4 equal vertical slices. If the full grid is 1000 x 500 pixels (width x height) then task 0 (--index 0) should process a grid 250 pixels x 500 pixels, the left most quarter of the grid. --index 1 will process the left of middle quarter, --index 2 the right of middle quarter and --index 3 the right most quarter. Each process saves to separate files that cover a subset of the full grid and with the index in the name to indicate the relevant index part. The final merge grid script will combine these strips back to the full grid.
--path_to_tide_models Path to the tidal constituent files. Default: in-data-3p/World_EOT20_2021 within that folder we have the following:
EOT20: <path_to_tide_models>/EOT20/ocean_tides/
EOT20_load: <path_to_tide_models>/EOT20/load_tides/
--tide_model_grid: path to the grid of locations to perform the tidal statics on, default: working/AU_AIMS_EOT20-model-grid.tif
--start-date: Start of the simulation
--end-date: End of the simulation
--working_path: Path to the working folder that will save the partial grid calculations. default: working

Script restarting
The script should also allow resumption of the processing, so that if the script is interrupted it can be restarted and not much processing will be lost. This could be achieved by having GeoTiffs for each of the tide statistics that are progressively saved. i.e. the grid starts as no-data values, then as pixels are calculated the raster is saved to disk occasionally (say every 100th pixel calculated). When the script starts it could look for these intermediate rasters, then skip through any pixels that have already been processed. If a grid cell is no-data and the land_grid indicates that the cell should be processed. These intermediate grids should be saved in a 'working' directory and should be in GeoTiff format. They should have a default name of EOT20_{Statistic name}_{Start year}_{End year}.tif.

Merge-Strips script:
This script should merge the grid subsets generated by Tidal-Stats script (specified by --split). It should combine them back into the full final grid. It will be run manually once all the input grids have been calculated. Should have an option to specify the. 
--split: Number of parts of the processing was split into
--working_path: Path to the input files
--start-date: Start of the simulation used for the Tidal-Stats. Used by this script to know the filenames in the working_path.
--end-date: End of the simulation
--output: output file name path and prefix (default: public/AU_AIMS_EOT20-Tide-Stats)

Questions:
How exactly should MLWS and MHWS be computed? - Take a time series average of identified low/high tide events around around full and new moon dates. Estimate these dates by starting with a reference new moon date and the average length of the lunar synodic month.

For LAT/HAT versus LPT/HPT naming: Do we want a parameter (or flag) that switches the output names based on the simulation period (e.g., if running a full 19 year simulation, use “LAT/HAT”, otherwise “LPT/HPT”)? - Have a command line argument that allows the switching of the names used in the working files names 
 
Which tidal constituents from the EOT20 dataset are to be used in the simulation? Should all available constituents be used, or only a subset (e.g., M2, S2, K1, O1, etc.)?
Use all constituents, or what ever the default is when using pyTMD

The specification notes that the grid centers should align with multiples of the grid cell size (default 1/8°). Should this alignment be enforced for any provided resolution, or only for the default 1/8° grid
The output grid does not need to align with the EOT20 grid. Interpolation or extrapolation using pyTMD should be used to calculate the tide values at the locations of the specified grid.

The negative buffer (land_overlap) is applied to the land shapefile. Given that the data are in EPSG:4326, do we assume the buffer is in degrees? - Yes the negative buffer should be in EPSG:4326 because the underlying EOT20 grid is in that same CRS.

Should the buffering be applied in a robust way (using, for example, a geospatial library like Shapely/GEOS) given the potentially complex geometry of the coastline?
Yes a robust approach should be used. The coastline dataset is far more detailed than the tidal modelling. We want to ensure that small island features, when a large negative buffer is applied to them the feature disappears. We want to ensure that aliasing doesn't occur where the tidal model grid pixels are excluded by a tiny island in the middle of a pixel.

No-Data Conventions: What value should be used to represent “no data” in the output GeoTIFFs? - The output values should be saved as 32bit float values because they are small floating point numbers (like -1.45 m) and to the normal no data value for 32 bit float should be used.

Intermediate Saving and Restart Capability: The specification mentions saving progress every 100 pixels. Should this interval be fixed or configurable? - This should be a constant in the script, not a command line argument.

Intermediate Saving and Restart Capability: Do we need to maintain any additional metadata (e.g., a progress log or checkpoint file) to facilitate restarting, or is checking the GeoTIFF for “no data” sufficient? - The no data value in the GeoTiff files should be sufficient, when combined with checking the Tide-Model-Grid for if it is a cell that needs to be calculated.

The work is to be split into vertical slices. If the grid dimensions aren’t perfectly divisible by the number of splits (e.g., width not evenly divisible by N), how should we handle the extra columns? - The last split width should be expanded to include any remainder.

For clarity, should the --index parameter be 0-based (0 to N-1) or 1-based (1 to N)? - It should be 0-based.

Output Naming: How should the output files be named for each split? For instance, should the index be appended to the file name (e.g., “..._strip_0.tif”, “..._strip_1.tif”, etc.)? - That seems reasonable. Both the Tidal-Stats and Merge-Strips scripts will need to follow the same convention.

Default Paths: Are the default paths for the land mask and tide model constituents final, or should they be parameterized more flexibly? - These are fixed to align with a download script that downloads and setups these input files.

Output Metadata: What metadata (e.g., simulation start/end dates, time step, grid resolution, CRS) should be embedded in the GeoTIFFs? - As much metadata as possible should be added to the GeoTiff files.

Merge-Strips Script Options: For the final merge, do we assume that all partial GeoTIFFs have identical CRS, resolution, and alignment? - Yes because they will be created by the Tidal-Stats script.

User Feedback: What level of detail is expected in progress reporting? For example, is a simple print statement sufficient, or do we want to integrate a progress bar (e.g., using tqdm)? - Please use a progress bar. Ideally we want to have progress updates every 1 - 20 sec and so a progress bar will result in less screen clutter.

Error Handling: How should the scripts handle errors (e.g., file I/O issues, missing constituents)? Should they log and continue, or abort processing? - Errors should stop the script with a detailed message.