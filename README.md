# EOT20-tidal-stats: Tidal statistics for Australia - Processing scripts
This repository provides a record of the scripts used in the development of the AU_NESP-MaC-3-17_AIMS_EOT20-tidal-stats dataset. More details are provided in the dataset metadata:

Tidal Statistics for Australia (Tidal range, LAT, HAT, MLWS, MHWS) derived from the EOT20 tidal model (NESP MaC 3.17, AIMS) (V1) [Data set]. eAtlas. https://doi.org/10.26274/z8b6-zx94

## Introduction

Tides play a crucial role in shaping coastal and marine environments, influencing processes such as sediment transport, water clarity, and the distribution of ecosystems. In particular, the tidal range—the difference between high and low tides—has significant implications for the physical and biological dynamics of intertidal regions. For coral reefs, tidal fluctuations determine the exposure and submersion of reef structures, influencing reef zonation, biodiversity, and habitat accessibility. Accurate and detailed tidal information is therefore essential for understanding, monitoring, and managing these critical ecosystems.

Despite the importance of tidal dynamics, datasets that map key tidal statistics, such as Lowest Astronomical Tide (LAT), Highest Astronomical Tide (HAT), Mean Low Spring Water (MLSW), Mean High Spring Water (MHSW), and tidal range, remain limited in spatial resolution and regional specificity. This gap is particularly evident for Australian waters, which host diverse and globally significant coral reef systems, such as the Great Barrier Reef, and extensive intertidal regions. High-quality tidal data are essential for mapping and monitoring these environments, particularly in applications such as delineating LAT boundaries for navigation and habitat mapping, and assessing the impacts of tidal range on nearshore turbidity and satellite imagery interpretation.

To address this need, we developed a high-resolution dataset derived from the EOT20 global tidal model, tailored specifically to Australian waters. This dataset provides spatially explicit information on LAT, HAT, MLSW, MHSW, and tidal range, offering a detailed understanding of tidal dynamics across the region.

The primary driver for the development of this dataset was to assist in the mapping of coral reefs and intertidal zones, particularly in areas where tidal range plays a significant role in modulating visibility and habitat exposure. For instance, understanding the relationship between tidal range and satellite-derived waterline positions can guide the use of all-tide or low-tide composite imagery for detecting LAT. This distinction is critical in regions with high tidal ranges, where the disparity between Mean Sea Level (MSL) and LAT can introduce significant errors in mapping efforts.

# Tide Modelling Background
The Empirical Ocean Tide model 2020 (EOT20) is a comprehensive global ocean tide model developed by the Deutsches Geodätisches Forschungsinstitut of the Technische Universität München (DGFI-TUM). EOT20 utilizes residual tidal analysis of multi-mission satellite altimetry data spanning from 1992 to 2019 against the FES2014b tide model (Hart-Davis, et al., 2021a).

Tidal constituents are the individual oscillatory components of tides, each corresponding to a specific gravitational interaction between the Earth, Moon, and Sun. These constituents are identified by unique names and letter codes (e.g., M2, S2, K1) and are characterized by their amplitude, phase, and frequency. By summing these components, tide predictions can be made for any given location (Parker, 2007). Models like EOT20 represent these tidal oscillations as spatial raster maps, showing the relative strength and phase of each constituent across the globe. The model's values are adjusted using satellite observations to best match real-world tides, accounting for variations in bathymetry and other dynamic oceanographic factors.

EOT20 provides seventeen tidal constituents, including both primary components like M2, S2, N2, K2, and K1, as well as minor constituents such as 2N2, J1, M4, MF, and MM. These are provided on a global 0.125-degree grid stored in 17 NetCDF files (Hart-Davis, et al., 2021b). This high-resolution representation allows for precise modeling of tidal behaviors across the world's oceans. 

# Methods:

The high resolution tidal dataset was generated using a suite of Python scripts that integrate the EOT20 global tidal model with robust geospatial processing routines. The overall methodology consists of three primary stages: the creation of a processing grid (Tide-Model-Grid), the computation of tidal statistics (Tidal-Stats), and the subsequent merging of spatial subsets (Merge-Strips). These stages work in concert to produce spatially explicit estimates of tidal parameters—including estimates of the Lowest and Highest Astronomical Tides (LAT/HAT) or their predicted counterparts (LPT/HPT), as well as Mean Low Water Springs (MLWS) and Mean High Water Springs (MHWS)—tailored to the coastal and intertidal environments of Australian waters.

# References
Hart-Davis, M. G., Piccioni, G., Dettmering, D., Schwatke, C., Passaro, M., and Seitz, F. (2021a) _EOT20: a global ocean tide model from multi-mission satellite altimetry_. Earth Syst. Sci. Data, 13, 3869–3884. https://doi.org/10.5194/essd-13-3869-2021

Hart-Davis Michael, Piccioni Gaia, Dettmering Denise, Schwatke Christian, Passaro Marcello, Seitz Florian (2021b). _EOT20 - A global Empirical Ocean Tide model from multi-mission satellite altimetry_. \[Dataset\] SEANOE. https://doi.org/10.17882/79489

Parker, B. (2007) _Tidal Analysis and Prediction NOAA Special Publication NOS CO-OPS 3_. National Oceanic and Atmospheric Administration, https://tidesandcurrents.noaa.gov/publications/Tidal_Analysis_and_Predictions.pdf

# Script descriptions
The overall goal of the workflow is to generate high-resolution tidal datasets for Australian coastal and intertidal zones using the EOT20 global tidal model. The process is broken down into three modular scripts that each play a specific role in the workflow:

## Data Download Script (01-download-input-data.py):
This script downloads all the source data used in the tidal modelling, including the EOT20 model tidal constituent data files and land clipping data.

## Tide-Model-Grid Script (02-tide_model_grid.py):
This script sets the stage by generating a spatial grid (output as a GeoTIFF) that identifies which grid cells require tidal processing. Using a user-defined bounding box and cell resolution (default 1/8°). This grid acts as a mask for where the tidal modelling should be performed. The EOT20 tidal constituents do not provides over land and due to the large grid size a significant portion of intertidal areas have no tidal constituents available. To overcome this limitation we ensure that the Tide-Model-Grid includes one extra pixel overlap with land and in the Tidal-Stats script we use the extrapolation capabilities of the pyTMD library to calculate the tides in these locations. 

To improve the performance of the modelling this script also creates a clipped version of the EOT20 tidal constituents to match the user-defined bounding box.

## Tidal-Stats Script (03-tidal_stats.py):
Building on the grid created by the first script, this script performs the core tidal simulations and computes statistics for each grid cell that is flagged for processing. Using the pyTMD library (or a placeholder function in our example), the script simulates tidal elevations over a specified time period (with a default time step of 0.5 hours) and extracts key statistics such as the lowest and highest predicted tides (LPT/HPT) and mean low and high water springs (MLWS/MHWS). To handle high-resolution data efficiently, the work is divided into vertical slices (parallelizable via user-specified split and index parameters), and the script supports restart capability by periodically saving intermediate results.

## Merge-Strips Script (04-merge_strips.py):
After the tidal statistics have been computed for each vertical slice, this final script merges the separate GeoTIFF outputs into complete, contiguous raster datasets for each tidal statistic. The merging process assumes that all partial outputs share the same coordinate reference system, resolution, and alignment, and the final merged files include all the necessary metadata for further analysis or dissemination.

# Model run configuration

Each model run is setup using a configuration YAML file. We use a configuration file, rather than command line parameters, because each of the scripts require similar configuration information. Scripts still have some command line parameters to control the execution of the script, such as the level of parallelisation.

## YAML Configuration File
The configuration file is written in YAML and contains all the parameters needed to define a model run. This file is used by the grid creation and tide model data clipping scripts. Here is a description of each parameter:

- **land_mask_path**
  _Type_: String
  _Description_: Path to the land mask shapefile (in geographic coordinates). This file is used to identify land areas to be excluded from the grid. The `01-download-input-data.py` script downloads and sets up a land mask at: `in-data-3p/ne_10m_land/ne_10m_land.shp`

- **grid_bbox**
  _Type_: List of four floats
  _Description_: Specifies the bounding box for the grid as \[min_lon, min_lat, max_lon, max_lat\] in degrees.

- **grid_cell_size**
  _Type_: Float
  _Description_: The resolution (in degrees) of the grid cells. The EOT20 model has a resolution of 1/8 degree, so use this as a starting point. 

- **land_overlap_px**
  _Type_: Integer
  _Description_: The number of grid cells by which to negatively buffer the land mask. This expands the water area for which we process the tidal statistics. This is needed because the coarse grid structure of the tide model means that inshore areas, particularly river mouths, and bays, might get excluded. Having a land_overlap_px greater than zero helps ensure that all intertidal regions are covered. For the ne_10m_land landmask it doesn't represent rivers and bays very well and so an overlap of 2 pixels is needed. 

- **grid_path**
  _Type_: String
  _Description_: The output path for the generated grid GeoTIFF. This grid represents the area where tide model statistics will be calculated and is used as input for further tidal modelling. This grid is generated by `02-tide_model_grid.py`.

- **tide_model_path**
  _Type_: String
  _Description_: Path to the directory containing the tide model constituent EOT NetCDF files. This is an input for all scripts processing tidal data. The `01-download-input-data.py` saves the model constituent files in `in-data-3p/World_EOT20_2021`.

- **clipped_tide_model_path**
  _Type_: String
  _Description_: The output path for the generated clipped tide model constituent files created by `02-tide_model_grid.py`. Clipping the model data significantly speeds up the tidal modelling calculations. `03-tidal-stats.py` uses these files to drive the calculation of the tidal modelling statistics.

- **clipping_buffer_deg**
  _Type_: Float
  _Description_: A buffer (in degrees) to be added around the grid bounding box when clipping the tide model constituent files. This helps reduce edge effects during processing, particularly with the grid bounding box is not align exactly to the tide model grid. A value equal to one model grid cell is probably sufficient or 1/8 degree.

- **start_date**
  _Type_: YYYY-MM-DD
  _Description_: Start date for the tidal modelling to calculate the statistics from. For LAT and HAT estimates the simulation should be 19 years in length.

- **end_date**
  _Type_: YYYY-MM-DD
  _Description_: End date for the tidal modelling to calculate the statistics from. For LAT and HAT estimates the simulation should be 19 years in length.  

- **time_step**
  _Type_: Float
  _Description_: Time step in hours

- **working_path**
  _Type_: String
  _Description_: Path to a working space where each of the tidal modelling intermediate files will be saved.

- **lpt_label**
  _Type_: String
  _Description_: Label to used for the estimate of the Lowest Astronomical Tide (LAT) estimate. For a proper estimate of LAT a simulation of least 19 years is needed to cover the full lunar nodal cycle. In this case used 'LAT'. For shorter simulations a different name could be used such as 'LPT' for Lowest Predicted Tide. This label is used in the naming of the generated files to represent the statistic generated.

  - **hpt_label**
  _Type_: String
  _Description_: Label to used for the estimate of the Highest Astronomical Tide (HAT) estimate. For a proper estimate of HAT a simulation of least 19 years is needed to cover the full lunar nodal cycle. In this case used 'HAT'. For shorter simulations a different name could be used such as 'HPT' for Highest Predicted Tide. This label is used in the naming of the generated files to represent the statistic generated.



A sample configuration file (config.yaml) might look like this:

```yaml
land_mask_path: "in-data-3p/ne_10m_land/ne_10m_land.shp"
grid_bbox: [122, -18, 124, -16]
grid_cell_size: 0.125
land_overlap_px: 2
grid_path: "working/EOT20-King-sound/grid.tif"
tide_model_path: "in-data-3p/World_EOT20_2021"
clipped_tide_model_path: "working/EOT20-King-sound"
clipping_buffer_deg: 0.5
```

Using a YAML configuration file in this manner improves reproducibility by keeping all model-run parameters in one place. Subsequent scripts in your workflow can simply load the same YAML file to ensure consistency across the entire processing chain.