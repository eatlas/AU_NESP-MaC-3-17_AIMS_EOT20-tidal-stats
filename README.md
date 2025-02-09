# EOT20-tidal-stats: Tidal statistics for Australia - Processing scripts
This repository provides a record of the scripts used in the development of the AU_NESP-MaC-3-17_AIMS_EOT20-tidal-stats dataset:

Lawrey, E. (2025) Tidal Statistics for Australia (Range, LAT, HAT, MLWS, MHWS) derived from the EOT20 tidal model (NESP MaC 3.17, AIMS). eAtlas. https://eatlas.org.au/geonetwork/srv/eng/catalog.search#/metadata/1f070809-ab43-4c4d-a805-e3b6621b7477

Tides play a crucial role in shaping coastal and marine environments, influencing processes such as sediment transport, water clarity, and the distribution of ecosystems. In particular, the tidal range—the difference between high and low tides—has significant implications for the physical and biological dynamics of intertidal regions. For coral reefs, tidal fluctuations determine the exposure and submersion of reef structures, influencing reef zonation, biodiversity, and habitat accessibility. Accurate and detailed tidal information is therefore essential for understanding, monitoring, and managing these critical ecosystems.

Despite the importance of tidal dynamics, datasets that map key tidal statistics, such as Lowest Astronomical Tide (LAT), Highest Astronomical Tide (HAT), Mean Low Spring Water (MLSW), Mean High Spring Water (MHSW), and tidal range, remain limited in spatial resolution and regional specificity. This gap is particularly evident for Australian waters, which host diverse and globally significant coral reef systems, such as the Great Barrier Reef, and extensive intertidal regions. High-quality tidal data are essential for mapping and monitoring these environments, particularly in applications such as delineating LAT boundaries for navigation and habitat mapping, and assessing the impacts of tidal range on nearshore turbidity and satellite imagery interpretation.

To address this need, we developed a high-resolution dataset derived from the EOT20 global tidal model, tailored specifically to Australian waters. This dataset provides spatially explicit information on LAT, HAT, MLSW, MHSW, and tidal range, offering a detailed understanding of tidal dynamics across the region.

The primary driver for the development of this dataset was to assist in the mapping of coral reefs and intertidal zones, particularly in areas where tidal range plays a significant role in modulating visibility and habitat exposure. For instance, understanding the relationship between tidal range and satellite-derived waterline positions can guide the use of all-tide or low-tide composite imagery for detecting LAT. This distinction is critical in regions with high tidal ranges, where the disparity between Mean Sea Level (MSL) and LAT can introduce significant errors in mapping efforts.

# Methods:

The high resolution tidal dataset was generated using a suite of Python scripts that integrate the EOT20 global tidal model with robust geospatial processing routines. The overall methodology consists of three primary stages: the creation of a processing grid (Tide-Model-Grid), the computation of tidal statistics (Tidal-Stats), and the subsequent merging of spatial subsets (Merge-Strips). These stages work in concert to produce spatially explicit estimates of tidal parameters—including estimates of the Lowest and Highest Astronomical Tides (LAT/HAT) or their predicted counterparts (LPT/HPT), as well as Mean Low Water Springs (MLWS) and Mean High Water Springs (MHWS)—tailored to the coastal and intertidal environments of Australian waters.

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
