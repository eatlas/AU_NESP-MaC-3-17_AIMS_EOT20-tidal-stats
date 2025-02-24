# EOT20-tidal-stats: Tidal statistics for Australia - Processing scripts
[![Metadata record](https://img.shields.io/badge/DOI-10.26274%2Fz8b6--zx94-blue)](
https://doi.org/10.26274/z8b6-zx94)

This repository provides a record of the scripts used in the development of the AU_NESP-MaC-3-17_AIMS_EOT20-tidal-stats dataset. More details are provided in the dataset metadata:

Tidal Statistics for Australia (Tidal range, LAT, HAT, MLWS, MHWS) derived from the EOT20 tidal model (NESP MaC 3.17, AIMS) (V1) [Data set]. eAtlas. https://doi.org/10.26274/z8b6-zx94



## Introduction

Tides play a crucial role in shaping coastal and marine environments, influencing processes such as sediment transport, water clarity, the distribution of ecosystems and the structure of coral reefs. In particular, the tidal range, the difference between high and low tides, has significant implications for the physical and biological dynamics of intertidal regions. For coral reefs, tidal fluctuations determine the exposure and submersion of reef structures, influencing reef zonation, biodiversity, and habitat structure. Accurate and detailed tidal information is therefore essential for understanding these critical ecosystems.

Despite the importance of tidal dynamics there are few public datasets that make tidal statistics, such as Lowest Astronomical Tide (LAT), Highest Astronomical Tide (HAT), Mean Low Spring Water (MLSW), Mean High Spring Water (MHSW), and tidal range readily available at a national scale. [Digital Earth Australia's Intertidal mapping](https://www.ga.gov.au/scientific-topics/dea/dea-data-and-products/dea-intertidal) combined tidal modelling as an integral part of interpretting marine satellite imagery (Bishop-Taylor, et al., 2019). They used tidal modelling to link the height of the water with each satellite image, then combined this with calculating the Normalised Differene Water Index to create 3-dimensional profiles of the shallow portions of the intertidal zone. They showed that sun synchronous satellites, such as the Sentinel 2 satellite used for the coastal imaging, only observes a limited portion of the full tidal range. As part of their analysis they developed downscaled intertidal tidal statistics such as LAT and HAT, based on the EOT20 dataset. The same tidal model that we use in this dataset. In this dataset we calculate additional tidal statistics that are useful for analysing the intertidal zone.

This dataset seeks to create national scale tidal statistics at sufficient resolution that fully represents the detail in the original tidal model. In this dataset we calculate the tidal statistics at 4x the resolution of the EOT20 model resulting in a grid resolution of 1/32 degree. 

The primary driver for the development of this dataset was to assist in the mapping of coral reefs and intertidal zones, particularly in areas where tidal range plays a significant role in modulating visibility and habitat exposure. For instance, understanding the relationship between tidal range and satellite-derived waterline positions can guide the use of all-tide or low-tide composite imagery for detecting LAT. This distinction is critical in regions with high tidal ranges, where the disparity between Mean Sea Level (MSL) and LAT can introduce significant errors in mapping efforts.

# Tide Modelling Background
The Empirical Ocean Tide model 2020 (EOT20) is a comprehensive global ocean tide model developed by the Deutsches Geodätisches Forschungsinstitut of the Technische Universität München (DGFI-TUM). EOT20 utilizes residual tidal analysis of multi-mission satellite altimetry data spanning from 1992 to 2019 against the FES2014b tide model (Hart-Davis, et al., 2021a).

Tidal constituents are the individual oscillatory components of tides, each corresponding to a specific gravitational interaction between the Earth, Moon, and Sun. These constituents are identified by unique names and letter codes (e.g., M2, S2, K1) and are characterized by their amplitude, phase, and frequency. By summing these components, tide predictions can be made for any given location (Parker, 2007). Models like EOT20 represent these tidal oscillations as spatial raster maps, showing the relative strength and phase of each constituent across the globe. The model's values are adjusted using satellite observations to best match real-world tides, accounting for variations in bathymetry and other dynamic oceanographic factors.

EOT20 provides seventeen tidal constituents, including both primary components like M2, S2, N2, K2, and K1, as well as minor constituents such as 2N2, J1, M4, MF, and MM. These are provided on a global 0.125-degree grid stored in 17 NetCDF files (Hart-Davis, et al., 2021b). This high-resolution representation allows for precise modeling of tidal behaviors across the world's oceans. 

# Methods:

The high resolution tidal dataset was generated using a suite of Python scripts. The overall methodology consists of four primary stages: download the tidal model and other source data, the creation of a processing grid (Tide-Model-Grid), the computation of tidal statistics (Tidal-Stats), and the subsequent merging of spatial subsets (Merge-Strips). These stages work in concert to produce spatially explicit estimates of tidal parameters—including estimates of the Lowest and Highest Astronomical Tides (LAT/HAT) or their predicted counterparts (LPT/HPT), as well as Mean Low Water Springs (MLWS) and Mean High Water Springs (MHWS)—tailored to the coastal and intertidal environments of Australian waters.

# References

Bishop-Taylor, R., Sagar, S., Lymburner, L., & Beaman, R. J. (2019). Between the tides: Modelling the elevation of Australia’s exposed intertidal zone at continental scale. Estuarine, Coastal and Shelf Science, 223, 115–128. https://doi.org/10.1016/j.ecss.2019.03.006

Hart-Davis, M. G., Piccioni, G., Dettmering, D., Schwatke, C., Passaro, M., and Seitz, F. (2021a) _EOT20: a global ocean tide model from multi-mission satellite altimetry_. Earth Syst. Sci. Data, 13, 3869–3884. https://doi.org/10.5194/essd-13-3869-2021

Hart-Davis Michael, Piccioni Gaia, Dettmering Denise, Schwatke Christian, Passaro Marcello, Seitz Florian (2021b). _EOT20 - A global Empirical Ocean Tide model from multi-mission satellite altimetry_. \[Dataset\] SEANOE. https://doi.org/10.17882/79489

Parker, B. (2007) _Tidal Analysis and Prediction NOAA Special Publication NOS CO-OPS 3_. National Oceanic and Atmospheric Administration, https://tidesandcurrents.noaa.gov/publications/Tidal_Analysis_and_Predictions.pdf


# License
The code associated with this dataset is made available under an MIT license. The generated dataset is made available under a Creative Commons Attribution 4.0 International license.


# Installation Guide

This repository provides both **Conda** (`environment.yaml`) and **pip** (`requirements.txt`) options for setting up dependencies. Below are the installation steps for **Linux, Windows, and HPC** environments. Note: that the HPC and Linux instructions are untested and is currently just a proposed method based on what I think might work.

## 1. Prerequisites
- Ensure **Python 3.9+** is installed.
- If using Conda, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or load the **conda module** (on HPC).
- If using pip, ensure system dependencies like **GDAL, PROJ, and GEOS** are installed. These should be installed using the provided pip instructions, however success will vary across platforms.

## 2. Clone the Repository
```bash
git clone https://github.com/eatlas/AU_NESP-MaC-3-17_AIMS_EOT20-tidal-stats
cd AU_NESP-MaC-3-17_AIMS_EOT20-tidal-stats
```

## 3A. Using Conda (Recommended)
Conda is recommended because it makes installing of GDAL, PROJ and GEOS more straight forward.

1. Create the Conda environment. This step can take 10 min.
    ```bash
    conda env create -f environment-3-13.yaml
    ```
2. Activate the environment
    ```bash
    conda activate venv_eot20_3_13
    ```
A second version the environment script is provided (`environment-3-9-7.yaml`) that builds older version of Python and libraries that has been tested to work. This is provided as a reference to help getting the code to run on older versions of Python. 

I found that conda struggles building the environments when the specific library version numbers are not specified. The two versions provided were tested to work on Windows.

## 3B. Using Pip (Alternative)
The virtual environment will be created at the root of the folder that the command is run. It is most common to run this in the project folder. For this you will already need Python to be installed. The code has been tested on Python 3.9.7

For setting up with plain Python in Windows.
1. Download [Python 3.13](https://www.python.org/downloads/release/python-3131/). 
2. Install Python to the default user space `C:\Users\<user>\AppData\Local\Programs\Python\Python313\`, replacing `<user>` for you username.
3. Create the virtual environment in the project folder. Open a command window and navigate to the project directory. 
    ```cmd
    cd <path to project repo>\AU_NESP-MaC-3-17_AIMS_EOT20-tidal-stats
    "C:\Users\<user>\AppData\Local\Programs\Python\Python313\python.exe" -m venv venv_eot20_3_13
    ```
    **Remember to swap out the user** in the path.
4. Activate the environment
  For windows (command prompt CMD)
  ```cmd
  venv_eot20_3_13\Scripts\activate.bat
  ```

  For macOS/Linux (untested)
  ```bash
  source eot20_venv/bin/activate
  ```
4. Install dependencies
    ```cmd
    pip install -r requirements-3-13.txt
    ```
There is also a `requirements-3-9.txt` that is the version of libraries that are known to work with Python 3.9.

## 4. Verify installation
This should flush out whether the setup is mostly working. Once this is working you should run the quick test run.
```bash
python -c "import rasterio, geopandas, pyTMD; print('All libraries imported successfully')"
```

## 5. HPC Installation Notes
This section is a place holder for documentation on how to get this code running on a HPC.

# Testing the code
Once you have an operating setup with all the libraries installed your first run of the code should be to run one of the small simulations that only takes a few minutes. This will identify any issues with the setup prior to performing the full run of the code.

## Quick test run
1. **Download source data**: This includes the EOT20 model tide constituent files and the land masking files.
    ```bash
    python 01-download-input-data.py
    ```
2. **Create the simulation grid**: Calculate a grid that we will perform the tidal simulation on. A YAML configuration file is used to specify the all the parameters of the simulation such as the bounding box, resolution, time extent, etc. In this quick run we will use a preconfigured example config file to simulate just one month for a small section of the Kimberly in Western Australia. This simulation will only take less than 1 minute. This script will make the model grid, and create a cropped version of the EOT20 tidal constituent files to make the simulation faster.
    ```bash
    python 02-tide_model_grid.py --config config/king-sound-quick-test.yaml
    ```
3. **Run the tidal model**: This will perform the tidal modelling over the simulation grid for the time period specified in the configuration file, then calculate the tidal statistics for each pixel in the grid.
    ```bash
    python 03-tidal_stats.py --config config/king-sound-quick-test.yaml
    ```
    This script has a feature to split the work into processes for parallel processing, but in this quick test it is not needed.
4. **Consolidate the result and save to its final destination**: When the tidal simulation is split across multiple processes, each process calculates a separate portion of the grid (organised as strips). This script merges all the strips back into one raster per statistic and saves the result to the final destination (specified in the YAML config). In this run because we didn't split the job up, so this script simply makes a copy of the data into the final destination `working/EOT20-king-sound/`
    ```bash
    python 04-merge_strips.py --config config/king-sound-quick-test.yaml
    ```
    In `working/EOT20-king-sound/` you should find four GeoTiff files  
    - `HPT_2024-01_2024-01.tif`
    - `LPT_2024-01_2024-01.tif`
    - `MHWS_2024-01_2024-01.tif`
    - `MLWS_2024-01_2024-01.tif`

    These files can be loaded into QGIS or ArcGIS Pro for viewing. Windows will not correctly show these images properly because the data in the files is in 32 bit float.

## Parallel test run
This run is much larger than the quick test run as it simulated northern Australia over a 1 year period. This test is long enough that it is worth splitting up into multiple parallel runs. In this example we have split the number of parallel runs into 4 so that it can be run locally on a laptop for testing purposes. 

Parallelism is achieved by splitting the task into multiple independed runs of the `03-tidal_stats.py` script that each process an independent strip of the final grid. The results are then merged together using the `04-merge_strips.py`. To split the processing we indicate to the script how many vertical strips the grid should be `split` and then provide each process with an `index` to indicate which portion it should process.

Setup the Grid:
```bash
python 02-tide_model_grid.py --config config/northern-au-test.yaml
```
Start four command line windows to run four processes in parallel. On windows I am using conda and so each of these is an Anaconda prompt corresponding to the environment needed for this code. Running this code took 3 hr 18 min.

```bash
python 03-tidal_stats.py --config config/northern-au-test.yaml --split 4 --index 0

python 03-tidal_stats.py --config config/northern-au-test.yaml --split 4 --index 1

python 03-tidal_stats.py --config config/northern-au-test.yaml --split 4 --index 2

python 03-tidal_stats.py --config config/northern-au-test.yaml --split 4 --index 3
```
Each script will write to standard out, but also write a log to `working\EOT20-nau-test\tmp`. The log files are intended to be useful for monitoring running the script on HPC.

Merge the results:
```bash
python 04-merge_strips.py --config config/northern-au-test.yaml --split 4
```

# Script descriptions
The overall goal of the workflow is to generate high-resolution tidal datasets for Australian coastal and intertidal zones using the EOT20 global tidal model. The process is broken down into four modular scripts that each play a specific role in the workflow:

## Data Download Script (01-download-input-data.py):
This script downloads all the source data used in the tidal modelling, including the EOT20 model tidal constituent data files and land clipping data.

## Tide-Model-Grid Script (02-tide_model_grid.py):
This script sets the stage by generating a spatial grid (output as a GeoTIFF) that identifies which grid cells require tidal processing. Using a user-defined bounding box and cell resolution (default 1/8°). This grid acts as a mask for where the tidal modelling should be performed. The EOT20 tidal constituents cover land areas and due to their large grid size a significant portion of intertidal areas have no tidal constituents available. To overcome this we ensure that the Tide-Model-Grid includes extra pixel overlaps with land and in the Tidal-Stats script we use the extrapolation capabilities of the pyTMD library to calculate the tides in these locations. 

To improve the performance of the modelling this script also creates a clipped version of the EOT20 tidal constituents to match the user-defined bounding box, with a bit of buffering.

## Tidal-Stats Script (03-tidal_stats.py):
Building on the grid created by the first script, this script performs the core tidal simulations and computes statistics for each grid cell that is flagged for processing. Using the pyTMD library, the script simulates tidal elevations over the specified time period and extracts key statistics such as the lowest and highest predicted tides (LPT/HPT) and mean low and high water springs (MLWS/MHWS). The MLWS and MHWS are calculated by averaging the tidal extremes during the periods (-0.5 days to +4days) of each new and full moons. 
The tidal modelling can be slow as as accurate long time series simulations are needed to calculate accuracte statistics. To allow for faster processing the tidal modelling can be processed in independent strips. The strips are then merged back into the final full grid with the merge-strips script. The script also supports restart cancelling and restarting the processing as it periodically saves intermediate results. On restart it will resume from where it left off. 

## Merge-Strips Script (04-merge_strips.py):
After the tidal statistics have been computed for each vertical slice, this final script merges the separate GeoTIFF outputs into complete, contiguous raster datasets for each tidal statistic. 

## Time series validation (05a-validation-time-series.py)
This script compares the predictions of the EOT20 model against tide gauges that are part of the Australian Baseline Sea Level Monitoring Project. This comparison is performed on monthly min, mean and max statistics. 

## Monthly tidal climatology validation (05b-validation-month-clim.py)
This script calculates the monthly climatology of the min, mean and max tides, with the goal of showing the typical tidal conditions experienced in each month. To estimate the monthly climatology we pool all the measurements for a given month (monthly min, mean, max statistics) from the entire time series to calculate the median monthly minimum, median monthly mean and the median monthly maximum. We use a median to help to remove outliers caused by storm events, to end up with a result that is closer to the tidal predictions that are just based on astrononmical modelling. For consistency with also apply calculate the tidal prediction climatology using the same process. First predicting a 19 year series at 30 min intervales, then calculating the monthly min, mean and max, then calculating the climatology across the 19 years combining the multiple results for each month using a median. While using a median on the monthly statistics will remove some of the noise introduced by storms it will not remove it completely. This is because the monthly minimum and maximum and biased to capture small transient events. If a location experiences at least one event per month then the minimum and maximum will be consistently shifted from the astronomical predictions. Since our goal is to provide a dataset that can be used to analyse the environmental conditions of the intertidal zone then it is more important to understand typical tidal conditions, rather than simply astronomical predictions.

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
region_name: "King Sound"
land_mask_path: "in-data-3p/ne_10m_land/ne_10m_land.shp"
grid_bbox: [122, -18, 124, -16]
grid_cell_size: 0.125
land_overlap_px: 2
grid_path: "working/EOT20-king-sound/grid.tif"
tide_model_path: "in-data-3p/World_EOT20_2021/ocean_tides"
clipped_tide_model_path: "working/EOT20-king-sound/ocean_tides"
clipping_buffer_deg: 0.125
start_date: "2024-01-01"
end_date: "2024-01-31"
time_step: 1
working_path: "working/EOT20-king-sound/tmp"
lat_label: 'LPT'
hat_label: 'HPT'
output_path_prefix: "working/EOT20-king-sound/"
```

Using a YAML configuration file in this manner improves reproducibility by keeping all model-run parameters in one place. Subsequent scripts in your workflow can simply load the same YAML file to ensure consistency across the entire processing chain.

---

# Debugging

 Getting a working environment with GDAL, PROJ and GEOS is difficult. This section is a set of notes working through the problems with getting a working conda environment where the various libraries would play nice with each other. It is unclear whether any of this documentation will help get the code running on a different version of Python or on a different platform.

### I have a new environment but the library imports fail
python -c "import rasterio, geopandas, pyTMD; print('All libraries imported successfully')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\elawrey\Anaconda3\envs\eot20_env\lib\site-packages\rasterio\__init__.py", line 27, in <module>
    from rasterio._vsiopener import _opener_registration
ImportError: DLL load failed while importing _vsiopener: The specified procedure could not be found.

Rasterio, GeoPandas, and other GIS libraries rely heavily on matching versions of GDAL, PROJ, libgeos, etc. If you install Rasterio from conda-forge, but GDAL or PROJ come from defaults, you often wind up with mismatched DLLs. I worked around this issue by building the environment with a different combination of library versions.

### The scripts fails - 03-tidal_stats.py fails part way through simulation
The 03-tidal_stats.py script sometimes fails midway through the simulation, seemingly due to a disk error. When it fails appears to be random and when it does fail, all parallel runs of the script fail with the error:
```
  File "rasterio\\_io.pyx", line 1466, in rasterio._io.DatasetWriterBase.__init__
  File "rasterio\\_io.pyx", line 332, in rasterio._io._delete_dataset_if_exists
  File "rasterio\\_err.pyx", line 195, in rasterio._err.exc_wrap_int
rasterio._err.CPLE_AppDefinedError: Deleting working/EOT20-nau-test/tmp\LPT_EOT20_2023-01_2023-12_strip_1.tif failed: Invalid argument
```
Since all parallel runs fail simultaneously it seems likely that this is due to a temporary glitch with read/write operations to the external HD that I have been running the simulations on.

### Conda install - Error while installing libjpeg package
When installing with conda:

```
ERROR conda.core.link:_execute(938): An error occurred while installing package 'conda-forge::libjpeg-turbo-3.0.0-hcfcfb64_1'.
Rolling back transaction: done

[Errno 13] Permission denied: 'C:\\Users\\elawrey\\Anaconda3\\pkgs\\libjpeg-turbo-3.0.0-hcfcfb64_1\\Library\\bin\\wrjpgcom.exe'
()
```
Deleting the `C:\\Users\\elawrey\\Anaconda3\\pkgs\\libjpeg-turbo-3.0.0-hcfcfb64_1` folder (which requires admin privledges) then rerunning the conda environment setup installation worked.


### Conda attempt 1: Using Conda - environment.yaml no library versions
I tried and failed building an environment using conda where I specify the Python version (3.9) and which libraries, but no version numbers for the libraries. This was specified in a `environment.yaml`. This ended up building the environment, but when I tested the enviornment it failed with a DLL error. This was possibly due to subtle version mismatches in the DLLs on Windows (theory proposed by ChatGPT, so take it with a grain of salt). The environment builds, but I ended up with `ImportError: DLL load failed while importing _vsiopener` error when loading `rasterio` and `geopandas`. 

To resolve this issue I tried forcing the environment to be built using a single channel. This was in theory to help ensure consistent DLLs:
```
conda env create -f environment.yaml
conda activate eot20_env
conda config --env --set channel_priority strict
conda install --override-channels -c conda-forge --update-deps --force-reinstall rasterio geopandas pyproj gdal
```
This was a very slow process, but when it completed it ended up with the following error:
```
UnsatisfiableError: The following specifications were found
to be incompatible with the existing python installation in your environment:

Specifications:

  - conda-env -> python[version='2.7.*|3.4.*|3.5.*']

Your python: python=3.9
```
Not sure what the problem was here. Conda seems to struggle to work out which versions of the libraries are compatible with each other. 


### Conda install attempt 2: Progress adding of libraries in Anaconda Navigator
I tried building the environment using Anaconda Navigator, progressively adding key libraries to the environment until all were added. The plan was to then save the `environment.yaml` from this build. The advantage of this approach is that it specifies in detail the exact version of every libraries and all dependencies. This means that to recreate this environment conda doesn't need to resolve dependencies.

For this I started with a vanilla Python 3.11.11 environment. A recent, but not the latest version. This version was chosen because I knew that these libraries work in this version. 
Using the `defaults` channel I started with the main key libraries:
- numpy 2.2.2
- pandas 2.2.3
- matplotlib-base 3.10.0
- xarray 2024.11.0
Success. Once these were installed I added spatial libraries:
- rasterio 1.3.10
- geopandas 0.14.2
- shapely 2.0.6

We now add the trailing dependencies. 
- pyyaml 6.0.2

After many hours the installation failed with an UnsatisfiableError. It seemed to indicate that maybe some packages didn't support Python 3.11.1.

I also discovered that pyTMD was not found in the `defaults` channel and so as per [pyTDM install guide](https://pytmd.readthedocs.io/en/latest/getting_started/Install.html) I added the `conda-forge` channel. 

I decided that this was a dead end. 

### Conda attempt 3: Using the version numbers of an existing working environment
My existing development Data Science Python environment has working libraries, but has many additional libraries that are not needed for the scripts in this dataset. The plan was to determine the version numbers of all the libraries and Python that matched the working environment. Create an `environment.yaml` from this information.

The existing working setup has the following:
python 3.9.7
rasterio 1.3.10
geopandas 0.12.2
shapely 2.0.1
numpy 1.26.0
pandas 2.1.1
matplotlib 3.7.2
xarray 2024.1.1
pyyaml 6.0.1
pytmd 2.2.0

I tried creating a `environmental.yaml` from this, but it led to a conflict between Shapely (2.0.1) and Rasterio (1.3.10) because they require different versions of GEOS. The error message indicated that Shapely 2.0.1 needs GEOS 3.11.1 - 3.11.2 and Rasterio 1.3.10 needs GEOS 3.12.1 or higher. Strangely the working environment has version 3.8.0 of GEOS installed and so should not technically work. This implies that the dependancy information associated with libraries is not reliable.

To find a working combination I tried incrementing the shapely version until there was an overlap between rasterio and shapely. This was done with:
```
conda create -n test_env -c conda-forge shapely=2.0.2 rasterio=1.3.10
```
Just doing this with two packages made testing much faster. This showed that shapely 2.0.2 and rasterio 1.3.10 can work together. So I adjusted the `environmental-3-9-7.yaml` to 
```yaml
name: geo_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9.7
  - rasterio=1.3.10
  - geopandas=0.12.2
  - shapely=2.0.2
  - numpy=1.26.0
  - pandas=2.1.1
  - matplotlib=3.7.2
  - xarray=2024.1.1
  - pyyaml=6.0.1
  - pytmd=2.2.0
```
```bash
conda env create -f environment-3-9-7.yaml
```

This combination was successful in building.
```
conda activate geo_env
python -c "import rasterio, geopandas, pyTMD; print('All libraries imported successfully')"
All libraries imported successfully
```

### Building an environment with the latest libraries 
Since Python 3.9.7 was quite old, I wanted a version of the build that would be more modern. I decided to simply try creating a `environment.yaml` specifying all the latest stable versions of the libraries and Python. To find the version numbers I just look at the source code repos in GitHub for each library (finding them with a Google Search for `<library name> source`) then looking at the release tags. 
This approach worked. It seems that the libraries are generally quite compatible with each other, even if conda can't find a solution. This result was saved to `environment-3-13.yaml`

### Pip install debugging - Pip install fails because it can't install Python - Windows
I started with the version of python specified in the requirements.txt file thinking that this would be a good way to specified a version of Python known to work that should be used, unfortunately the pip install fails with the following error:
```cmd
ERROR: Could not find a version that satisfies the requirement python==3.9.7 (from versions: none)
ERROR: No matching distribution found for python==3.9.7
```
The reason this error is that pip can not be used to install a version of python. This is not how pip works. Pip only installs packages into an existing Python installation. To fix this I removed `python==3.9.7` from the `requirements.txt`. I built a conda environment with the Python version, then tested the pip install.

### Pip install debugging - Running 02-tide_model_grid fails with a fiona problem
When running the quick test on a new installation from using pip I found the following problem:
```cmd 
python 02-tide_model_grid.py --config config/king-sound-quick-test.yaml
```

Started script with the following configuration:
```
Configuration parameters:
...
File "D:\AU_AIMS_MARB-S2-comp_p15\AU_NESP-MaC-3-17_AIMS_EOT20-tidal-stats\eot20_venv\lib\site-packages\geopandas\io\file.py", line 164, in _is_zip
    parsed = fiona.path.ParsedPath.from_uri(path)
AttributeError: module 'fiona' has no attribute 'path'
```
This error occurs because pip installed two incompatible version of fiona and geopandas. 

Running
```cmd
python -c "import geopandas as gpd, fiona; print(gpd.__version__); print(fiona.__version__)"
```
Returns:
```
0.12.2
1.10.1
```
It turns out that version 1.10.1 of Fiona is not compatible with GeoPandas 0.12.2. Unfortunately there are no compatibility tables and we instead rely on our conda install to find out that 0.12.2 of geopandas works with 1.9.6 of fiona and we add that to the `requirements.txt`. [Geopandas supports two engines](https://geopandas.org/en/latest/docs/user_guide/fiona_to_pyogrio.html) work reading and writing, Fiona and Pygrio. For version GeoPandas 0.12.2 that we have managed to get to work has fiona listed as a [dependancy](https://github.com/geopandas/geopandas/blob/efcb3675d94935ee19b06c75467f9ccc24eb8843/environment.yml) but there is no version constraints.

This error should only occur with older versions of Geopandas, as recent versions no longer use Fiona.
