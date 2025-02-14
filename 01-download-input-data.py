from data_downloader import DataDownloader

import os
import zipfile

# Create an instance of the DataDownloader class
downloader = DataDownloader(download_path="in-data-3p")
VERSION = '1-1'

# --------------------------------------------------------
# Function to unzip a file to a specified directory
def unzip_file(zip_file_path, extract_to):
    print(f"Unzipping {zip_file_path} to {extract_to}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped {zip_file_path} successfully!")
    
    
print("Downloading source data files. This will take a while ...")
    
# --------------------------------------------------------
# Australian Coastline 50K 2024 (NESP MaC 3.17, AIMS)
# https://eatlas.org.au/geonetwork/srv/eng/catalog.search#/metadata/c5438e91-20bf-4253-a006-9e9600981c5f
# Hammerton, M., & Lawrey, E. (2024). Australian Coastline 50K 2024 (NESP MaC 3.17, AIMS) (2nd Ed.) [Data set]. eAtlas. https://doi.org/10.26274/qfy8-hj59

# Use this version for overview maps
#direct_download_url = 'https://nextcloud.eatlas.org.au/s/DcGmpS3F5KZjgAG/download?path=%2FV1-1%2F&files=Simp'
#downloader.download_and_unzip(direct_download_url, 'AU_AIMS_Coastline_50k_2024', subfolder_name='Simp', flatten_directory=True)

# --------------------------------------------------------
# Natural Earth. (2025). Natural Earth 1:10m Physical Vectors - Land [Shapefile]. https://www.naturalearthdata.com/downloads/10m-physical-vectors/
direct_download_url = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip'
downloader.download_and_unzip(direct_download_url, 'ne_10m_land')


# --------------------------------------------------------
# Hart-Davis Michael, Piccioni Gaia, Dettmering Denise, Schwatke Christian, Passaro Marcello, Seitz Florian (2021). 
# EOT20 - A global Empirical Ocean Tide model from multi-mission satellite altimetry. SEANOE. https://doi.org/10.17882/79489
#
# Hart-Davis Michael G., Piccioni Gaia, Dettmering Denise, Schwatke Christian, Passaro Marcello, Seitz Florian (2021). 
# EOT20: a global ocean tide model from multi-mission satellite altimetry. Earth System Science Data, 13 (8), 3869-3884.
# https://doi.org/10.5194/essd-13-3869-2021


direct_download_url = 'https://www.seanoe.org/data/00683/79489/data/85762.zip'
eot20_folder = 'World_EOT20_2021'
downloader.download_and_unzip(direct_download_url, eot20_folder)

# Prepare paths for load_tides and ocean_tides zip files
base_path = os.path.join(downloader.download_path, eot20_folder)
load_tides_zip = os.path.join(base_path, "load_tides.zip")
ocean_tides_zip = os.path.join(base_path, "ocean_tides.zip")

# Unzip ocean_tides.zip
ocean_tides_folder = os.path.join(base_path, "ocean_tides")
if os.path.exists(ocean_tides_zip):
    print(f"{ocean_tides_folder} found. Skipping...")
else:
    unzip_file(ocean_tides_zip, base_path)

# Remove load_tides.zip as we don't use it in the simulation
if os.path.exists(load_tides_zip):
    print(f"Removing {load_tides_zip}...")
    os.remove(load_tides_zip)
    print(f"{load_tides_zip} removed successfully!")

print("All files are downloaded and prepared.")

