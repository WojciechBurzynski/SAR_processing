# SAR_processing
SAR data processing pipeline intended to deliver SAR data clustering. 

SOTA assesment of field suggest usage of supervised/self-supervised learning hybrid architectures (CNN with transformers) coupled with optical data fusion and some physics derived decision making.

This repository on the other hand attempts to deliver unsupervidsed clustering capability based on different models (KMeans, GMM, GMM + texture) with physics based constraints.
This is by no means finalized nor optimal pipeline - just the first draft approach to the topic.  

# Inputs
Sentinel-1 SAR VV and VH data as tiff files downloaded from Copernicus Browser https://browser.dataspace.copernicus.eu/ 

Filtered by: 
Image format: TIFF 32-bit
Image resolution: high
Coordinate system: UTM
Layers: Raw VV and VH

# Environment setup

In order to run sar_processing.py setup Python virtual environment following these steps:
1. create environment: python3 -m venv .venv
2. source it: source .venv/bin/activate
3. install dependencies:  pip install -r requirements.txt 

Environment setup could of course be improved by docker image usage and poetry but this is just how i made it for first attempt. 

# Usage

Once you have environment setup done you can run sar_processing.py using 
python3 sar_processing.py 

Steps that script does:  
1. Initialization and arguments parsing - there is only one argument for now --read_from_file that determines if SAR data should be read from tiff files or API but API implementation is not yet there so argument should be ommited (not required).
2. Hello message is displaied and Python version used (tested  on Python version: 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0])
3. Tiff files should be available upfront in "tiffs" folder. This is where script will serach for them and their names should comply with TIFF_NAMES.txt. If you want different tiff files, just add them to folder and change names in TIFF_NAMES.txt. It expects that VV tiff flie name will be first on the list and VH will be 2nd. 
4. Filtering and features extraction: 
- VV and VH images are cleaned, 
- converted to dB, 
- and then filtered using Lee speckle filter.
- VV/ VH ratio is computed as additional feature. 
- baseline plots are saved to files: vv and vh before and after filtering + ratio.
5. Clustering: Approach 1: KMeans:
- KMeans clustering is performed using following inuput features - vv_filtered, vh_filtered, ratio 
- raw clustered output is saved to file - kmeans_raw.png
- mapping of clusters to appriopriate classes is performed (based on signal nature: water darkest, etc.)
- clustered data with assigned labels is saved as plot to file - kmeans_result.png
6. Clustering: Approach 2: GMM:
- the same steps is performed for GMM method using following inuput features - vv_filtered, vh_filtered, ratio
7. Clustering: Approach 3: GMM + texture features:
- GMM method is extended by usage of texture features vv_tex and vh_tex so feature set is as following -  vv_filtered, vh_filtered, ratio, vv_tex, vh_tex
8. Clustering: Approach 4: physics based thresholds
- Thresholds are set for each image point based on heuristic thresholds
- Could/should be improved by thresholds set with use of statistical data analysis of provided tiffs. 
9. Combine physics based and other methods - in order to obtain best results approach for hybrid clustering is made by relying on heuristics first and ml methods later for refinement. Such outputs are generated for each of the Approaches. 
10.  Overlay on OSM for easier assesment of results - html files generation for OSM overlay of clustered data. 
