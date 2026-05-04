####################
# IMPORTS
####################

import sys 
import os 
import logging 
from pathlib import Path 
import argparse

################

import rasterio
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.mixture import GaussianMixture

from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import generic_filter

import folium
from rasterio.warp import transform_bounds
from PIL import Image



####################
# METHODS DEFINITION
####################

def lee_filter(
    img: np.ndarray,
    window_size: int = 5
    ) -> np.ndarray:

    """
    Apply Lee speckle filter to a SAR image.

    Parameters
    ----------
    img : np.ndarray
        Input SAR image (in dB scale).
    window_size : int, optional
        Size of the square moving window.

    Returns
    -------
    np.ndarray
        Speckle-reduced image

    """
    # --- handle invalid values ---
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # local statistics
    mean = uniform_filter(img, window_size)
    mean_sq = uniform_filter(img * img, window_size)

    # local variance
    variance = mean_sq - mean * mean

    # estimate noise variance
    noise_variance = np.nanmean(variance) 

    # avoid negatives to remove numerical errors
    variance = np.maximum(variance, 0.0)

    # Lee gain
    weights = variance / (variance + noise_variance + 1e-6)

    #final filtered img
    filtered = mean + weights * (img - mean)

    return filtered


def filter_and_extract_features(
    vv: np.ndarray,
    vh: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Apply preprocessing and feature extraction to Sentinel-1 SAR VV and VH data.

    Parameters
    ----------
    vv : np.ndarray
        VV polarization backscatter (linear scale).
    vh : np.ndarray
        VH polarization backscatter (linear scale).

    Returns
    -------
    vv_f : np.ndarray
        Filtered VV backscatter in dB.
    vh_f : np.ndarray
        Filtered VH backscatter in dB.
    ratio : np.ndarray
        Log-domain difference (VV - VH) representing polarization ratio.

    Description
    -----------
    The function:
    - masks non-positive values,
    - converts data to decibel (dB) scale,
    - applies Lee speckle filtering,
    - computes a log-domain VV/VH ratio (difference in dB).
    """

    #remove nonpositive backscatters
    vv[vv <= 0] = np.nan
    vh[vh <= 0] = np.nan

    #convert to dB
    vv_db = 10 * np.log10(vv)
    vh_db = 10 * np.log10(vh)

    #apply filtering 
    vv_filtered = lee_filter(vv_db)
    vh_filtered = lee_filter(vh_db)

    #compute ratio
    ratio = vv_filtered - vh_filtered

    return vv_filtered, vh_filtered, ratio, vv_db, vh_db

def kmeans_clustering(
        vv_f: np.ndarray, 
        vh_f: np.ndarray, 
        ratio: np.ndarray, 
        n_clusters=4
        )-> tuple[np.ndarray, np.ndarray]:
    
    """
    KMeans clustering 

    Parameters
    ----------
    vv_f : np.ndarray
        VV polarization backscatter (linear scale). Filtered
    vh_f : np.ndarray
        VH polarization backscatter (linear scale). Filtered

    Returns
    -------
    labels : np.ndarray
        Pixel assigmnents to certain cluster
    centers : np.ndarray
        Average SAR signature for cluster (needs to be further labeled)
    """

    X = np.stack([vv_f, vh_f, ratio], axis=-1)
    mask = np.isfinite(X).all(axis=-1)

    X_flat = X[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    model = KMeans(
        n_clusters=n_clusters, 
        n_init=10, 
        random_state=0)
    labels_flat = model.fit_predict(X_scaled)

    labels = np.full(mask.shape, -1, dtype=np.int32)
    labels[mask] = labels_flat

    centers = scaler.inverse_transform(model.cluster_centers_)

    return labels, centers

def gmm_clustering(
        vv_f: np.ndarray, 
        vh_f: np.ndarray, 
        ratio: np.ndarray, 
        n_clusters=4
        )-> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian Mixture clustering

    Parameters
    ----------
    vv_f : np.ndarray
        VV polarization backscatter (linear scale). Filtered
    vh_f : np.ndarray
        VH polarization backscatter (linear scale). Filtered

    Returns
    -------
    labels : np.ndarray
        Pixel assigmnents to certain cluster
    centers : np.ndarray
        Average SAR signature for cluster (needs to be further labeled)
    """

    X = np.stack([vv_f, vh_f, ratio], axis=-1)
    mask = np.isfinite(X).all(axis=-1)

    X_flat = X[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        reg_covar=1e-3,
        random_state=0
    )

    model.fit(X_scaled)
    labels_flat = model.predict(X_scaled)

    labels = np.full(mask.shape, -1, dtype=np.int32)
    labels[mask] = labels_flat

    centers = scaler.inverse_transform(model.means_)

    return labels, centers


def gmm_clustering_texture(
        vv_f: np.ndarray, 
        vh_f: np.ndarray, 
        ratio: np.ndarray, 
        n_clusters=4
        )-> tuple[np.ndarray, np.ndarray]:
    """
    GMM clustering extended by texture features.
    
    Parameters
    ----------
    vv_f : np.ndarray
        VV polarization backscatter (linear scale). Filtered
    vh_f : np.ndarray
        VH polarization backscatter (linear scale). Filtered

    Returns
    -------
    labels : np.ndarray
        Pixel assigmnents to certain cluster
    centers : np.ndarray
        Average SAR signature for cluster (needs to be further labeled)
    """

    vv_tex = generic_filter(vv_f, np.nanstd, size=5)
    vh_tex = generic_filter(vh_f, np.nanstd, size=5)

    X = np.stack([vv_f, vh_f, ratio, vv_tex, vh_tex], axis=-1)

    mask = np.isfinite(X).all(axis=-1)
    X_flat = X[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        reg_covar=1e-3,
        random_state=0
    )

    model.fit(X_scaled)
    labels_flat = model.predict(X_scaled)

    labels = np.full(mask.shape, -1, dtype=np.int32)
    labels[mask] = labels_flat

    centers = scaler.inverse_transform(model.means_)

    return labels, centers

def sar_physics_labels(
        vv_f: np.ndarray, 
        vh_f: np.ndarray, 
        )-> np.ndarray:
    """
    SAR heuristic rule-based classification. 
    Could/should be improved by thresholds set with use of statistical data analysis. 
    
    Parameters
    ----------
    vv_f : np.ndarray
        VV polarization backscatter (linear scale). Filtered
    vh_f : np.ndarray
        VH polarization backscatter (linear scale). Filtered

    Returns
    -------
    labels : np.ndarray
        Pixel assigmnents to certain cluster
    """
    
    labels = np.full(vv_f.shape, 3, dtype=np.int32)  # mixed/background

    # WATER 
    water = (vv_f < -15) & (vh_f < -18) 
    labels[water] = 0

    # VEGETATION
    veg = (vh_f > -13) & (~water)
    labels[veg] = 1

    # URBAN
    urban = (vv_f > -9) & (vh_f > -12) & (~water)
    labels[urban] = 2

    # VEGETATION - approach to extend by texture mapping but results were worse

    # vv_tex = generic_filter(vv_f, np.nanstd, size=5)
    # vh_tex = generic_filter(vh_f, np.nanstd, size=5)

    # veg =(
    #     (vh_f > -13) &
    #     (vh_tex < np.nanmedian(vh_tex)) &
    #     (~water)
    # )
    # labels[veg] = 1

    # vv_tex_mean = np.nanmean(vv_tex)

    # URBAN - approach to extend by texture mapping but results were worse

    # urban = (
    #     (vv_f > -9) &
    #     (vv_tex > vv_tex_mean) &   # high structural variation
    #     (ratio > 3) &              # VV dominance
    #     (~water) &
    #     (~veg)
    # )
    # labels[urban] = 2
 
    return labels


def plot_physics_labels(
    labels: np.ndarray,
    filename: str,
    title: str = "SAR Classification Map"
) -> None:
    """
    Plot SAR physics-based classification labels.

    Classes:
    0 = water
    1 = vegetation
    2 = urban
    3 = mixed/background
    """

    cmap = ListedColormap([
        "blue",     # 0 water
        "green",    # 1 vegetation
        "red",      # 2 urban
        "yellow"    # 3 mixed
    ])

    bounds = np.arange(-0.5, 4.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(8, 6))
    img = plt.imshow(labels, cmap=cmap, norm=norm)

    cbar = plt.colorbar(img, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "water",
        "vegetation",
        "urban",
        "mixed"
    ])

    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()



def save_plot_to_file(
    data: np.ndarray,
    filename: str,
    cmap: str = "gray",
    title: str | None = None,
    ) -> None:
    """
    Save a 2D array as an image plot.

    ```
    Parameters
    ----------
    data : np.ndarray
        Input 2D array to visualize (e.g., SAR backscatter or derived features).
    filename : str
        Output file path where the image will be saved.
    cmap : str, optional
        Matplotlib colormap used for visualization
    title : str | None, optional
        Title displayed on the plot.

    Returns
    -------
    None

    """

    vmin = np.nanpercentile(data, 2)
    vmax = np.nanpercentile(data, 98)

    plt.figure()
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

    if title:
        plt.title(title)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    

def save_clustered_data(
    labels: np.ndarray,
    filename: str = "artifacts/clusters.png",
    title: str = "SAR Clusters"
    ) -> None:
    """
    Save clustered SAR classification map to file.

    ```
    Parameters
    ----------
    labels : np.ndarray
        2D array of class labels (e.g., 0=water, 1=vegetation, etc.)
    filename : str
        Output file path
    title : str
        Plot title

    Returns
    -------
    None
    """

    # --- colormap ---
    cmap = ListedColormap([
        "blue",    # 0 water
        "green",   # 1 vegetation
        "red",     # 2 urban
        "yellow",  # 3 mixed
        "gray"     # 4 background/unknown
    ])

    bounds = np.arange(-0.5, 5.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # --- clean invalid labels ---
    vis_plot = labels.copy()
    vis_plot[vis_plot < 0] = 4  # unknown/background

    plt.figure(figsize=(8, 6))
    img = plt.imshow(vis_plot, cmap=cmap, norm=norm)

    cbar = plt.colorbar(img, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels([
        "water",
        "vegetation",
        "urban",
        "mixed",
        "unknown"
    ])

    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def map_clusters_to_classes(labels, centers, shape):

    # compute SAR physics indicators
    vv = centers[:, 0]
    vh = centers[:, 1]
    diff = vv - vh

    # --- identify WATER (darkest cluster) ---
    water = np.argmin(vv + vh)

    # --- identify URBAN (highest VV-VH contrast) ---
    urban = np.argmax(diff)

    # --- identify VEGETATION (moderate VH, not extreme diff) ---
    remaining = set(range(len(centers))) - {water, urban}

    veg = min(remaining, key=lambda i: abs(diff[i]))

    cluster_to_class = {
        water: 0,
        veg: 1,
        urban: 2
    }

    vis = np.full(shape, 3, dtype=np.int32)

    for k, cls in cluster_to_class.items():
        vis[labels == k] = cls

    return vis


def plot_raw_clusters(
        labels: np.ndarray, 
        filename: str, 
        title: str = "Clusters") -> None:
    """
    Plot raw cluster output without any class interpretation.

    Parameters
    ----------
    labels : np.ndarray
        2D array with cluster IDs (e.g. 0,1,2,3,-1)
    filename : str
        Output image path
    title : str
        Plot title

    Returns
    -------
    None
    """

    # --- simple colormap for raw clusters ---
    cmap = ListedColormap([
        "blue",     # cluster 0
        "yellow",    # cluster 1
        "red",      # cluster 2
        "green",   # cluster 3
        "purple",   # cluster 4 (if exists)
        "gray"      # fallback
    ])

    # handle -1 (invalid pixels)
    vis = labels.copy()
    vis[vis < 0] = 5

    bounds = np.arange(-0.5, 6.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(8, 6))
    plt.imshow(vis, cmap=cmap, norm=norm)

    cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4, 5])
    cbar.ax.set_yticklabels([
        "0",
        "1",
        "2",
        "3",
        "4",
        "invalid"
    ])

    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def combine_hybrid(labels_phys, labels_other_method):
    vis = np.full(labels_phys.shape, 4, dtype=np.int32)

    # 1. WATER - use phys
    vis[labels_phys == 0] = 0

    # 2. VEGETATION - use phys 
    vis[(labels_phys == 1) & (vis != 0)] = 1

    # 3. URBAN (only if other method agrees OR physics suggests)
    vis[(labels_phys == 2) | (labels_other_method == 2)] = 2

    # 4. MIXED = disagreement regions
    disagreement = labels_phys != labels_other_method
    vis[disagreement & (vis == 4)] = 3

    return vis


def create_sar_folium_map(
    vis: np.ndarray,
    raster_path: str,
    output_html: str = "sar_map.html",
    opacity: float = 0.6,
    zoom_start: int = 10
) -> None:
    """
    Overlay SAR classification map on OpenStreetMap using folium.

    Parameters
    ----------
    vis : np.ndarray
        2D array with class labels:
        0 = water
        1 = vegetation
        2 = urban
        3 = mixed/background

    raster_path : str
        Path to original SAR GeoTIFF (for georeferencing)

    output_html : str
        Output HTML file

    opacity : float
        Overlay transparency

    zoom_start : int
        Initial zoom level

    Returns
    -------
    None
    """

    # -------------------------
    # 1. Convert classes → RGB
    # -------------------------
    h, w = vis.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # water (blue)
    rgb[vis == 0] = [0, 0, 255]

    # vegetation (green)
    rgb[vis == 1] = [0, 255, 0]

    # urban (red)
    rgb[vis == 2] = [255, 0, 0]

    # mixed (yellow)
    rgb[vis == 3] = [255, 255, 0]

    # -------------------------
    # 2. Read georeference
    # -------------------------
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    latlon_bounds = transform_bounds(
        crs,
        "EPSG:4326",
        *bounds
    )

    # folium expects: [[south, west], [north, east]]
    overlay_bounds = [
        [latlon_bounds[1], latlon_bounds[0]],
        [latlon_bounds[3], latlon_bounds[2]]
    ]

    # -------------------------
    # 3. Create base map
    # -------------------------
    center_lat = (overlay_bounds[0][0] + overlay_bounds[1][0]) / 2
    center_lon = (overlay_bounds[0][1] + overlay_bounds[1][1]) / 2

    map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # -------------------------
    # 4. Add overlay
    # -------------------------
    img = Image.fromarray(rgb)

    folium.raster_layers.ImageOverlay(
        image=np.array(img),
        bounds=overlay_bounds,
        opacity=opacity,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(map)

    folium.LayerControl().add_to(map)

    # -------------------------
    # 5. Save
    # -------------------------
    map.save(output_html)

if __name__ == "__main__":

    # Initialization and agruments parsing

    logging.basicConfig(level=logging.INFO)
    logging.info("Hello from SAR processing script!")
    logging.info("Python version: {}".format(sys.version))

    #Parse command line arguments 
    parser = argparse.ArgumentParser(description="Script for SAR data processing")
    parser.add_argument("--read_from_file", type=bool, required=False, help="Define source of data")

    args = parser.parse_args()
    if args.read_from_file:
        read_from_file = args.read_from_file
    else:
        args.read_from_file = True

    ROOT_DIR = Path(__file__).resolve().parent

    logging.info("VV % VH SAR data will be read from files in folder - tiffs")
    #could be extended with reading from https://catalogue.dataspace.copernicus.eu/stac API

    tiff_directory = os.path.join(ROOT_DIR, "tiffs")
    output_directory = os.path.join(ROOT_DIR, "artifacts")
    os.makedirs(output_directory, exist_ok=True)

    # Load tiff names 
    tiff_names_file = os.path.join(tiff_directory, "TIFF_NAMES.txt")

    # Read file names into a list
    with open(tiff_names_file, "r") as f:
        log_files = [line.strip() for line in f if line.strip()]
    
    vv_file_name = log_files[0] #VV 
    vh_file_name = log_files[1] #VH

    logging.info(f"Opening tiff files: {vv_file_name} & {vh_file_name} as specified in TIFF_NAMES.txt")
    try: 
        vv = rasterio.open(f"{ROOT_DIR}/tiffs/{vv_file_name}.tiff").read(1)
        vh = rasterio.open(f"{ROOT_DIR}/tiffs/{vh_file_name}.tiff").read(1)
    except FileNotFoundError:
        logging.info(f"List file not found: {tiff_names_file}")
    except Exception as e:
        logging.info(f"Something went wrong: {e}")

    #Filtering and features extraction
    logging.info("Filtering and features extraction")

    vv_filtered, vh_filtered, ratio, vv_db, vh_db = filter_and_extract_features(vv = vv, vh = vh)

    logging.info("Saving baseline plots")
    
    save_plot_to_file(vv_db, "artifacts/baseline_vv.png", title="VV baseline dB")
    save_plot_to_file(vh_db, "artifacts/baseline_vh.png", title="VH baseline dB")

    save_plot_to_file(vv_filtered, "artifacts/vv_filtered.png", title="VV filtered")
    save_plot_to_file(vh_filtered, "artifacts/vh_filtered.png", title="VH filtered")
    save_plot_to_file(ratio, "artifacts/ratio.png", title="VV-VH ratio")

    ####################
    # CLUSTERING
    ####################

    logging.info("Starting clustering process")

    logging.info("Approach 1: KMeans")
    
    kmeans_labels, kmeans_centers = kmeans_clustering(vv_filtered, vh_filtered, ratio)
    logging.info("Saving clustered image - KMeans")

    plot_raw_clusters(
        kmeans_labels,
        "artifacts/kmeans_raw.png",
        title="Raw KMeans clusters"
    )

    labels_gmm = map_clusters_to_classes(
                    kmeans_labels,
                    kmeans_centers,
                    vv_filtered.shape
                    )

    save_clustered_data(labels_gmm, "artifacts/kmeans_result.png")
    
    logging.info("Approach 2: GMM")
    
    gmm_labels, gmm_centers = gmm_clustering(vv_filtered, vh_filtered, ratio)
    logging.info("Saving clustered image - GMM")

    plot_raw_clusters(
        gmm_labels,
        "artifacts/gmm_raw.png",
        title="Raw GMM clusters"
    )
    labels_gmm = map_clusters_to_classes(
                    gmm_labels,
                    gmm_centers,
                    vv_filtered.shape
                    )

    save_clustered_data(labels_gmm, "artifacts/gmm_result.png")


    logging.info("Approach 3: GMM + texture")
    
    tex_labels, tex_centers = gmm_clustering_texture(vv_filtered, vh_filtered, ratio)
    logging.info("Saving clustered image - GMM with texture features")

    plot_raw_clusters(
        tex_labels,
        "artifacts/tex_raw.png",
        title="Raw texture GMM clusters"
    )

    labels_tex = map_clusters_to_classes(
                    tex_labels,
                    tex_centers,
                    vv_filtered.shape
                    )

    save_clustered_data(labels_tex, "artifacts/tex_gmm_result.png")

    
    logging.info("Approach 4: Heuristic rulset")

    phys_labels = sar_physics_labels(vv_filtered, vh_filtered)
    logging.info("Saving clustered image - heuristic ruleset")

    plot_raw_clusters(
        phys_labels,
        "artifacts/physics_raw.png",
        title="Raw physics based clusters"
    )

    plot_physics_labels(phys_labels, "artifacts/physics_map.png")

    # Combine physics based and other methods
    vis_hybrid_phys_knn = combine_hybrid(phys_labels, kmeans_labels)
    save_clustered_data(vis_hybrid_phys_knn, "artifacts/vis_hybrid_phys_knn.png")

    vis_hybrid_phys_gmm = combine_hybrid(phys_labels, labels_gmm)
    save_clustered_data(vis_hybrid_phys_gmm, "artifacts/vis_hybrid_phys_gnn.png")

    vis_hybrid_phys_gmm_tex = combine_hybrid(phys_labels, tex_labels)
    save_clustered_data(vis_hybrid_phys_gmm_tex, "artifacts/vis_hybrid_phys_gmm_tex.png")
    
    # Overlay on OSM for easier assesment of results 
    
    vv_path = (f"{ROOT_DIR}/tiffs/{vv_file_name}.tiff")

    create_sar_folium_map(
        vis_hybrid_phys_knn,
        raster_path=vv_path,
        output_html="artifacts/OSM_overlay_vis_hybrid_phys_knn.html"
        )
    
    create_sar_folium_map(
        vis_hybrid_phys_gmm,
        raster_path=vv_path,
        output_html="artifacts/OSM_overlay_vis_hybrid_phys_gmm.html"
        )
    
    create_sar_folium_map(
        vis_hybrid_phys_gmm_tex,
        raster_path=vv_path,
        output_html="artifacts/OSM_overlay_vis_hybrid_phys_gmm_tex.html"
        )







    


