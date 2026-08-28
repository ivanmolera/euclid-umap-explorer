from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_TITLE_LINE_1 = "Euclid Strong-lensing Candidate Observation"
APP_TITLE_LINE_2 = "and Projection Explorer"
APP_TITLE = f"{APP_TITLE_LINE_1} {APP_TITLE_LINE_2}"
APP_VERSION = "v0.1.14"
EUCLID_LOGO_PATH = PROJECT_ROOT / "assets" / "euclid_logo.png"
EUCLID_FAVICON_PATH = PROJECT_ROOT / "assets" / "favicon.png"

MORPH_PATH = os.getenv(
    "MORPH_PATH",
    "/content/drive/MyDrive/catalogues/morphology_catalogue/morphology_catalogue.parquet",
)
CUTOUT_BASE = os.getenv(
    "CUTOUT_BASE",
    "/content/drive/MyDrive/catalogues/morphology_catalogue/cutouts_jpg_gz_arcsinh_vis_only",
)
PARQUET_PATH = os.getenv(
    "PARQUET_PATH",
    "/content/drive/MyDrive/catalogues/morphology_catalogue/representations_pca_40.parquet",
)
STRAIGHT_LINE_FILTERED_PARQUET_PATH = os.getenv(
    "STRAIGHT_LINE_FILTERED_PARQUET_PATH",
    "gs://euclid-umap-ivan-0424-data/catalogues/morphology_catalogue/"
    "representations_pca_40_artifacts_filtered_v2_global_hough_lines.parquet",
)
STRAIGHT_LINE_ARTIFACT_EXAMPLE_PATHS = tuple(
    PROJECT_ROOT / "assets" / "straight_line_artifacts" / f"artifact_{index:02d}.jpg"
    for index in range(1, 7)
)
LENS_PATH = os.getenv(
    "LENS_PATH",
    "/content/drive/MyDrive/catalogues/strong_lensing_catalogue/q1_discovery_engine_lens_catalog.csv",
)
LENS_IMG_BASE = os.getenv(
    "LENS_IMG_BASE",
    "/content/drive/MyDrive/catalogues/strong_lensing_catalogue/lens",
)
CACHE_DIR = Path(
    os.getenv("EUCLID_CACHE_DIR", Path.home() / ".cache" / "euclid-umap-explorer")
)
USE_LOCAL_CACHE = os.getenv("EUCLID_USE_LOCAL_CACHE", "1") != "0"
MAX_ALGORITHM_SECONDS = int(os.getenv("EUCLID_MAX_ALGORITHM_SECONDS", "600"))
DEFAULT_BIRCH_THRESHOLD = 8.0
DEFAULT_BIRCH_BRANCHING_FACTOR = 2
DEFAULT_BIRCH_BATCH_SIZE = 25_000

DEFAULT_CLUSTER_FEATURES = [
    "feat_pca_6",
    "feat_pca_0",
    "feat_pca_1",
    "feat_pca_27",
    "feat_pca_12",
    "feat_pca_29",
]
DEFAULT_LENS_GRADES = ["A", "B", "C"]
LENS_GRADE_OPTIONS = ["A", "B", "C"]
SUMMARY_RANDOM_OBJECTS = 3
SUMMARY_LENS_OBJECTS = 5
SUMMARY_THUMBNAIL_WIDTH = 90
SUMMARY_THUMBNAIL_PIXEL_SIZE = 180
SUMMARY_HISTOGRAM_BINS = 24
SUMMARY_HISTOGRAM_FEATURE_LIMIT = 6
SUMMARY_DISTPLOT_MAX_POINTS_PER_GROUP = 5_000
PCA_FILTER_RECOMMENDATION_MIN_OBJECTS = 25
PCA_FILTER_RECOMMENDATION_MIN_LENSES = 1
PCA_FILTER_RECOMMENDATION_MIN_RECALL = 0.05
DOWNLOAD_MAX_UMAP_ROWS = 5_000
DENDROGRAM_MAX_OBJECTS = 800
DENDROGRAM_TRUNCATE_CLUSTERS = 60
IMAGE_BYTES_CACHE_MAX_ITEMS = 256
DISPLAY_IMAGE_CACHE_MAX_ITEMS = 128
DISPLAY_IMAGE_MAX_PIXEL_SIZE = 900
DISPLAY_IMAGE_RENDER_SIZE_PX = 420
ARC_DETECTION_CACHE_MAX_ITEMS = 64
THUMBNAIL_CACHE_MAX_ITEMS = 512
PCA_FILTER_OPERATORS = [">", ">=", "<", "<=", "between"]
LENS_GRADE_HELP = (
    "Grade A: secure or almost secure lens candidates with clear lensing features "
    "(expert score > 2.0).\n\n"
    "Grade B: probable lens candidates requiring additional confirmation "
    "(expert score > 1.5).\n\n"
    "Grade C: possible lens candidates with lens-like morphology that may still "
    "be explained by other physical structures (expert score > 1.0)."
)
STRAIGHT_LINE_ARTIFACT_FILTER_HELP = (
    "Straight-line artifacts are identified with Canny edge detection and a "
    "probabilistic Hough transform. Candidate lines must be long, contain "
    "continuous high-response pixels, contrast with their immediate surroundings, "
    "and reach the image borders. When enabled, BIRCH uses the PCA catalogue from "
    "which images containing these high-confidence artifacts were removed."
)
PARAMETER_HELP = {
    "threshold": (
        "BIRCH radius threshold. Larger values create broader clusters; "
        "smaller values split the data into more compact groups."
    ),
    "branching_factor": (
        "Maximum number of subclusters kept at each BIRCH tree node. Higher "
        "values can preserve more structure, with extra memory cost."
    ),
    "batch_size": (
        "Number of catalogue rows processed at once during BIRCH fitting and "
        "prediction. Larger batches can be faster but use more memory."
    ),
    "n_neighbors": (
        "UMAP neighborhood size. Lower values emphasize local structure; higher "
        "values preserve broader global structure."
    ),
    "min_dist": (
        "Minimum distance between nearby points in the UMAP layout. Lower values "
        "form tighter groups; higher values spread points out."
    ),
    "Maximum objects": (
        "Upper limit for objects drawn in the UMAP view, used to keep interaction "
        "responsive on large clusters."
    ),
}
