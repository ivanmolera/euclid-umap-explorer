from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_TITLE_LINE_1 = "Euclid Strong-lensing Candidate Observation"
APP_TITLE_LINE_2 = "and Projection Explorer"
APP_TITLE = f"{APP_TITLE_LINE_1} {APP_TITLE_LINE_2}"
APP_VERSION = "v0.1.30"
EUCLID_LOGO_PATH = PROJECT_ROOT / "assets" / "euclid_logo.png"
EUCLID_FAVICON_PATH = PROJECT_ROOT / "assets" / "favicon.png"
LENS_GRADE_EXAMPLE_PATHS = {
    grade: tuple(
        PROJECT_ROOT
        / "assets"
        / "lens_grades"
        / f"grade_{grade.lower()}_{index}.jpg"
        for index in range(1, 4)
    )
    for grade in ("A", "B", "C")
}

MORPH_PATH = os.getenv(
    "MORPH_PATH",
    "/content/drive/MyDrive/catalogues/morphology_catalogue/morphology_catalogue.parquet",
)
PHYSICAL_MEASUREMENTS_PATH = os.getenv(
    "PHYSICAL_MEASUREMENTS_PATH",
    "gs://euclid-umap-ivan-0424-data/catalogues/morphology_catalogue/"
    "useful_physical_measurements.parquet",
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
    "representations_pca_40_artifacts_filtered_v3_2_continuous_multiscale_hough_lines.parquet",
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
DEFAULT_BIRCH_THRESHOLD = 6.2
DEFAULT_BIRCH_BRANCHING_FACTOR = 50
FILTERED_BIRCH_THRESHOLD = 5.5
FILTERED_BIRCH_BRANCHING_FACTOR = 42
DEFAULT_BIRCH_BATCH_SIZE = 25_000
DEFAULT_UMAP_CLUSTER_MIN_OBJECTS = 100
ALL_PCA_FEATURES = tuple(f"feat_pca_{index}" for index in range(40))
LENS_DISPLACED_PCA_FEATURES = (
    "feat_pca_6",
    "feat_pca_0",
    "feat_pca_12",
    "feat_pca_1",
    "feat_pca_27",
    "feat_pca_10",
    "feat_pca_8",
    "feat_pca_13",
)
DEFAULT_BIRCH_FEATURES = ALL_PCA_FEATURES
DEFAULT_BIRCH_SCALING = "standard"
DEFAULT_UMAP_N_NEIGHBORS = 10
DEFAULT_UMAP_MIN_DIST = 0.0
DEFAULT_SEMISUPERVISED_UMAP_N_NEIGHBORS = 10
DEFAULT_SEMISUPERVISED_UMAP_MIN_DIST = 0.15
DEFAULT_SEMISUPERVISED_UMAP_TARGET_WEIGHT = 0.5

DEFAULT_CLUSTER_FEATURES = list(LENS_DISPLACED_PCA_FEATURES)
DEFAULT_LENS_GRADES = ["A", "B", "C"]
LENS_GRADE_OPTIONS = ["A", "B", "C"]
UMAP_POINT_COLORS = {
    "Grade A": "#d62728",
    "Grade B": "#ff7f0e",
    "Grade C": "#f2c94c",
    "Lens candidate": "#d62728",
    "Unknown": "#6f9fc5",
    "Canonical": "#2ca02c",
    "Anomaly": "#111111",
}
UMAP_POINT_SYMBOLS = {
    "Grade A": "circle",
    "Grade B": "square",
    "Grade C": "cross",
    "Lens candidate": "circle",
    "Unknown": "circle",
    "Canonical": "diamond",
    "Anomaly": "x",
}
UMAP_POINT_ORDER = [
    "Grade A",
    "Grade B",
    "Grade C",
    "Unknown",
    "Canonical",
    "Anomaly",
]
SUMMARY_RANDOM_OBJECTS = 3
SUMMARY_LENS_OBJECTS = 5
SUMMARY_VISUAL_CLUSTER_LIMIT = 20
SUMMARY_THUMBNAIL_WIDTH = 90
SUMMARY_THUMBNAIL_PIXEL_SIZE = 180
SUMMARY_HISTOGRAM_BINS = 24
SUMMARY_HISTOGRAM_FEATURE_LIMIT = 6
SUMMARY_DISTPLOT_MAX_POINTS_PER_GROUP = 5_000
PCA_FILTER_RECOMMENDATION_MIN_OBJECTS = 25
PCA_FILTER_RECOMMENDATION_MIN_LENSES = 1
PCA_FILTER_RECOMMENDATION_MIN_RECALL = 0.05
PHYSICAL_FILTER_RECOMMENDATION_MIN_VALID_OBJECTS = 25
PHYSICAL_FILTER_RECOMMENDATION_MIN_LENSES = 2
PHYSICAL_FILTER_RECOMMENDATION_MIN_UNKNOWNS = 2
DOWNLOAD_MAX_UMAP_ROWS = 5_000
DENDROGRAM_MAX_OBJECTS = 800
DENDROGRAM_TRUNCATE_CLUSTERS = 60
IMAGE_BYTES_CACHE_MAX_ITEMS = 256
DISPLAY_IMAGE_CACHE_MAX_ITEMS = 128
DISPLAY_IMAGE_MAX_PIXEL_SIZE = 900
DISPLAY_IMAGE_RENDER_SIZE_PX = 420
ARC_DETECTION_CACHE_MAX_ITEMS = 64
ENABLE_ARC_LIKE_STRUCTURE_DETECTION = (
    os.getenv("EUCLID_ENABLE_ARC_DETECTION", "0") == "1"
)
ENABLE_SLED_CHECK = os.getenv("EUCLID_ENABLE_SLED", "0") == "1"
ENABLE_EXCLUDED_ARTIFACT_ASSIGNMENT_UI = (
    os.getenv("EUCLID_ENABLE_EXCLUDED_ARTIFACT_UI", "0") == "1"
)
THUMBNAIL_CACHE_MAX_ITEMS = 512
PCA_FILTER_OPERATORS = [">", ">=", "<", "<=", "between"]
LABEL_DENSITY_HELP = (
    "**Labelled candidate fraction**\n\n"
    "$$\n"
    r"p_c = \frac{L_c}{N_c}"
    "\n$$\n\n"
    r"$L_c$: candidates matching the selected reference grades. "
    r"$N_c$: all objects in cluster $c$, including Unknown. "
    r"Displayed as $100p_c$%, before PCA/physical filters."
    "\n\n"
    "This measures reference labels, not spatial density or lens probability. "
    "Unknown is not a confirmed non-lens."
)
ENRICHMENT_HELP = (
    "**Label enrichment**\n\n"
    "$$\n"
    r"E_c = \frac{L_c/N_c}{L/N}"
    "\n$$\n\n"
    r"$L_c$, $N_c$: labelled candidates and all objects in cluster $c$. "
    r"$L$, $N$: corresponding counts in the catalogue analysed in this run. "
    "Uses the same selected grades, before PCA/physical filters.\n\n"
    r"$E_c = 1$: catalogue average; above 1: over-representation; below 1: "
    r"under-representation. If $L = 0$, undefined (displayed as 0).\n\n"
    "Not lens probability, purity or significance. Check cluster size and labelled "
    "counts. Candidates are not necessarily confirmed lenses; Unknown is not a "
    "confirmed non-lens."
)
LABEL_ENTROPY_HELP = (
    "**Binary Shannon entropy**\n\n"
    "$$\n"
    r"H = -p\log_2 p - (1-p)\log_2(1-p)"
    "\n$$\n\n"
    r"- $H$: entropy of labelled versus Unknown status, in bits." "\n"
    r"- $p = L_c/N_c$: labelled fraction, between 0 and 1 (not a percentage)." "\n"
    r"- $L_c$: candidates matching the selected grades (A/B/C by default), pooled." "\n"
    r"- $N_c$: all objects in cluster $c$, including Unknown." "\n"
    r"- $1-p$: Unknown fraction; $\log_2$: base-2 logarithm."
    "\n\n"
    r"$H = 0$: all labelled or all Unknown; $H = 1$: 50/50 mixture. "
    r"Example: 10 labelled + 90 Unknown gives $p=0.1$, $H \approx 0.469$ bits. "
    r"Convention: $0\log_2 0 = 0$."
    "\n\n"
    "Uses the original cluster before PCA/physical filters. Unknown is not a "
    "confirmed non-lens. Entropy measures label mix, not morphology or lens "
    "probability; interpret alongside enrichment and labelled counts."
)
LENS_GRADE_HELP = {
    "A": (
        "High-confidence reference candidates with clear lensing features. "
        "The catalogue grade does not by itself establish spectroscopic or modelling confirmation."
    ),
    "B": (
        "Probable reference lens candidates requiring additional confirmation."
    ),
    "C": (
        "Lower-confidence reference candidates whose lens-like features may have "
        "other explanations. This is not a confirmed non-lens category."
    ),
}
LENS_GRADE_HELP_NOTE = (
    "Uses catalogue grades, not expert-score thresholds. Selected grades count "
    "as labelled candidates; others are Unknown, not confirmed non-lenses. "
    "Changes apply on Run clustering. Grades do not train BIRCH."
)
STRAIGHT_LINE_ARTIFACT_FILTER_HELP = (
    "On: use the prefiltered catalogue. Off: use the complete catalogue. "
    "The offline Canny/Hough filter excludes clear, long, continuous straight-line "
    "artifacts and keeps ambiguous cases; it does not edit images. "
    "Changes apply on Run clustering."
)
PARAMETER_HELP = {
    "threshold": (
        "Maximum RMS radius for merging into a BIRCH CF subcluster in standardized "
        "40-PCA space. Larger values generally produce broader groups. "
        "Not an angular radius or a distance limit for every final cluster member. "
        "Lens labels are not used."
    ),
    "branching_factor": (
        "Maximum CF subclusters per tree node before splitting. Controls tree "
        "capacity and memory use, not the number of final clusters. Changing it "
        "can change assignments; larger values do not guarantee better enrichment."
    ),
    "batch_size": (
        "Rows processed per BIRCH fitting/prediction batch. "
        f"Fixed at {DEFAULT_BIRCH_BATCH_SIZE:,} to match precalculated runs. "
        "Cached results require matching data, preprocessing and BIRCH settings."
    ),
    "n_neighbors": (
        "UMAP neighborhood size: lower values emphasize local structure; higher "
        "values emphasize broader structure. Capped at fitted objects minus one. "
        "Does not set the number of clusters."
    ),
    "min_dist": (
        "UMAP layout packing: lower values allow tighter groups; higher values "
        "spread points out. Not a guaranteed minimum separation or physical "
        "distance. Does not change BIRCH membership."
    ),
    "target_weight": (
        "Reference-grade influence on semi-supervised UMAP. At 0, only PCA data "
        "are used. A=2, B=1, C=0 are categories, not confidence weights; "
        "Unknown=-1 is unlabelled. Label-guided proximity is not independent "
        "evidence for lensing."
    ),
    "Maximum objects": (
        "Limit used both to fit and to draw UMAP after filters. Sampling prioritizes "
        "canonical/anomalous objects, then labelled candidates, then random objects "
        "(seed 42). Labels can be over-represented; original cluster metrics "
        "are unchanged."
    ),
}
CLUSTER_EXTREME_HELP = {
    "Canonical": (
        "Observed object nearest the mean in selected PCA components, standardized "
        "within the original cluster. Not a synthetic image or the BIRCH CF centre. "
        "UMAP recalculates this after filters, so its landmark may differ."
    ),
    "Anomalous": (
        "Observed object farthest from the mean in selected PCA components, "
        "standardized within the original cluster. Not a LOF score or lens "
        "probability. UMAP recalculates this after filters, so its landmark may differ."
    ),
}
RECOMMENDED_FILTER_HELP = (
    "Tests >= and <= cuts at the original cluster's 5th-95th percentiles, choosing "
    "the eligible cut with highest enrichment x square root of recall. "
    "Baseline: that cluster's objects with a usable value, not the whole catalogue. "
    "Recall: fraction of labelled candidates retained. Label-based, before active "
    "filters; exploratory, not independent validation."
)
ANALYSIS_FLOW_STEPS = (
    "Run BIRCH on the chosen catalogue using all 40 standardized PCA components, "
    "without lens labels. Matching cached results are reused.",
    "Inspect clusters, ordered by labelled-candidate count, then labelled fraction "
    "and cluster size by default.",
    "Review cutouts, property distributions and label-based recommended cuts.",
    "Apply PCA/physical filters to select a subset; original cluster metrics stay unchanged.",
    "Fit UMAP on selected standardized PCA components, sampling at Maximum objects if needed.",
    "Optionally run Ward clustering in PCA space, then label-guided UMAP on a subcluster.",
    "Export selected or loaded UMAP objects (labelled and Unknown), "
    f"up to {DOWNLOAD_MAX_UMAP_ROWS:,} rows.",
    "Inspect cutouts and Aladin positions. Neither enrichment nor map proximity confirms lensing.",
)
