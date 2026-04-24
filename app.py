from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import umap
from PIL import Image
from sklearn.preprocessing import StandardScaler


APP_TITLE = "Euclid UMAP Explorer"
BASE_DIR = Path(__file__).resolve().parent
MORPHOLOGY_DIR = BASE_DIR / "data" / "morphology"
STRONG_LENSES_DIR = BASE_DIR / "data" / "strong_lenses"
CUTOUTS_DIR = BASE_DIR / "data" / "cutouts"

CATALOG_EXTENSIONS = {".csv", ".parquet", ".pq", ".feather"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
ID_COLUMN_CANDIDATES = ("object_id", "OBJECT_ID", "obj_id", "id", "ID")


def list_catalog_files(directory: Path) -> list[Path]:
    """Return supported local catalogue files from a data directory."""
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in CATALOG_EXTENSIONS
    )


@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    """Load a local catalogue file.

    This is intentionally isolated so future versions can swap local files for
    Google Drive, Google Cloud Storage, or signed URL loaders.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    if suffix == ".feather":
        return pd.read_feather(file_path)

    raise ValueError(f"Unsupported catalogue format: {file_path.suffix}")


def find_id_column(df: pd.DataFrame) -> str | None:
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    normalized = {column.lower(): column for column in df.columns}
    return normalized.get("object_id")


def normalize_object_ids(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def add_lens_flag(morphology: pd.DataFrame, lenses: pd.DataFrame) -> pd.DataFrame:
    if "object_id" not in morphology.columns:
        raise ValueError("The morphology catalogue must contain an object_id column.")

    lens_id_column = find_id_column(lenses)
    if lens_id_column is None:
        raise ValueError(
            "The strong lensing catalogue must contain an object_id-like column."
        )

    lens_ids = set(normalize_object_ids(lenses[lens_id_column]).dropna())
    merged = morphology.copy()
    merged["_object_id_key"] = normalize_object_ids(merged["object_id"])
    merged["is_lens"] = merged["_object_id_key"].isin(lens_ids)
    merged = merged.drop(columns=["_object_id_key"])
    return merged


def detect_pca_columns(df: pd.DataFrame) -> list[str]:
    def pca_index(column: str) -> int:
        try:
            return int(column.removeprefix("feat_pca_"))
        except ValueError:
            return 10_000

    pca_columns = [column for column in df.columns if column.startswith("feat_pca_")]
    return sorted(pca_columns, key=pca_index)


def detect_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"cluster", "is_lens"}
    excluded.update(
        column
        for column in df.columns
        if column.lower() == "id"
        or column.lower() == "object_id"
        or column.lower().endswith("_id")
    )
    excluded.update({"id_str", "hdf5_loc"})

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in excluded]


def select_cluster(df: pd.DataFrame, cluster_value: object | None) -> pd.DataFrame:
    if cluster_value is None or "cluster" not in df.columns:
        return df
    return df[df["cluster"].astype("string") == str(cluster_value)]


def sample_for_display(df: pd.DataFrame, max_objects: int) -> pd.DataFrame:
    if len(df) <= max_objects:
        return df.copy()
    return df.sample(n=max_objects, random_state=42).copy()


@st.cache_data(show_spinner=True)
def compute_umap_embedding(
    data: pd.DataFrame,
    selected_features: list[str],
    n_neighbors: int,
    min_dist: float,
) -> pd.DataFrame:
    clean = data.dropna(subset=selected_features).copy()
    if clean.empty:
        return clean

    n_neighbors = min(n_neighbors, max(2, len(clean) - 1))
    scaled = StandardScaler().fit_transform(clean[selected_features])
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42,
    )
    embedding = reducer.fit_transform(scaled)
    clean["umap_x"] = embedding[:, 0]
    clean["umap_y"] = embedding[:, 1]
    return clean


def find_cutout_path(row: pd.Series) -> Path | None:
    identifiers: list[str] = []
    for column in ("id_str", "object_id"):
        value = row.get(column)
        if pd.notna(value):
            identifiers.append(str(value).strip())

    if not identifiers or not CUTOUTS_DIR.exists():
        return None

    for identifier in identifiers:
        for extension in IMAGE_EXTENSIONS:
            candidate = CUTOUTS_DIR / f"{identifier}{extension}"
            if candidate.exists():
                return candidate
    return None


def selected_point_index(event: object) -> int | None:
    if not event:
        return None

    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return None

    if not points:
        return None

    customdata = points[0].get("customdata")
    if isinstance(customdata, Iterable) and not isinstance(customdata, str):
        customdata = list(customdata)[0] if customdata else None

    try:
        return int(customdata)
    except (TypeError, ValueError):
        return None


def show_object_details(row: pd.Series, selected_features: list[str]) -> None:
    st.subheader("Objeto seleccionado")

    details = {
        "id_str": row.get("id_str", ""),
        "object_id": row.get("object_id", ""),
        "cluster": row.get("cluster", ""),
        "is_lens": bool(row.get("is_lens", False)),
        "umap_x": row.get("umap_x", np.nan),
        "umap_y": row.get("umap_y", np.nan),
    }
    st.dataframe(pd.DataFrame([details]), use_container_width=True, hide_index=True)

    st.markdown("**Variables usadas en UMAP**")
    st.dataframe(
        pd.DataFrame(
            [{"variable": feature, "value": row.get(feature)} for feature in selected_features]
        ),
        use_container_width=True,
        hide_index=True,
    )

    cutout_path = find_cutout_path(row)
    if cutout_path is None:
        st.info("No se encontró una imagen local asociada en data/cutouts/.")
        return

    image = Image.open(cutout_path)
    st.image(image, caption=cutout_path.name, use_container_width=True)


def render_empty_state() -> None:
    st.info(
        "Coloca los catálogos descargados manualmente en data/morphology/ "
        "y data/strong_lenses/ para empezar."
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    morphology_files = list_catalog_files(MORPHOLOGY_DIR)
    lens_files = list_catalog_files(STRONG_LENSES_DIR)

    if not morphology_files or not lens_files:
        render_empty_state()
        st.stop()

    with st.sidebar:
        st.header("Datos")
        morphology_path = st.selectbox(
            "Catálogo morfológico",
            morphology_files,
            format_func=lambda path: path.name,
        )
        lens_path = st.selectbox(
            "Catálogo strong lenses",
            lens_files,
            format_func=lambda path: path.name,
        )

    try:
        morphology_df = load_catalog(str(morphology_path))
        lens_df = load_catalog(str(lens_path))
        data = add_lens_flag(morphology_df, lens_df)
    except Exception as exc:
        st.error(f"No se pudieron cargar o cruzar los catálogos: {exc}")
        st.stop()

    pca_columns = detect_pca_columns(data)
    numeric_columns = detect_numeric_feature_columns(data)
    default_features = pca_columns[: min(10, len(pca_columns))] or numeric_columns[:5]

    with st.sidebar:
        st.header("UMAP")
        cluster_value = None
        if "cluster" in data.columns:
            clusters = sorted(data["cluster"].dropna().astype("string").unique())
            selected_cluster = st.selectbox("Cluster", ["Todos"] + clusters)
            cluster_value = None if selected_cluster == "Todos" else selected_cluster
        else:
            st.caption("No se encontró columna cluster.")

        filtered = select_cluster(data, cluster_value)

        feature_options = numeric_columns
        selected_features = st.multiselect(
            "Variables",
            feature_options,
            default=[feature for feature in default_features if feature in feature_options],
        )

        n_neighbors = st.slider("n_neighbors", 2, 100, 15)
        min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, step=0.01)
        max_objects = st.slider("Máximo de objetos", 100, 50_000, 5_000, step=100)

    if not selected_features:
        st.warning("Selecciona al menos una variable numérica para calcular UMAP.")
        st.stop()

    display_data = sample_for_display(filtered, max_objects)

    if len(display_data) < 3:
        st.warning("Se necesitan al menos 3 objetos para calcular UMAP.")
        st.stop()

    embedding_df = compute_umap_embedding(
        display_data,
        selected_features,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )

    if embedding_df.empty:
        st.warning("No quedan objetos con valores completos para las variables seleccionadas.")
        st.stop()

    embedding_df = embedding_df.reset_index(drop=True)
    embedding_df["point_index"] = embedding_df.index
    embedding_df["lens_label"] = np.where(embedding_df["is_lens"], "Lens", "No lens")

    summary_left, summary_middle, summary_right = st.columns(3)
    summary_left.metric("Objetos en cluster/filtro", f"{len(filtered):,}")
    summary_middle.metric("Objetos visualizados", f"{len(embedding_df):,}")
    summary_right.metric("Lentes visualizados", f"{int(embedding_df['is_lens'].sum()):,}")

    hover_columns = [
        column
        for column in ("id_str", "object_id", "cluster", "lens_label")
        if column in embedding_df.columns
    ]

    fig = px.scatter(
        embedding_df,
        x="umap_x",
        y="umap_y",
        color="lens_label",
        symbol="lens_label",
        custom_data=["point_index"],
        hover_data=hover_columns,
        color_discrete_map={"Lens": "#d62728", "No lens": "#1f77b4"},
        symbol_map={"Lens": "star", "No lens": "circle"},
        labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2", "lens_label": "Tipo"},
        height=680,
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.75})
    fig.update_layout(
        legend_title_text="Objeto",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        clickmode="event+select",
    )

    left, right = st.columns([2, 1])
    with left:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="umap_selection",
        )

    selected_index = selected_point_index(event)
    with right:
        if selected_index is None:
            st.info("Selecciona un punto del mapa para ver sus detalles.")
        else:
            show_object_details(embedding_df.loc[selected_index], selected_features)


if __name__ == "__main__":
    main()
