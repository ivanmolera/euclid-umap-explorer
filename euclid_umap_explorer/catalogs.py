from __future__ import annotations

from typing import Iterable
from pathlib import Path

import pandas as pd
import streamlit as st

from .storage import cached_input_path, is_gcs_path

def detect_pca_columns(df: pd.DataFrame) -> list[str]:
    return detect_pca_column_names(df.columns)

def detect_pca_column_names(columns: Iterable[str]) -> list[str]:
    def pca_index(column: str) -> int:
        try:
            return int(column.removeprefix("feat_pca_"))
        except ValueError:
            return 10_000

    return sorted(
        [column for column in columns if str(column).startswith("feat_pca_")],
        key=pca_index,
    )

def normalize_object_ids(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    normalized = normalized.str.strip("'\"")
    return normalized

def ensure_object_id_from_id_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "object_id" not in df.columns:
        if "id_str" not in df.columns:
            raise ValueError("The PCA parquet must contain id_str or object_id.")
        df["object_id"] = df["id_str"].astype("string").str.split("_").str[-1]
    df["object_id"] = normalize_object_ids(df["object_id"])
    return df

@st.cache_data(show_spinner=False)
def load_pca_catalog(parquet_path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(cached_input_path(parquet_path))
    df = ensure_object_id_from_id_str(df)
    feature_cols = detect_pca_columns(df)

    if not feature_cols:
        raise ValueError("No feat_pca_* columns were found in the PCA parquet file.")

    keep_cols = ["id_str", "object_id", "hdf5_loc", *feature_cols]
    keep_cols = [column for column in keep_cols if column in df.columns]

    work_df = (
        df[keep_cols]
        .dropna(subset=feature_cols)
        .reset_index(drop=True)
        .copy()
    )
    return work_df, feature_cols

def normalize_lens_grades(grades: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted({str(grade).strip().upper() for grade in grades if str(grade).strip()})
    )

@st.cache_data(show_spinner=False)
def load_lens_catalog(lens_path: str, selected_grades: tuple[str, ...]) -> pd.DataFrame:
    lens_df = pd.read_csv(cached_input_path(lens_path), dtype={"object_id": "string"})
    if "object_id" not in lens_df.columns:
        raise ValueError("The lens catalogue must contain object_id.")

    lens_df = lens_df.copy()
    lens_df["object_id"] = normalize_object_ids(lens_df["object_id"])

    if selected_grades and "grade" in lens_df.columns:
        lens_df = lens_df[
            lens_df["grade"].astype(str).str.strip().str.upper().isin(selected_grades)
        ].copy()

    columns = [column for column in ("object_id", "id_str", "grade") if column in lens_df.columns]
    return lens_df[columns].dropna(subset=["object_id"]).drop_duplicates("object_id")

def merge_lens_flags(work_df: pd.DataFrame, lens_df: pd.DataFrame) -> pd.DataFrame:
    lens_meta = lens_df.copy()
    rename_map = {}
    if "id_str" in lens_meta.columns:
        rename_map["id_str"] = "lens_id_str"
    if "grade" in lens_meta.columns:
        rename_map["grade"] = "lens_grade"
    lens_meta = lens_meta.rename(columns=rename_map)

    merged = work_df.merge(lens_meta, on="object_id", how="left")
    lens_object_ids = set(normalize_object_ids(lens_df["object_id"]).dropna())
    merged["is_lens"] = normalize_object_ids(merged["object_id"]).isin(lens_object_ids)

    if "id_str" in work_df.columns and "id_str" in lens_df.columns:
        lens_id_strs = set(lens_df["id_str"].astype("string").str.strip().dropna())
        merged["is_lens"] = merged["is_lens"] | (
            merged["id_str"].astype("string").str.strip().isin(lens_id_strs)
        )

    return merged

@st.cache_data(show_spinner=False)
def load_morphology_object(morph_path: str, object_id: str) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    if not object_id:
        return pd.DataFrame()

    if is_gcs_path(morph_path):
        filesystem, path = pafs.FileSystem.from_uri(morph_path)
        dataset = ds.dataset(path, format="parquet", filesystem=filesystem)
    else:
        path = cached_input_path(morph_path)
        if not Path(path).exists():
            return pd.DataFrame()
        dataset = ds.dataset(path, format="parquet")

    if "object_id" not in dataset.schema.names:
        return pd.DataFrame()

    field_type = dataset.schema.field("object_id").type
    filter_value: object = object_id
    if pa.types.is_integer(field_type):
        try:
            filter_value = int(object_id)
        except ValueError:
            return pd.DataFrame()

    table = dataset.to_table(filter=ds.field("object_id") == filter_value)
    if table.num_rows == 0 and not pa.types.is_string(field_type):
        table = dataset.to_table(filter=ds.field("object_id") == object_id)
    if table.num_rows == 0:
        return pd.DataFrame()

    return table.slice(0, 1).to_pandas()
