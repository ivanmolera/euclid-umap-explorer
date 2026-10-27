"""Exploratory comparison of image evidence with catalogue reference membership."""
from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

from .analysis import add_cluster_extreme_roles
from .catalogs import normalize_object_ids


DIAGNOSTIC_SEED = 42


def diagnostic_signature(data, features, group_column="cluster"):
    columns = ["object_id", group_column, *features]
    columns += [column for column in ("is_lens", "lens_grade", "id_str") if column in data]
    digest = hashlib.sha256()
    digest.update("|".join(columns).encode())
    digest.update(pd.util.hash_pandas_object(data[columns], index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _reference_mask(data, grades=None):
    flags = data["is_lens"].fillna(False).astype(bool)
    if grades is not None:
        flags &= data.get("lens_grade", pd.Series("", index=data.index)).fillna("").isin(grades)
    return flags


def build_diagnostic_profiles(data, features, *, reference=None, group_column="cluster"):
    """Canonical selection is identical to previews and never uses the grades."""
    features = list(features)
    if not features or len(set(features)) != len(features) or any(
        not field.startswith("feat_pca_") for field in features
    ):
        raise ValueError("Use unique catalogue PCA features.")
    required = ["object_id", "id_str", "is_lens", group_column, *features]
    if any(column not in data for column in required):
        raise ValueError("Missing diagnostic input columns.")
    work = data.copy()
    work["object_id"] = normalize_object_ids(work.object_id)
    if work.object_id.isna().any() or work.object_id.duplicated().any():
        raise ValueError("Diagnostics require unique object IDs.")
    if work[group_column].isna().any():
        raise ValueError("Missing group identifier.")
    reference = work if reference is None else reference
    if reference.object_id.duplicated().any() or not set(work.object_id).issubset(
        set(reference.object_id.astype(str))
    ):
        raise ValueError("Reference population must contain each diagnostic object exactly once.")
    values = reference[features].to_numpy(dtype=np.float64)
    complete = np.isfinite(values).all(axis=1)
    if not complete.any():
        raise ValueError("No complete PCA reference vectors.")
    scaler = StandardScaler().fit(values[complete])
    rows, canonicals = [], {}
    for group_id, members in work.groupby(group_column, sort=True):
        group_id = int(group_id)
        members = members.reset_index(drop=True)
        clean = members[np.isfinite(members[features].to_numpy(dtype=float)).all(axis=1)]
        canonical = None
        if len(clean) >= 2:
            marked = add_cluster_extreme_roles(clean, features)
            canonical = marked.loc[marked.is_canonical].iloc[0].copy()
            canonicals[group_id] = canonical
        rms = np.nan
        if len(clean):
            scaled = scaler.transform(clean[features].to_numpy(dtype=float))
            rms = float(np.sqrt(np.mean((scaled - scaled.mean(axis=0)) ** 2)))
        flags = _reference_mask(members)
        grades = members.get("lens_grade", pd.Series("", index=members.index)).fillna("")
        rows.append({
            "cluster": group_id, "n_objects": len(members), "n_lenses": int(flags.sum()),
            "n_unknown": int((~flags).sum()), "n_pca_objects": len(clean), "pca_rms": rms,
            **{f"n_grade_{grade.lower()}": int((flags & grades.eq(grade)).sum()) for grade in "ABC"},
            "canonical": str(canonical.object_id) if canonical is not None else "",
            "canonical_grade": str(canonical.get("lens_grade", "")) if canonical is not None
                and bool(canonical.is_lens) else "",
            "canonical_is_lens": bool(canonical.is_lens) if canonical is not None else False,
        })
    return pd.DataFrame(rows), canonicals


def select_diagnostic_cohort(profiles, *, min_objects=100, max_clusters=200,
                             seed=DIAGNOSTIC_SEED, preview_ids=None):
    if min_objects < 1 or max_clusters < 1:
        raise ValueError("Sample limits must be positive.")
    eligible = profiles.loc[profiles.n_objects.ge(min_objects)].sort_values("cluster")
    if preview_ids is not None:
        eligible = eligible.loc[eligible.cluster.isin(preview_ids)]
    if len(eligible) > max_clusters:
        rng = np.random.default_rng(seed)
        eligible = eligible.iloc[np.sort(rng.choice(len(eligible), max_clusters, replace=False))]
    return eligible.reset_index(drop=True)


def attach_evidence(profiles, evidence):
    result = profiles.copy()
    scores, statuses, regions, flags = [], [], [], []
    for row in result.itertuples():
        item = evidence.get(str(row.canonical), {})
        score = item.get("score")
        if score is not None and (not np.isfinite(score) or not 0 <= score <= 100):
            raise ValueError("Invalid visual score.")
        scores.append(float(score) if score is not None else np.nan)
        statuses.append(item.get("status", "no_canonical" if not row.canonical else "not_processed"))
        regions.append(len(item.get("regions", [])))
        flags.append(item.get("quality_flag", ""))
    return result.assign(visual_score=scores, score_status=statuses,
                         n_regions=regions, image_quality_flag=flags)


def add_canonical_scores(summary, evidence):
    result = summary.copy()
    scores, statuses = [], []
    for canonical in result.canonical:
        item = evidence.get(str(canonical), {})
        score = item.get("score")
        scores.append(float(score) if score is not None else np.nan)
        statuses.append(item.get("status", "not_scored"))
    result["canonical_score"] = scores
    result["canonical_score_status"] = statuses
    return result


def add_reference_comparison(frame, reference, *, grades=None, exclude_canonical=True):
    result = frame.copy()
    reference_flags = _reference_mask(reference, grades)
    n_reference, n_catalogue = int(reference_flags.sum()), len(reference)
    if grades is None:
        counts = result.n_lenses.to_numpy(dtype=int)
        canonical_reference = result.canonical_is_lens.to_numpy(dtype=bool)
    else:
        counts = sum((result[f"n_grade_{g.lower()}"].to_numpy(dtype=int) for g in grades),
                     np.zeros(len(result), dtype=int))
        canonical_reference = result.canonical_is_lens & result.canonical_grade.isin(grades)
        canonical_reference = canonical_reference.to_numpy(dtype=bool)
    remove = (result.canonical.ne("").to_numpy() if exclude_canonical else np.zeros(len(result), dtype=bool))
    removed_reference = remove & canonical_reference
    n_members = result.n_objects.to_numpy(dtype=int) - remove
    member_references = counts - removed_reference
    total = n_catalogue - remove
    total_references = n_reference - removed_reference
    if (member_references < 0).any() or (counts > result.n_objects.to_numpy()).any():
        raise ValueError("Reference counts are inconsistent with group memberships.")
    fractions = np.divide(member_references, n_members, out=np.full(len(result), np.nan), where=n_members > 0)
    base = np.divide(total_references, total, out=np.full(len(result), np.nan), where=total > 0)
    enrichment = np.divide(fractions, base, out=np.full(len(result), np.nan), where=base > 0)
    result["n_reference_full"] = counts
    result["n_reference_compared"] = member_references
    result["n_members_compared"] = n_members
    result["reference_fraction"] = fractions
    result["reference_baseline"] = base
    result["enrichment"] = enrichment
    return result


def diagnostic_statistics(frame):
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["visual_score", "enrichment"])
    output = {"n_compared": len(clean), "spearman": None, "pearson": None}
    if len(clean) >= 3 and clean.visual_score.nunique() > 1 and clean.enrichment.nunique() > 1:
        output.update(spearman=float(spearmanr(clean.visual_score, clean.enrichment).statistic),
                      pearson=float(pearsonr(clean.visual_score, clean.enrichment).statistic))
    return output


def classify_diagnostic_regions(frame, score_threshold=70., enrichment_threshold=10.):
    if not 0 <= score_threshold <= 100 or not np.isfinite(enrichment_threshold) or enrichment_threshold < 0:
        raise ValueError("Invalid diagnostic thresholds.")
    result = frame.copy()
    valid = np.isfinite(result.visual_score) & np.isfinite(result.enrichment)
    high_score = result.visual_score.ge(score_threshold)
    enriched = result.enrichment.ge(enrichment_threshold)
    result["diagnostic_region"] = np.select(
        [valid & high_score & enriched, valid & high_score & ~enriched,
         valid & ~high_score & enriched, valid & ~high_score & ~enriched],
        ["High score / enriched", "High score / low enrichment",
         "Low score / enriched", "Low score / low enrichment"], default="Not evaluable",
    )
    return result


def build_diagnostic_figure(frame, *, score_threshold=70., enrichment_threshold=10., title="Cluster diagnostics"):
    import plotly.graph_objects as go

    clean = frame.loc[np.isfinite(frame.visual_score) & np.isfinite(frame.enrichment)].copy()
    figure = go.Figure()
    custom = [[str(row.cluster), str(row.canonical), int(row.n_objects), int(row.n_reference_compared),
               int(row.n_unknown), row.diagnostic_region] for row in clean.itertuples()]
    figure.add_trace(go.Scatter(
        x=clean.visual_score, y=clean.enrichment, mode="markers", customdata=custom,
        marker={"size": clean.n_unknown, "sizemode": "area", "sizemin": 5,
                "sizeref": max(2. * clean.n_unknown.max() / 48. ** 2, 1.) if len(clean) else 1.,
                "color": clean.pca_rms, "colorscale": "Cividis", "opacity": .85,
                "line": {"width": 1, "color": "#9da6af"},
                "symbol": np.where(clean.n_unknown.eq(0), "x", "circle"),
                "colorbar": {"title": "PCA RMS<br>dispersion", "thickness": 14}},
        hovertemplate=("Cluster %{customdata[0]}<br>Canonical %{customdata[1]}"
                       "<br>Visual evidence %{x:.0f}/100<br>Enrichment %{y:.2f}x"
                       "<br>Objects %{customdata[2]}<br>Compared references %{customdata[3]}"
                       "<br>Unknown %{customdata[4]}<br>%{customdata[5]}<extra></extra>"),
        selected={"marker": {"opacity": 1.}}, unselected={"marker": {"opacity": .65}},
    ))
    figure.add_vline(x=score_threshold, line_dash="dash", line_color="#8a929b")
    figure.add_hline(y=enrichment_threshold, line_dash="dash", line_color="#8a929b")
    figure.update_layout(
        title=title, xaxis_title="Canonical visual evidence (0-100, uncalibrated)",
        yaxis_title="Catalogue-reference enrichment (x)", height=480,
        margin={"l": 15, "r": 20, "t": 45, "b": 15}, showlegend=False,
        clickmode="event+select", dragmode="zoom",
    )
    figure.update_xaxes(range=[-3, 103])
    figure.update_yaxes(rangemode="tozero")
    return figure


def read_validation_labels(content: bytes):
    labels = pd.read_csv(BytesIO(content), dtype={"object_id": "string"})
    return _validate_review_labels(labels)


def _validate_review_labels(labels):
    if not {"object_id", "is_lens"}.issubset(labels):
        raise ValueError("Validation CSV requires object_id and is_lens (0 or 1).")
    labels = labels[["object_id", "is_lens"]].copy()
    if not labels.object_id.dropna().map(lambda value: isinstance(value, str)).all():
        raise ValueError("Validation object IDs must be exact integer strings.")
    labels["object_id"] = labels.object_id.astype("string")
    labels["object_id"] = labels.object_id.str.strip()
    if labels.object_id.isna().any() or not labels.object_id.str.fullmatch(r"-?\d+").all():
        raise ValueError("Object IDs must be integer strings, never floats or scientific notation.")
    if labels.object_id.duplicated().any() or labels.is_lens.isna().any() or not labels.is_lens.isin([0, 1]).all():
        raise ValueError("Use unique IDs and explicit binary validation labels; Unknown is not a negative.")
    return labels


def independent_validation_auc(data, frame, labels, *, group_column="cluster", exclude_canonical=True):
    from sklearn.metrics import roc_auc_score

    labels = _validate_review_labels(labels)
    members = data[["object_id", group_column]].rename(columns={group_column: "cluster"}).copy()
    members["object_id"] = normalize_object_ids(members.object_id)
    if members.object_id.duplicated().any():
        raise ValueError("Duplicate diagnostic memberships.")
    validated = members.merge(labels.rename(columns={"is_lens": "validation_target"}),
                              on="object_id", validate="one_to_one")
    validated = validated.merge(frame[["cluster", "visual_score", "canonical"]],
                                on="cluster", validate="many_to_one")
    if exclude_canonical:
        validated = validated.loc[validated.object_id.ne(validated.canonical)]
    validated = validated.loc[np.isfinite(validated.visual_score)]
    auc = None
    if validated.validation_target.nunique() == 2:
        auc = float(roc_auc_score(validated.validation_target, validated.visual_score))
    return {"auc": auc, "n_validated": len(validated), "n_submitted": len(labels),
            "n_positive": int(validated.validation_target.sum()),
            "n_negative": int(validated.validation_target.eq(0).sum())}
