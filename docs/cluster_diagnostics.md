# Cluster Diagnostic Matrix

This opt-in exploratory panel compares automatic evidence in a canonical morphology
cutout with the concentration of reference candidates in that group. It is available
in the BIRCH summary and separately for UMAP hierarchical subclusters. Selecting a
diagnostic point changes only the displayed canonical image, not the cluster being
analysed, its membership, PCA, UMAP or the main summary metrics.

The clustering summary includes a sortable `Canonical score` column and exports
the score and its status in the clustering CSV. It shows scores already computed
in the current session for the same catalogue and PCA features, including earlier
diagnostic cohorts. Missing or unevaluable cutouts stay blank, not zero. Computing
a cohort refreshes that summary immediately; hiding the diagnostic panel does not
remove the computed scores. A new clustering result invalidates them.

## Image Score

The fixed detector is ported from the visual-evidence cell of
`notebooks/escope_lens_plausibility_review_colab.ipynb`. It uses the original
morphology JPEG, not a resized thumbnail or the reference-catalogue RGB image.
It normalizes contrast, estimates a centre, thresholds positive residuals in an
annulus and measures tangential alignment, elongation, angular extent, radial
coherence and contrast. Long border lines are masked as possible artifacts.
The score combines the best region (up to 65), a second region (up to 20) and
multi-region geometry (up to 15). No catalogue labels enter this calculation.

The 0-100 result is **uncalibrated geometric evidence, not a lens probability**.
It is not Cheng et al.'s manual assumed-probability classification, nor a
reproduction of that paper's ROC performance. Rings, spiral structure, neighbours,
centring, JPEG processing, noise and artifacts can affect it. A high score can be
a false positive, and a low score does not exclude a lens. Missing, undersized
or insufficient-contrast images are not evaluable, rather than scored as zero.

The detail view preserves the original cutout next to a separate annotated copy:
red regions and numbers, a yellow estimated centre, a cyan median marked radius,
and orange excluded lines. The cyan circle is **not an Einstein-radius estimate**.
Region measurements and score contributions are included for inspection.

## Groups and Sampling

The canonical is the same actual member used by the existing preview: the member
nearest the centroid after within-group standardization of the selected preview
PCA components. These can be fewer than the 40 used for dispersion. Changing this
selection invalidates diagnostic profiles and scores in the summary. Both feature
lists are recorded in the manifest. It is not a synthetic
centroid image and cannot establish the purity of the remaining members.

The default cohort uses only a minimum group size (100 objects) and a seeded
sample of at most 200 eligible groups (seed 42). It has no label-count requirement.
The optional first-20-preview cohort is explicitly labelled as label-prioritized.
Changing sample settings requires another run; other comparison controls reuse
the cached image scores. Scoring stops loading new images after a 120-second
budget; unprocessed images are marked `time_limit`, not negative. A single image
request can still exceed that budget. Detector results are cached by image bytes
and detector version.

The scatter uses visual score on X, reference enrichment on Y, Unknown count as
bubble area (with a minimum visible size; a cross means zero Unknown) and PCA RMS
dispersion as colour. Dispersion is the RMS deviation per coordinate from the
group mean, after applying a StandardScaler fitted on complete PCA vectors in
the current full catalogue reference population. Lower means more compact.
Subcluster dispersion uses that same catalogue scaling, not a separately fitted
scale for each parent or sample.

## Reference Comparison

The reference can use the run's selected grades, A+B, or A only. These are
catalogue candidates, not universal physical confirmations. Unknown means absent
from the selected run labels; it never supplies a negative class. The Unknown
bubble count remains tied to the run when diagnostic reference grades change.

By default each canonical is excluded from both its group counts and the catalogue
baseline for that group's comparison:

```text
E_k = [(L_k - l_c) / (N_k - 1)] / [(L - l_c) / (N - 1)]
```

Here `l_c` is one if the canonical matches the diagnostic reference grades and
zero otherwise. Without this option, `E_k = (L_k/N_k)/(L/N)`. A zero reference
baseline yields an undefined enrichment, not zero. The underlying summary metrics
are not changed. UMAP subcluster enrichment uses the complete current clustering
catalogue as its baseline, not only the selected parent or filtered UMAP subset.

The score and enrichment thresholds (70 and 10x initially) define neutral review
regions. They are adjustable operational boundaries, not calibrated decisions.
No region confirms a cluster, proves a discovery, assigns confidence to its
Unknown members or discards a population.

## Correlation and Validation

Pearson and Spearman are calculated over groups with finite scores and enrichment,
with at least three groups and non-constant values. They are exploratory sample
associations. Prior label-based parameter selection, representative selection,
unequal group sizes and image quality limit their interpretation. Correlation
does not prove optimal PCA retention, generalization or lens discovery.

ROC AUC is unavailable by default. It can be computed only for an uploaded CSV
with exact string `object_id` values and explicit independent `is_lens` labels
(0 or 1), plus acknowledgement that these labels were assessed independently of
the displayed scores. Both classes must match scored groups. Each validated
object inherits its group's canonical score; canonicals are omitted when the
leave-canonical-out option is enabled. Unlabelled catalogue objects never become
negative examples. This is an AUC on the matched reviewed subset, not a global
catalogue AUC, an independent test split by itself, or an expected value of 0.86.

## Export

The ZIP includes the comparison table, marked-region measurements and a manifest
with detector version/configuration, cohort settings/seed, PCA columns, input
signatures, reference grades, exclusion setting, thresholds, library versions,
image source paths/hashes, centres and score contributions. The signatures cover
membership, selected PCA vectors and reference labels. Image files and uploaded
validation labels are not copied into the export. Their source data must be
preserved separately to reproduce the run.
