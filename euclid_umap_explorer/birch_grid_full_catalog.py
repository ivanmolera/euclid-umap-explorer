"""Run the paired BIRCH grid on the unfiltered 40-PCA morphology catalogue."""

from . import birch_grid_benchmark


BUCKET = "gs://euclid-umap-ivan-0424-data"
FILTERED_BENCHMARK = f"{BUCKET}/benchmarks/birch_grid_v32_20261013"


def main() -> None:
    birch_grid_benchmark.main(
        default_parquet=(
            f"{BUCKET}/catalogues/morphology_catalogue/representations_pca_40.parquet"
        ),
        default_objects=380_111,
        default_reference_index=f"{FILTERED_BENCHMARK}/object_index.parquet",
        default_reference_fit=f"{FILTERED_BENCHMARK}/fits/t6.2_b50.npz",
    )


if __name__ == "__main__":
    main()
