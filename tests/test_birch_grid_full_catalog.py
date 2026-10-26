import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_umap_explorer.birch_grid_benchmark import load_input, protocol
from euclid_umap_explorer.config import DEFAULT_BIRCH_FEATURES


class FullCatalogueGridTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.parquet = root / "full.parquet"
        self.lenses = root / "lenses.csv"
        self.reference = root / "reference.parquet"
        self.fit = root / "fit.npz"
        ids = [f"Q1_R1_1_{number}" for number in range(5)]
        frame = pd.DataFrame({"id_str": ids})
        for feature in DEFAULT_BIRCH_FEATURES:
            frame[feature] = np.arange(5, dtype=np.float32)
        frame.to_parquet(self.parquet, index=False)
        pd.DataFrame({"object_id": ["0", "3"], "grade": ["A", "B"]}).to_csv(
            self.lenses, index=False
        )
        self.reference_rows = pd.DataFrame({
            "id_str": [ids[0], ids[2], ids[3], ids[4]],
            "object_id": ["0", "2", "3", "4"],
            "grade": ["A", "Unknown", "B", "Unknown"],
            "label_split": ["train", "unknown", "test", "unknown"],
        })
        self.reference_rows.to_parquet(self.reference, index=False)
        np.savez_compressed(self.fit, scaler_mean=np.ones(40), scaler_scale=np.full(40, 2.0))

    def test_reuses_reference_splits_and_scaler(self):
        data = load_input(
            str(self.parquet), str(self.lenses), 5, 2,
            str(self.reference), str(self.fit),
        )
        self.assertEqual(np.flatnonzero(data["train"]).tolist(), [0])
        self.assertEqual(np.flatnonzero(data["val"]).tolist(), [])
        self.assertEqual(np.flatnonzero(data["test"]).tolist(), [3])
        np.testing.assert_allclose(data["z"][:, 0], [-0.5, 0.0, 0.5, 1.0, 1.5])
        self.assertTrue(data["z"].flags.c_contiguous)

    def test_rejects_changed_reference_order(self):
        self.reference_rows.iloc[::-1].to_parquet(self.reference, index=False)
        with self.assertRaisesRegex(ValueError, "ordered subset"):
            load_input(
                str(self.parquet), str(self.lenses), 5, 2,
                str(self.reference), str(self.fit),
            )

    def test_unpaired_protocol_retains_original_schema(self):
        original = protocol(str(self.parquet), str(self.lenses))
        paired = protocol(str(self.parquet), str(self.lenses),
                          str(self.reference), str(self.fit))
        self.assertEqual(original["scaling"], "standard")
        self.assertNotIn("reference_index", original)
        self.assertIn("reference_index", paired)
        self.assertIn("reference_fit", paired)


if __name__ == "__main__":
    unittest.main()
