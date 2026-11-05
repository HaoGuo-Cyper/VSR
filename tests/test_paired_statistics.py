import unittest
from vsr.paired_statistics import paired_summary


def rows(values, seeds=(42, 43)):
    return [{"record_id": record, "source_id": source, "seed": seed, "metrics": {"quality": value + offset}}
            for record, source, value, offset in values for seed in seeds]


class PairedStatisticsTest(unittest.TestCase):
    def test_seed_repeats_not_video_units_and_sources_not_clips(self):
        base = rows([("a", "upload1", 0, 0), ("b", "upload1", 0, 0), ("c", "upload2", 0, 0)])
        candidate = rows([("a", "upload1", 0, 3), ("b", "upload1", 0, 3), ("c", "upload2", 0, 0)])
        result = paired_summary({"Base": base, "Candidate": candidate}, "Base", replicates=5000)
        self.assertEqual(result["video_count"], 3)
        self.assertEqual(result["source_count"], 2)
        delta = result["comparisons"]["Candidate"]["quality"]
        self.assertEqual(delta["delta_video_macro"], 2)
        self.assertEqual(delta["delta_source_macro"], 1.5)
        self.assertEqual(delta["ci95_source_cluster_video_macro"], [0, 3])
        self.assertFalse(result["protocol"]["training_repeat_uncertainty_included"])

    def test_common_seed_effect_cancels_pairing_and_order_invariant(self):
        base = [{"record_id": r, "source_id": r, "seed": s, "metrics": {"quality": v}}
                for r in ("a", "b", "c") for s, v in ((42, -100), (43, 100))]
        candidate = [{**r, "metrics": {"quality": r["metrics"]["quality"] + 1}} for r in base]
        first = paired_summary({"Base": base, "Candidate": candidate}, "Base", replicates=500)
        second = paired_summary({"Candidate": candidate[::-1], "Base": base[::-1]}, "Base", replicates=500)
        self.assertEqual(first, second)
        self.assertEqual(first["comparisons"]["Candidate"]["quality"]["ci95_source_cluster_video_macro"], [1, 1])

    def test_reject_incomplete_seeds_duplicate_units_and_unpaired_sets(self):
        good = rows([("a", "source", 0, 0), ("b", "source2", 0, 0)])
        for bad in (good[:-1], good + [good[0]], [{**r, "source_id": "other"} for r in good],
                    [{**r, "metrics": {"quality": float("nan")}} for r in good]):
            with self.assertRaises(ValueError):
                paired_summary({"Base": good, "Candidate": bad}, "Base", replicates=100)

    def test_single_source_reports_no_spurious_confidence_interval(self):
        base = rows([("a", "source", 0, 0)])
        candidate = rows([("a", "source", 0, 2)])
        result = paired_summary({"Base": base, "Candidate": candidate}, "Base", replicates=100)
        self.assertIsNone(result["comparisons"]["Candidate"]["quality"]["ci95_source_macro"])


if __name__ == "__main__":
    unittest.main()
