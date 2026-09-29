"""Synthetic tests only. No competition data, models or platform predictions."""
import csv
import tempfile
import unittest
from pathlib import Path
from bf_tap_r2.ensemble_gain_audit import REQUIRED, audit, decompose


class AuditTests(unittest.TestCase):
    def test_error_cancellation(self):
        out = decompose([100.], [98.], [[98.], [102.], [100.]])
        self.assertAlmostEqual(out['net_isolated_score_gain'], 1.)
        self.assertGreater(out['absolute_error_cancellation_gain'], 0)
        self.assertLess(out['decomposition_max_error'], 1e-12)

    def test_averaging_can_lose_against_current(self):
        out = decompose([100.], [100.], [[100.], [104.], [106.]])
        self.assertLess(out['net_isolated_score_gain'], 0.)
        self.assertAlmostEqual(out['absolute_error_cancellation_gain'], 0.)

    def test_identical_members_zero_gain(self):
        out = decompose([100., 200.], [99., 205.], [[99., 205.]] * 3)
        self.assertAlmostEqual(out['net_isolated_score_gain'], 0.)
        self.assertAlmostEqual(out['absolute_error_cancellation_gain'], 0.)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            decompose([100.], [99.], [[float('nan')]])

    def test_zero_denominator(self):
        with self.assertRaises(ValueError):
            decompose([0.], [1.], [[1.]])

    def rows(self):
        return [dict(zip(REQUIRED, ['tap_iron', seed, str(fold), f'id{fold}',
                                    str(100+fold), str(98+fold), str(98+fold),
                                    str(102+fold), str(100+fold)]))
                for seed in ['42', '3407'] for fold in range(5)]

    def run_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.csv'
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=REQUIRED)
                writer.writeheader()
                writer.writerows(rows)
            return audit(path)

    def test_complete_coverage(self):
        out = self.run_rows(self.rows())
        self.assertEqual(out['targets']['tap_iron']['positive_split_count'], 2)
        self.assertFalse(out['provenance_verified'])

    def test_row_order_invariant(self):
        self.assertEqual(self.run_rows(self.rows()), self.run_rows(list(reversed(self.rows()))))

    def test_duplicate_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            self.run_rows(rows + [rows[0]])

    def test_missing_fold_rejected(self):
        with self.assertRaises(ValueError):
            self.run_rows(self.rows()[:-1])

    def test_label_mismatch_rejected(self):
        rows = self.rows()
        rows[-1]['actual'] = '200'
        with self.assertRaises(ValueError):
            self.run_rows(rows)

    def test_native_replay_required(self):
        rows = self.rows()
        rows[-1]['seed_42_prediction'] = '200'
        with self.assertRaises(ValueError):
            self.run_rows(rows)

    def test_missing_value_rejected(self):
        rows = self.rows()
        rows[-1]['actual'] = ''
        with self.assertRaises(ValueError):
            self.run_rows(rows)


if __name__ == '__main__':
    unittest.main()
