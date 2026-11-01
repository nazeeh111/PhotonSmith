"""Incomplete or inconsistent benchmark evidence must never earn passing gates."""
import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('boundary_summary', ROOT/'scripts/summarize_boundary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
FROZEN = ROOT/'benchmarks/boundary/confirmation-v2'


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        for name in ('inputs.json','manifest.json','cases.json','raw.jsonl'):
            shutil.copyfile(FROZEN/name, self.folder/name)

    def read_cases(self):
        return json.loads((self.folder/'cases.json').read_text())

    def save_cases(self, cases):
        (self.folder/'cases.json').write_text(json.dumps(cases))

    def test_full_frozen_run_recomputes_published_result(self):
        result = summary.summarize(self.folder)
        published = json.loads((FROZEN/'summary.json').read_text())
        def compare(actual, expected):
            self.assertEqual(type(actual), type(expected))
            if isinstance(expected, dict):
                self.assertEqual(actual.keys(), expected.keys())
                for key in expected:
                    compare(actual[key], expected[key])
            elif isinstance(expected, list):
                self.assertEqual(len(actual), len(expected))
                for left, right in zip(actual, expected):
                    compare(left, right)
            elif isinstance(expected, float):
                # Transcendental reductions vary by a few ulps across libm/CPU.
                # Counts, identities, structure and pass/fail gates stay exact.
                self.assertTrue(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15),
                                (actual, expected))
            else:
                self.assertEqual(actual, expected)
        compare(result, published)

    def test_completed_prefix_cannot_stand_in_for_48_cases(self):
        cases = self.read_cases()[:8]
        ids = {c['case']['id'] for c in cases}
        raw = [json.loads(row) for row in (self.folder/'raw.jsonl').read_text().splitlines()]
        self.save_cases(cases)
        (self.folder/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw if r['case'] in ids))
        with self.assertRaisesRegex(ValueError, 'complete frozen input plan'):
            summary.summarize(self.folder)

    def test_missing_or_changed_frozen_inputs_are_rejected(self):
        path = self.folder/'inputs.json'
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'required'):
            summary.summarize(self.folder)
        path.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError, 'input hash'):
            summary.summarize(self.folder)

    def test_case_parameters_must_match_frozen_inputs(self):
        cases = self.read_cases()
        cases[0]['case']['wavelength'] += .01
        self.save_cases(cases)
        with self.assertRaisesRegex(ValueError, 'frozen input plan'):
            summary.summarize(self.folder)

    def test_missing_or_duplicate_raw_rows_are_rejected(self):
        path = self.folder/'raw.jsonl'
        rows = path.read_text().splitlines(keepends=True)
        for changed in (rows[:-1], rows[:-1]+rows[:1]):
            path.write_text(''.join(changed))
            with self.assertRaisesRegex(ValueError, 'raw observations'):
                summary.summarize(self.folder)

    def test_case_timings_cannot_disagree_with_raw_evidence(self):
        cases = self.read_cases()
        cases[0]['observations']['native'][0]['wall'] *= .5
        self.save_cases(cases)
        with self.assertRaisesRegex(ValueError, 'differ from raw'):
            summary.summarize(self.folder)

    def test_nonfinite_verification_cannot_pass(self):
        cases = self.read_cases()
        cases[0]['verification']['max_power_error'] = float('nan')
        self.save_cases(cases)
        with self.assertRaisesRegex(ValueError, 'Invalid numerical verification'):
            summary.summarize(self.folder)


if __name__ == '__main__':
    unittest.main()
