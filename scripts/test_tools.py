"""Checks for scenario input validation and evidence parsing; no network required."""

import unittest

from embedded_evidence import cfi_frames
from serve import scenario_args


class ToolsTests(unittest.TestCase):
    def test_scenario_rejects_nonfinite_unknown_and_fractional_counts(self):
        for value in (
            {"noise": float("nan")},
            {"out": "/tmp"},
            {"false_detections": 0.5},
            {"noise": True},
            {"delay_ms": -1},
        ):
            with self.assertRaises((ValueError, TypeError)):
                scenario_args(value)
        self.assertEqual(
            scenario_args({"noise": 2, "false_detections": 3}),
            ["--noise", "2", "--false-detections", "3", "--frames", "1000"],
        )
        self.assertIn(
            "1250", scenario_args({"blackout_start": 20, "blackout_duration": 3})
        )

    def test_cfi_parser_does_not_claim_dynamic_frames(self):
        source = "func:\n.cfi_startproc\n.cfi_def_cfa_offset 104\n.cfi_endproc\nother:\n.cfi_startproc\n.cfi_def_cfa_register %rbp\n.cfi_endproc"
        rows = cfi_frames(source)
        self.assertEqual(rows[0]["static_frame_bytes"], 96)
        self.assertIsNone(rows[1]["static_frame_bytes"])


if __name__ == "__main__":
    unittest.main()
