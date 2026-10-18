"""Faults in recorded evidence must fail the suite's integration contract."""

import copy
import unittest

from scenario_suite import audit, percentile


def recording():
    return {
        "config": {"seed": 42},
        "summary": {"accepted": 1, "valid_fraction": 1},
        "frames": [
            {
                "t": 0,
                "state": "tracking",
                "valid": True,
                "age": 0,
                "available": True,
                "used": True,
                "reason": "none",
                "p": [0, 0, 1],
                "v": [0, 0, 0],
                "w": [0, 0, 0],
                "q": [1, 0, 0, 0],
                "sigma": [0.01] * 12,
                "error": 0.01,
                "angle": 1,
            }
        ],
    }


class ScenarioSuiteTests(unittest.TestCase):
    def test_contract_rejects_age_covariance_and_numerical_corruption(self):
        self.assertTrue(audit(recording(), "integrity")["contract_passed"])
        for key, value, expected in (
            ("age", 0.6, "invalid_health_contract"),
            ("q", [2, 0, 0, 0], "quaternion_norm"),
            ("sigma", [0.6] * 12, "unbounded_valid_sigma"),
            ("p", [float("nan"), 0, 1], "nonfinite_or_incomplete_state"),
            ("available", False, "update_during_blackout"),
        ):
            data = copy.deepcopy(recording())
            data["frames"][0][key] = value
            result = audit(data, "integrity")
            self.assertFalse(result["contract_passed"])
            self.assertIn(expected, result["violations"])

    def test_missing_estimates_are_not_zero_error_successes(self):
        data = recording()
        f = data["frames"][0]
        f.update(valid=False, used=False, state="acquiring", reason="stale_measurement")
        for key in ("p", "v", "w", "q", "sigma", "error", "angle"):
            f[key] = None
        data["summary"].update(accepted=0, valid_fraction=0)
        result = audit(data, "reject_stale")
        self.assertTrue(result["contract_passed"])
        self.assertEqual(result["estimate_coverage"], 0)
        self.assertIsNone(result["position_p95_m"])
        self.assertIsNone(result["valid_position_worst_m"])
        self.assertIsNone(percentile([], 0.95))

    def test_unknown_contract_and_duplicate_time_fail(self):
        data = recording()
        with self.assertRaises(ValueError):
            audit(data, "flight_ready")
        data["frames"].append(copy.deepcopy(data["frames"][0]))
        self.assertIn("nonmonotonic_time", audit(data, "integrity")["violations"])

    def test_integrity_pass_does_not_hide_accuracy_failure(self):
        data = recording()
        data["summary"].update(
            position_rms_m=0.34,
            worst_position_m=1.1,
            attitude_rms_deg=3,
            recovery_s=0.5,
        )
        result = audit(data, "integrity")
        self.assertTrue(result["contract_passed"])
        self.assertFalse(result["tracking_guard_passed"])
        self.assertIn("worst_position_m", result["tracking_guard_failures"])


if __name__ == "__main__":
    unittest.main()
