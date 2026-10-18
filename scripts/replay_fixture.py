"""Generate a small timestamped pose-source fixture, not camera validation evidence."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "output/replay"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(151):
        t = i * 0.02 + (0.1 if i >= 25 else 0)
        captured = t - 0.08 if i == 25 else (0.52 if i == 26 else t)
        pose = (
            None
            if 50 <= i < 80
            else {"position_m": [0, 0, 2], "quaternion_wxyz": [1, 0, 0, 0]}
        )
        rows.append({"captured_s": captured, "delivered_s": t, "pose": pose})
    (root / "faults.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
