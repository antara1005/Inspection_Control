"""Append-only log of autofocus peak-hold sweeps.

Backs the S1.5 depth-noise experiment of the Hybrid Autofocus test plan: every
record/drive cycle appends one summary row, so the run-to-run spread of the
focal distance measured *at peak focus* can be compared against the depth of
field. Because the lens's true focus distance is fixed, that spread is an
end-to-end estimate of the depth-map error.

Layout under ``log_dir``::

    sweeps.jsonl                     one JSON object per record cycle
    raw/sweep_00007_20260825T2114Z.csv   optional per-sweep samples

JSONL (not CSV) for the summary: rows are appended live by a running node
across many sessions, and JSONL tolerates schema growth without rewriting a
header or silently misaligning columns. Read it back with
``pandas.read_json(path, lines=True)``.

The raw sidecar keeps the full (focal_distance, focus_value) sweep so the peak
can be re-fit offline and the plateau-vs-traversed-peak question (design doc
§4.4 B4) answered without re-running the robot.

Nothing here raises: a logging failure must never take down the control node.
"""

import csv
import json
import os
from datetime import datetime, timezone


class SweepLogger:
    """Append summary rows (and optional raw sweeps) to a directory on disk.

    Self-disables after a directory-creation failure so a bad path produces one
    warning rather than an error per record cycle.
    """

    def __init__(self, log_dir, summary_name="sweeps.jsonl", keep_raw=True):
        self.log_dir = log_dir
        self.summary_path = os.path.join(log_dir, summary_name)
        self.raw_dir = os.path.join(log_dir, "raw")
        self.keep_raw = keep_raw
        self.enabled = True
        self.disabled_reason = None
        # Continue the run index across sessions so rows stay globally ordered.
        self._next_index = self._count_existing()

    def _count_existing(self):
        try:
            with open(self.summary_path, "r") as fh:
                return sum(1 for line in fh if line.strip())
        except OSError:
            return 0

    def append(self, record, raw_sweep=None):
        """Append one summary row; returns (summary_path, error_message).

        `record` is a JSON-serialisable dict; `index`, `wall_time` and
        `sweep_file` are filled in here. `raw_sweep` is an iterable of
        (focal_distance_m, focus_value) pairs, written to a CSV sidecar when
        `keep_raw` is set.
        """
        if not self.enabled:
            return None, self.disabled_reason
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except OSError as e:
            self.enabled = False
            self.disabled_reason = f"cannot create {self.log_dir}: {e}"
            return None, self.disabled_reason

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        row = dict(record)
        row["index"] = self._next_index
        row["wall_time"] = datetime.now(timezone.utc).isoformat()
        row["sweep_file"] = None

        if self.keep_raw and raw_sweep:
            name = f"sweep_{self._next_index:05d}_{stamp}.csv"
            try:
                os.makedirs(self.raw_dir, exist_ok=True)
                with open(os.path.join(self.raw_dir, name), "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["focal_distance_m", "focus_value"])
                    w.writerows(raw_sweep)
                row["sweep_file"] = os.path.join("raw", name)
            except OSError as e:
                # Losing the sidecar must not lose the summary row.
                row["sweep_file"] = None
                row["raw_error"] = str(e)

        try:
            with open(self.summary_path, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except (OSError, TypeError) as e:
            return None, f"append to {self.summary_path} failed: {e}"

        self._next_index += 1
        return self.summary_path, None
