from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..artifacts import atomic_write_json, stable_digest


def write_registry(directory: str | Path, records: list[dict[str, object]]) -> None:
    destination = Path(directory)
    frame = pd.DataFrame(records)
    frame.to_csv(destination / "candidate_registry.csv", index=False)
    atomic_write_json(
        destination / "candidate_registry.json",
        {
            "schema_version": 1,
            "records": records,
            "records_sha256": stable_digest(records),
        },
    )
