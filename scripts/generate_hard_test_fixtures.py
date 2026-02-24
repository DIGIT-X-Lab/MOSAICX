#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "tests" / "datasets" / "generated_hardcases"
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_path = out_dir / "cohort_edge_cases.csv"
    rows = [
        {
            "Subject": "KV101",
            "Sex": "F",
            "Ethnicity": "Japanese",
            "BMI": "22.4",
            "ScanDate": "2025-08-01",
            "Notes": "baseline",
        },
        {
            "Subject": "KV102",
            "Sex": "M",
            "Ethnicity": "Japanese",
            "BMI": "25.1",
            "ScanDate": "2025-08-01",
            "Notes": "baseline",
        },
        {
            "Subject": "KV103",
            "Sex": "F",
            "Ethnicity": "German",
            "BMI": "29.3",
            "ScanDate": "2025-09-10",
            "Notes": "follow-up",
        },
        {
            "Subject": "KV104",
            "Sex": "M",
            "Ethnicity": "German",
            "BMI": "31.0",
            "ScanDate": "2025-09-10",
            "Notes": "follow-up",
        },
        {
            "Subject": "KV105",
            "Sex": "Other",
            "Ethnicity": "Unknown",
            "BMI": "",
            "ScanDate": "2025-09-10",
            "Notes": "missing BMI",
        },
    ]

    with cohort_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {cohort_path}")


if __name__ == "__main__":
    main()
