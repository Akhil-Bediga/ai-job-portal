"""ETL pipeline runner: raw files -> MySQL + MongoDB."""
from __future__ import annotations

import time

from src.etl.extract import extract
from src.etl.load import load
from src.etl.transform import transform


def run() -> None:
    started = time.perf_counter()

    print("[1/3] extract")
    raw = extract()
    print(f"      jobs={len(raw.jobs):,} resumes={len(raw.resumes):,}")

    print("[2/3] transform")
    td = transform(raw)

    print("[3/3] load")
    counts = load(td)

    elapsed = time.perf_counter() - started
    print("\ndone")
    for name, n in counts.items():
        print(f"    {name:<14} {n:>10,} rows")
    print(f"  elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    run()
