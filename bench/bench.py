"""Reproducible single-threaded benchmarks against upstream hnswlib 0.8."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import hnswlib as upstream  # noqa: E402
import mojo_hnswlib as mojo  # noqa: E402


def timeit(function, repeat: int = 3):
    best = math.inf
    value = None
    for _ in range(repeat):
        start = time.perf_counter()
        value = function()
        best = min(best, time.perf_counter() - start)
    return best, value


def make_index(cls, data, M=16, ef_construction=100):
    index = cls(space="l2", dim=data.shape[1])
    index.init_index(
        max_elements=len(data),
        M=M,
        ef_construction=ef_construction,
        random_seed=17,
    )
    index.set_num_threads(1)
    index.add_items(data, num_threads=1)
    return index


def recall_at_10(found, truth):
    return np.mean(
        [len(set(map(int, actual)) & set(map(int, wanted))) / 10
         for actual, wanted in zip(found, truth)]
    )


def milliseconds(seconds):
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    rng = np.random.default_rng(123)
    data = rng.normal(size=(3000, 32)).astype(np.float32)
    queries = rng.normal(size=(500, 32)).astype(np.float32)

    # Load and JIT-independent shared-library initialization before timing.
    warm = mojo.Index("l2", 2)
    warm.init_index(2)
    warm.add_items([[0, 0], [1, 1]])
    warm.knn_query([[0, 0]])

    mojo_build, mojo_index = timeit(lambda: make_index(mojo.Index, data), repeat=1)
    upstream_build, upstream_index = timeit(
        lambda: make_index(upstream.Index, data), repeat=1
    )

    exact = np.sum((queries[:, None] - data[None]) ** 2, axis=2)
    truth = np.argsort(exact, axis=1)[:, :10]
    rows = [
        (
            "build 3,000 x 32 (M=16, ef_construction=100)",
            mojo_build,
            upstream_build,
            None,
            None,
        )
    ]
    for ef in (20, 100):
        mojo_index.set_ef(ef)
        upstream_index.set_ef(ef)
        mojo_time, mojo_result = timeit(
            lambda: mojo_index.knn_query(queries, k=10, num_threads=1), repeat=3
        )
        upstream_time, upstream_result = timeit(
            lambda: upstream_index.knn_query(queries, k=10, num_threads=1), repeat=3
        )
        rows.append(
            (
                f"query 500 x k=10 (ef={ef})",
                mojo_time,
                upstream_time,
                recall_at_10(mojo_result[0], truth),
                recall_at_10(upstream_result[0], truth),
            )
        )

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| case | mojo-hnswlib | hnswlib 0.8 | upstream / Mojo | recall (Mojo / upstream) |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, ours, theirs, our_recall, their_recall in rows:
        recall = (
            "n/a"
            if our_recall is None
            else f"{our_recall:.3f} / {their_recall:.3f}"
        )
        print(
            f"| {name} | {milliseconds(ours)} | {milliseconds(theirs)} | "
            f"{theirs / ours:.3f}x | {recall} |"
        )


if __name__ == "__main__":
    main()
