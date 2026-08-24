# mojo-hnswlib

`mojo-hnswlib` is a standalone HNSW approximate-nearest-neighbour index with
its SIMD distance and heuristic neighbour-selection kernels written in
[Mojo](https://www.modular.com/mojo). Its Python `Index` class follows the
names, signatures, return shapes, dtypes, and distance conventions of
[`hnswlib`](https://github.com/nmslib/hnswlib) 0.8 for the covered subset.

This is a real multilayer HNSW graph, not a brute-force index. Insertion uses
exponentially distributed levels, greedy descent through upper layers,
`ef_construction` candidate search, the HNSW diversity heuristic, bounded
bidirectional links, and a best-first level-zero search controlled by `ef`.

## What is covered

| area | coverage |
| --- | --- |
| spaces | `l2` (squared Euclidean), `ip` (`1 - dot`), and `cosine` (`1 - cosine`) |
| lifecycle | `Index`, `init_index`, `add_items`, `resize_index`, `save_index`, `load_index` |
| query | `knn_query`, `set_ef`, callable label filters |
| mutation | existing-label updates, `mark_deleted`, `unmark_deleted`, deleted-slot replacement |
| inspection | `get_items`, `get_ids_list`, `get_current_count`, `get_max_elements`, upstream-style properties |
| controls | `set_num_threads` and per-call `num_threads` are API-compatible controls |

Query results and distance semantics are parity-tested against the real
upstream `hnswlib` package. Tests also compare both indexes with brute-force
NumPy ground truth, assert float32 distances and uint64 labels, and check the
port's graph degree, symmetry, deletion, filtering, update, resize, and
persistence behaviour.

Graph construction and traversal are single-threaded; `num_threads` is retained
and reported for call compatibility but does not create graph workers. Upstream
binary index files are not interchangeable with this repository's versioned
NumPy-based format. Upstream's pickle protocol and C++ API are not covered. The
saved format uses NumPy object serialization and must only be loaded from a
trusted source. Replacing or updating an item
rebuilds the graph to preserve correct connectivity, so it is correct but
intended for occasional mutation rather than streaming updates.

## Install

The repository pins its own Mojo nightly and installs upstream `hnswlib` for
parity tests and benchmarks:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-hnswlib.so`. Importing the Python module
also rebuilds the library when `src/hnsw.mojo` is newer.

## Usage

```python
import numpy as np
import mojo_hnswlib as hnswlib

rng = np.random.default_rng(7)
vectors = rng.normal(size=(10_000, 64)).astype(np.float32)
queries = rng.normal(size=(5, 64)).astype(np.float32)

index = hnswlib.Index(space="cosine", dim=64)
index.init_index(max_elements=len(vectors), M=16, ef_construction=200)
index.add_items(vectors, ids=np.arange(len(vectors)))
index.set_ef(80)

labels, distances = index.knn_query(queries, k=10)
print(labels.shape, distances.shape)
```

For an upstream-shaped import in application code, use
`import mojo_hnswlib as hnswlib`; the covered method calls then remain the same.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86_64. Both implementations use one thread, identical float32 data,
`M=16`, and the same construction/search parameters. Times are the best of
three runs for queries and one complete measured graph build.

| case | mojo-hnswlib | hnswlib 0.8 | upstream / Mojo | recall (Mojo / upstream) |
| --- | ---: | ---: | ---: | ---: |
| build 3,000 x 32 (`M=16`, `ef_construction=100`) | 1.84 s | 213.9 ms | 0.116x | n/a |
| query 500 x k=10 (`ef=20`) | 35.4 ms | 8.0 ms | 0.225x | 0.924 / 0.879 |
| query 500 x k=10 (`ef=100`) | 59.4 ms | 32.6 ms | 0.549x | 1.000 / 0.999 |

Upstream remains faster, but level-zero best-first traversal executes in one
Mojo call and leaves its candidates ordered in reusable scratch buffers.
Pruning similarly fuses distance calculation, ordering, and diversity selection.
Stable NumPy buffer addresses are cached across calls, upper-layer distance
scans reuse scratch storage, and the common query path reads results directly
from those buffers. This removes the dominant temporary allocations, copies,
validation passes, and FFI address conversions while preserving NumPy ownership.

There is intentionally no GPU path. L2 and dot-product distance have less than
two arithmetic operations per byte moved, while graph construction and
traversal add irregular link reads and branches. Bulk independent distance
scans stay serial below 4,096 vectors and use eight CPU workers above that
threshold; graph mutation and traversal remain single-threaded.

Run the benchmark only through the task, which takes a machine-wide lock:

```bash
pixi run bench
```

## How it works

Python owns all long-lived memory: a C-contiguous float32 vector matrix, uint64
external labels, random levels, ragged adjacency lists for each node and layer,
and a compact fixed-degree level-zero link mirror. Cosine vectors are normalized
once at insertion, matching upstream's stored-vector behaviour. Deleted nodes
remain traversable but are excluded from results.

`src/hnsw.mojo` is one compilation unit. Python calls it through `ctypes`;
buffers cross the C ABI as integer addresses and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside non-parametric
`@export(...) ... abi("C")` functions. Mojo performs distance scans with four
independent SIMD accumulators and a scalar remainder, complete level-zero heap
traversal, and the dependent HNSW diversity-selection loop. NumPy owns inputs,
outputs, reusable scratch buffers, and their lifetimes, so those buffers remain
zero-copy across the FFI boundary.

## License

MIT
