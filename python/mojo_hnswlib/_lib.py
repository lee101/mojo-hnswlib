"""ctypes bridge to the Mojo SIMD distance kernels."""

from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "hnsw.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-hnswlib.so")

I = ctypes.c_int64
F32 = ctypes.c_float


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_HNSWLIB_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_HNSWLIB_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = _mojo_command() + ["build", "--emit", "shared-lib", SRC, "-o", LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None
_RUNTIME: ctypes.CDLL | None = None
_CPU_DEVICE: int | None = None
_PARALLEL_DISTANCE_THRESHOLD = 4096


def _release_parallel_runtime() -> None:
    global _CPU_DEVICE
    if _RUNTIME is not None and _CPU_DEVICE is not None:
        _RUNTIME.KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice(_CPU_DEVICE)
        _CPU_DEVICE = None


def _ensure_parallel_runtime() -> None:
    global _RUNTIME, _CPU_DEVICE
    if _CPU_DEVICE is not None:
        return
    _RUNTIME = ctypes.CDLL("libKGENCompilerRTShared.so", mode=ctypes.RTLD_GLOBAL)
    _RUNTIME.KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice.argtypes = []
    _RUNTIME.KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice.restype = ctypes.c_void_p
    _RUNTIME.KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice.argtypes = [ctypes.c_void_p]
    _RUNTIME.KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice.restype = None
    _CPU_DEVICE = _RUNTIME.KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice()


atexit.register(_release_parallel_runtime)


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build(), mode=ctypes.RTLD_GLOBAL)
        _LIBRARY.mh_distance_indexed.argtypes = [I, I, I, I, I, I, I]
        _LIBRARY.mh_distance_indexed.restype = None
        _LIBRARY.mh_distance_all.argtypes = [I, I, I, I, I, I]
        _LIBRARY.mh_distance_all.restype = None
        _LIBRARY.mh_pair_distance.argtypes = [I, I, I, I]
        _LIBRARY.mh_pair_distance.restype = F32
        _LIBRARY.mh_search_layer_zero.argtypes = [I] * 15
        _LIBRARY.mh_search_layer_zero.restype = I
        _LIBRARY.mh_select_neighbors.argtypes = [I, I, I, I, I, I, I, I]
        _LIBRARY.mh_select_neighbors.restype = I
        _LIBRARY.mh_prune_neighbors.argtypes = [I] * 9
        _LIBRARY.mh_prune_neighbors.restype = I
    return _LIBRARY


def addr(array: np.ndarray) -> int:
    if array.size == 0:
        raise ValueError("cannot take an FFI address for an empty array")
    return ctypes.addressof(ctypes.c_char.from_buffer(array))


def _array(
    name: str,
    value: np.ndarray,
    dtype: np.dtype,
    ndim: int,
    *,
    writable: bool = True,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype).name}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _space(space: int) -> int:
    if not isinstance(space, (int, np.integer)) or int(space) not in (0, 1, 2):
        raise ValueError("space must be 0 (l2), 1 (ip), or 2 (cosine)")
    return int(space)


def _vectors(data: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = _array("data", data, np.dtype(np.float32), 2)
    query = _array("query", query, np.dtype(np.float32), 1)
    if data.shape[1] <= 0 or query.shape[0] != data.shape[1]:
        raise ValueError("query length must equal the positive data dimension")
    return data, query


def distance_indexed(
    data: np.ndarray, query: np.ndarray, indices: np.ndarray, result: np.ndarray, space: int
) -> None:
    data, query = _vectors(data, query)
    indices = _array("indices", indices, np.dtype(np.int64), 1)
    result = _array("result", result, np.dtype(np.float32), 1)
    space = _space(space)
    if result.shape[0] != indices.shape[0]:
        raise ValueError("result length must equal indices length")
    if not len(indices):
        return
    if np.any(indices < 0) or np.any(indices >= data.shape[0]):
        raise IndexError("indices contain a row outside data")
    if len(indices) >= _PARALLEL_DISTANCE_THRESHOLD:
        _ensure_parallel_runtime()
    lib().mh_distance_indexed(
        addr(data), addr(query), addr(indices), addr(result), len(indices), data.shape[1], space
    )


def distance_all(
    data: np.ndarray, query: np.ndarray, result: np.ndarray, count: int, space: int
) -> None:
    data, query = _vectors(data, query)
    result = _array("result", result, np.dtype(np.float32), 1)
    space = _space(space)
    if not isinstance(count, (int, np.integer)) or not 0 <= int(count) <= data.shape[0]:
        raise ValueError("count must be between zero and the number of data rows")
    count = int(count)
    if result.shape[0] < count:
        raise ValueError("result is shorter than count")
    if count == 0:
        return
    if count >= _PARALLEL_DISTANCE_THRESHOLD:
        _ensure_parallel_runtime()
    lib().mh_distance_all(addr(data), addr(query), addr(result), count, data.shape[1], space)


def pair_distance(a: np.ndarray, b: np.ndarray, space: int) -> float:
    a = _array("a", a, np.dtype(np.float32), 1)
    b = _array("b", b, np.dtype(np.float32), 1)
    space = _space(space)
    if len(a) == 0 or len(a) != len(b):
        raise ValueError("a and b must have the same positive length")
    return float(lib().mh_pair_distance(addr(a), addr(b), len(a), space))


def search_layer_zero(
    data: np.ndarray,
    query: np.ndarray,
    links: np.ndarray,
    link_counts: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_distances: np.ndarray,
    best_ids: np.ndarray,
    best_distances: np.ndarray,
    visited: np.ndarray,
    entry: int,
    visit_token: int,
    ef: int,
    space: int,
    *,
    _trusted_graph: bool = False,
) -> int:
    data, query = _vectors(data, query)
    links = _array("links", links, np.dtype(np.int64), 2)
    link_counts = _array("link_counts", link_counts, np.dtype(np.int64), 1)
    candidate_ids = _array("candidate_ids", candidate_ids, np.dtype(np.int64), 1)
    candidate_distances = _array(
        "candidate_distances", candidate_distances, np.dtype(np.float32), 1
    )
    best_ids = _array("best_ids", best_ids, np.dtype(np.int64), 1)
    best_distances = _array(
        "best_distances", best_distances, np.dtype(np.float32), 1
    )
    visited = _array("visited", visited, np.dtype(np.int64), 1)
    space = _space(space)
    rows, degree = links.shape
    if rows == 0 or degree == 0 or data.shape[0] < rows:
        raise ValueError("links must have non-zero shape covered by data")
    if len(link_counts) != rows or len(visited) != rows:
        raise ValueError("link_counts and visited must match links rows")
    if not _trusted_graph:
        if np.any(link_counts < 0) or np.any(link_counts > degree):
            raise ValueError("link_counts must be between zero and links width")
        used = np.arange(degree)[None, :] < link_counts[:, None]
        if np.any(used & ((links < 0) | (links >= rows))):
            raise IndexError("links contain a node outside links rows")
    if not isinstance(entry, (int, np.integer)) or not 0 <= int(entry) < rows:
        raise IndexError("entry is outside links rows")
    if not isinstance(visit_token, (int, np.integer)) or not 0 < int(
        visit_token
    ) <= np.iinfo(np.int64).max:
        raise ValueError("visit_token must be a positive int64")
    if not isinstance(ef, (int, np.integer)) or int(ef) <= 0:
        raise ValueError("ef must be positive")
    if min(
        len(candidate_ids),
        len(candidate_distances),
        len(best_ids),
        len(best_distances),
    ) < rows:
        raise ValueError("search scratch buffers must hold one value per links row")
    return int(
        lib().mh_search_layer_zero(
            addr(data),
            addr(query),
            addr(links),
            addr(link_counts),
            addr(candidate_ids),
            addr(candidate_distances),
            addr(best_ids),
            addr(best_distances),
            addr(visited),
            entry,
            visit_token,
            ef,
            links.shape[1],
            data.shape[1],
            space,
        )
    )


def select_neighbors(
    data: np.ndarray,
    candidates: np.ndarray,
    distances: np.ndarray,
    selected: np.ndarray,
    limit: int,
    space: int,
) -> int:
    data = _array("data", data, np.dtype(np.float32), 2)
    candidates = _array("candidates", candidates, np.dtype(np.int64), 1)
    distances = _array("distances", distances, np.dtype(np.float32), 1)
    selected = _array("selected", selected, np.dtype(np.int64), 1)
    space = _space(space)
    if data.shape[1] <= 0:
        raise ValueError("data dimension must be positive")
    if len(distances) != len(candidates):
        raise ValueError("distances length must equal candidates length")
    if np.any(candidates < 0) or np.any(candidates >= data.shape[0]):
        raise IndexError("candidates contain a row outside data")
    if not isinstance(limit, (int, np.integer)) or not 0 <= int(limit) <= min(
        len(candidates), len(selected)
    ):
        raise ValueError("limit exceeds candidates or selected capacity")
    if len(candidates) == 0 or int(limit) == 0:
        return 0
    return int(
        lib().mh_select_neighbors(
            addr(data),
            addr(candidates),
            addr(distances),
            addr(selected),
            len(candidates),
            limit,
            data.shape[1],
            space,
        )
    )


def prune_neighbors(
    data: np.ndarray,
    candidates: np.ndarray,
    distances: np.ndarray,
    selected: np.ndarray,
    count: int,
    limit: int,
    query_position: int,
    space: int,
) -> int:
    data = _array("data", data, np.dtype(np.float32), 2)
    candidates = _array("candidates", candidates, np.dtype(np.int64), 1)
    distances = _array("distances", distances, np.dtype(np.float32), 1)
    selected = _array("selected", selected, np.dtype(np.int64), 1)
    space = _space(space)
    if data.shape[1] <= 0:
        raise ValueError("data dimension must be positive")
    if not isinstance(count, (int, np.integer)) or not 0 <= int(count) <= min(
        len(candidates), len(distances)
    ):
        raise ValueError("count exceeds candidates or distances capacity")
    count = int(count)
    if not isinstance(limit, (int, np.integer)) or not 0 <= int(limit) <= min(
        count, len(selected)
    ):
        raise ValueError("limit exceeds count or selected capacity")
    if not isinstance(query_position, (int, np.integer)) or not 0 <= int(
        query_position
    ) < data.shape[0]:
        raise IndexError("query_position is outside data")
    if count and (
        np.any(candidates[:count] < 0)
        or np.any(candidates[:count] >= data.shape[0])
    ):
        raise IndexError("candidates contain a row outside data")
    if count == 0 or int(limit) == 0:
        return 0
    return int(
        lib().mh_prune_neighbors(
            addr(data),
            addr(candidates),
            addr(distances),
            addr(selected),
            count,
            limit,
            query_position,
            data.shape[1],
            space,
        )
    )
