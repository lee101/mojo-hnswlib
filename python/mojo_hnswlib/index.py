"""A caller-owned HNSW graph whose numerical distance loop runs in Mojo."""

from __future__ import annotations

import heapq
import os
from collections.abc import Callable, Iterable

import numpy as np

from ._lib import (
    distance_all,
    distance_indexed,
    pair_distance,
    prune_neighbors,
    search_layer_zero,
    select_neighbors,
)

_SPACE_CODES = {"l2": 0, "ip": 1, "cosine": 2}


class Index:
    """Approximate nearest-neighbour index compatible with hnswlib's Python API."""

    def __init__(self, space: str, dim: int):
        if space not in _SPACE_CODES:
            raise RuntimeError("Space name must be one of l2, ip, or cosine")
        if not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.space = space
        self.dim = int(dim)
        self._space_code = _SPACE_CODES[space]
        self._initialized = False
        self._data = np.empty((0, self.dim), dtype=np.float32)
        self._labels = np.empty(0, dtype=np.uint64)
        self._levels = np.empty(0, dtype=np.int16)
        self._links: list[list[list[int]]] = []
        self._label_to_pos: dict[int, int] = {}
        self._deleted: set[int] = set()
        self._entrypoint = -1
        self._max_level = -1
        self.M = 0
        self.ef_construction = 0
        self.ef = 10
        self.num_threads = os.cpu_count() or 1
        self.max_elements = 0
        self._random_seed = 100
        self._rng = np.random.default_rng(self._random_seed)
        self._allow_replace_deleted = False
        self._graph_capacity = 0
        self._link0 = np.empty((0, 0), dtype=np.int64)
        self._link0_counts = np.empty(0, dtype=np.int64)
        self._candidate_ids = np.empty(0, dtype=np.int64)
        self._candidate_distances = np.empty(0, dtype=np.float32)
        self._best_ids = np.empty(0, dtype=np.int64)
        self._best_distances = np.empty(0, dtype=np.float32)
        self._visited = np.empty(0, dtype=np.int64)
        self._visit_token = 0

    @property
    def element_count(self) -> int:
        return len(self._labels)

    def get_current_count(self) -> int:
        return self.element_count

    def get_max_elements(self) -> int:
        return self.max_elements

    def init_index(
        self,
        max_elements: int,
        M: int = 16,
        ef_construction: int = 200,
        random_seed: int = 100,
        allow_replace_deleted: bool = False,
    ) -> None:
        if not isinstance(max_elements, (int, np.integer)):
            raise TypeError("max_elements must be an integer")
        if not isinstance(M, (int, np.integer)):
            raise TypeError("M must be an integer")
        if not isinstance(ef_construction, (int, np.integer)):
            raise TypeError("ef_construction must be an integer")
        if not isinstance(random_seed, (int, np.integer)):
            raise TypeError("random_seed must be an integer")
        if max_elements <= 0:
            raise ValueError("max_elements must be positive")
        if M <= 1:
            raise ValueError("M must be greater than one")
        if ef_construction <= 0:
            raise ValueError("ef_construction must be positive")
        self.max_elements = int(max_elements)
        self.M = int(M)
        self.ef_construction = max(int(ef_construction), self.M)
        self._random_seed = int(random_seed)
        self._rng = np.random.default_rng(self._random_seed)
        self._allow_replace_deleted = bool(allow_replace_deleted)
        self._data = np.empty((self.max_elements, self.dim), dtype=np.float32)
        self._labels = np.empty(0, dtype=np.uint64)
        self._levels = np.empty(0, dtype=np.int16)
        self._links = []
        self._label_to_pos = {}
        self._deleted = set()
        self._entrypoint = -1
        self._max_level = -1
        self._graph_capacity = 0
        self._link0 = np.empty((0, 2 * self.M), dtype=np.int64)
        self._link0_counts = np.empty(0, dtype=np.int64)
        self._candidate_ids = np.empty(0, dtype=np.int64)
        self._candidate_distances = np.empty(0, dtype=np.float32)
        self._best_ids = np.empty(0, dtype=np.int64)
        self._best_distances = np.empty(0, dtype=np.float32)
        self._visited = np.empty(0, dtype=np.int64)
        self._visit_token = 0
        self._ensure_graph_capacity(min(self.max_elements, 1024))
        self._initialized = True

    def set_ef(self, ef: int) -> None:
        if not isinstance(ef, (int, np.integer)):
            raise TypeError("ef must be an integer")
        if ef <= 0:
            raise ValueError("ef must be positive")
        self.ef = int(ef)

    def set_num_threads(self, num_threads: int) -> None:
        if not isinstance(num_threads, (int, np.integer)):
            raise TypeError("num_threads must be an integer")
        self.num_threads = int(num_threads)

    def resize_index(self, new_size: int) -> None:
        self._require_initialized()
        if not isinstance(new_size, (int, np.integer)):
            raise TypeError("new_size must be an integer")
        if new_size <= 0:
            raise ValueError("new_size must be positive")
        if new_size < self.element_count:
            raise RuntimeError("Cannot resize below the current element count")
        if new_size == self.max_elements:
            return
        resized = np.empty((int(new_size), self.dim), dtype=np.float32)
        resized[: self.element_count] = self._data[: self.element_count]
        self._data = resized
        self.max_elements = int(new_size)

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("The index is not initialized; call init_index first")

    def _ensure_graph_capacity(self, required: int) -> None:
        if required <= self._graph_capacity:
            return
        capacity = max(required, min(self.max_elements, max(1, 2 * self._graph_capacity)))
        link0 = np.empty((capacity, 2 * self.M), dtype=np.int64)
        link0_counts = np.zeros(capacity, dtype=np.int64)
        visited = np.zeros(capacity, dtype=np.int64)
        if self._graph_capacity:
            link0[: self._graph_capacity] = self._link0
            link0_counts[: self._graph_capacity] = self._link0_counts
            visited[: self._graph_capacity] = self._visited
        self._link0 = link0
        self._link0_counts = link0_counts
        self._candidate_ids = np.empty(capacity, dtype=np.int64)
        self._candidate_distances = np.empty(capacity, dtype=np.float32)
        self._best_ids = np.empty(capacity, dtype=np.int64)
        self._best_distances = np.empty(capacity, dtype=np.float32)
        self._visited = visited
        self._graph_capacity = capacity

    def _sync_link0(self, position: int) -> None:
        neighbors = self._links[position][0]
        count = len(neighbors)
        if count:
            self._link0[position, :count] = neighbors
        self._link0_counts[position] = count

    def _prepare_vectors(self, values) -> np.ndarray:
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise RuntimeError(f"Wrong dimensionality: expected {self.dim}")
        vectors = np.ascontiguousarray(vectors)
        if self.space == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = np.divide(
                vectors,
                norms,
                out=np.zeros_like(vectors),
                where=norms != 0,
            )
        return vectors

    def _random_level(self) -> int:
        return min(32, int(-np.log(max(float(self._rng.random()), 1e-12)) / np.log(self.M)))

    def _distances(self, query: np.ndarray, positions: Iterable[int]) -> np.ndarray:
        indices = np.ascontiguousarray(list(positions), dtype=np.int64)
        result = np.empty(len(indices), dtype=np.float32)
        if len(indices):
            distance_indexed(self._data, query, indices, result, self._space_code)
        return result

    def _distance_one(self, query: np.ndarray, position: int) -> float:
        return pair_distance(query, self._data[position], self._space_code)

    def _greedy(self, query: np.ndarray, entry: int, level: int) -> tuple[int, float]:
        current = entry
        current_distance = self._distance_one(query, current)
        changed = True
        while changed:
            changed = False
            neighbors = self._links[current][level] if level < len(self._links[current]) else []
            if not neighbors:
                break
            distances = self._distances(query, neighbors)
            best_offset = int(np.argmin(distances))
            best_distance = float(distances[best_offset])
            if best_distance < current_distance:
                current = neighbors[best_offset]
                current_distance = best_distance
                changed = True
        return current, current_distance

    def _search_layer(
        self, query: np.ndarray, entries: Iterable[int], ef: int, level: int
    ) -> list[tuple[float, int]]:
        visited: set[int] = set()
        candidates: list[tuple[float, int]] = []
        best: list[tuple[float, int]] = []
        entry_list = list(entries)
        if level == 0 and len(entry_list) == 1:
            self._visit_token += 1
            if self._visit_token == np.iinfo(np.int64).max:
                self._visited.fill(0)
                self._visit_token = 1
            count = search_layer_zero(
                self._data[: self.element_count],
                query,
                self._link0[: self.element_count],
                self._link0_counts[: self.element_count],
                self._candidate_ids[: self.element_count],
                self._candidate_distances[: self.element_count],
                self._best_ids[: self.element_count],
                self._best_distances[: self.element_count],
                self._visited[: self.element_count],
                int(entry_list[0]),
                self._visit_token,
                ef,
                self._space_code,
                _trusted_graph=True,
            )
            return sorted(
                (float(self._best_distances[i]), int(self._best_ids[i]))
                for i in range(count)
            )
        entry_distances = self._distances(query, entry_list)
        for position, distance in zip(entry_list, entry_distances):
            pos = int(position)
            dist = float(distance)
            if pos in visited:
                continue
            visited.add(pos)
            heapq.heappush(candidates, (dist, pos))
            heapq.heappush(best, (-dist, pos))

        while candidates:
            candidate_distance, candidate = heapq.heappop(candidates)
            worst = -best[0][0]
            if len(best) >= ef and candidate_distance > worst:
                break
            neighbors = (
                self._links[candidate][level]
                if level < len(self._links[candidate])
                else []
            )
            unseen = [neighbor for neighbor in neighbors if neighbor not in visited]
            if not unseen:
                continue
            visited.update(unseen)
            distances = self._distances(query, unseen)
            for neighbor, distance in zip(unseen, distances):
                dist = float(distance)
                if len(best) < ef or dist < -best[0][0]:
                    heapq.heappush(candidates, (dist, neighbor))
                    heapq.heappush(best, (-dist, neighbor))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-negative_distance, position) for negative_distance, position in best)

    def _select_neighbors(
        self, query: np.ndarray, candidates: list[tuple[float, int]], limit: int
    ) -> list[int]:
        ordered = sorted(candidates)
        count = len(ordered)
        if not count:
            return []
        for offset, (distance, position) in enumerate(ordered):
            self._candidate_ids[offset] = position
            self._candidate_distances[offset] = distance
        selected_count = select_neighbors(
            self._data,
            self._candidate_ids[:count],
            self._candidate_distances[:count],
            self._best_ids[: min(limit, count)],
            min(limit, count),
            self._space_code,
        )
        return self._best_ids[:selected_count].tolist()

    def _prune(self, position: int, level: int, limit: int) -> None:
        neighbors = self._links[position][level]
        if len(neighbors) <= limit:
            return
        count = len(neighbors)
        self._candidate_ids[:count] = neighbors
        selected_count = prune_neighbors(
            self._data,
            self._candidate_ids,
            self._candidate_distances,
            self._best_ids,
            count,
            limit,
            position,
            self._space_code,
        )
        selected = self._best_ids[:selected_count].tolist()
        selected_set = set(selected)
        for dropped in neighbors:
            if dropped not in selected_set:
                reverse = self._links[dropped][level]
                if position in reverse:
                    reverse.remove(position)
                    if level == 0:
                        self._sync_link0(dropped)
        self._links[position][level] = selected
        if level == 0:
            self._sync_link0(position)

    def _insert(self, vector: np.ndarray, label: int) -> None:
        position = self.element_count
        level = self._random_level()
        self._ensure_graph_capacity(position + 1)
        self._data[position] = vector
        self._labels = np.append(self._labels, np.uint64(label))
        self._levels = np.append(self._levels, np.int16(level))
        self._links.append([[] for _ in range(level + 1)])
        self._label_to_pos[label] = position

        if self._entrypoint < 0:
            self._entrypoint = position
            self._max_level = level
            return

        entry = self._entrypoint
        for layer in range(self._max_level, level, -1):
            entry, _ = self._greedy(vector, entry, layer)

        for layer in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(vector, [entry], self.ef_construction, layer)
            degree = 2 * self.M if layer == 0 else self.M
            neighbors = self._select_neighbors(vector, candidates, degree)
            self._links[position][layer] = neighbors
            if layer == 0:
                self._sync_link0(position)
            for neighbor in list(neighbors):
                if neighbor not in self._links[position][layer]:
                    continue
                self._links[neighbor][layer].append(position)
                if layer == 0:
                    offset = int(self._link0_counts[neighbor])
                    if offset < self._link0.shape[1]:
                        self._link0[neighbor, offset] = position
                        self._link0_counts[neighbor] = offset + 1
                self._prune(neighbor, layer, degree)
            if candidates:
                entry = candidates[0][1]

        if level > self._max_level:
            self._entrypoint = position
            self._max_level = level

    def _rebuild(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        deleted_labels: set[int],
    ) -> None:
        capacity = self.max_elements
        M = self.M
        ef_construction = self.ef_construction
        seed = self._random_seed
        allow_replace_deleted = self._allow_replace_deleted
        current_ef = self.ef
        current_threads = self.num_threads
        self.init_index(capacity, M, ef_construction, seed, allow_replace_deleted)
        for vector, label in zip(vectors, labels):
            self._insert(vector, int(label))
        self._deleted = {
            self._label_to_pos[label]
            for label in deleted_labels
            if label in self._label_to_pos
        }
        self.ef = current_ef
        self.num_threads = current_threads

    def add_items(
        self,
        data,
        ids=None,
        num_threads: int = -1,
        replace_deleted: bool = False,
    ) -> None:
        self._require_initialized()
        if not isinstance(num_threads, (int, np.integer)):
            raise TypeError("num_threads must be an integer")
        vectors = self._prepare_vectors(data)
        if ids is None:
            labels = np.arange(self.element_count, self.element_count + len(vectors), dtype=np.int64)
        else:
            labels = np.asarray(ids)
            if labels.ndim == 0:
                labels = labels.reshape(1)
            if labels.ndim != 1 or len(labels) != len(vectors):
                raise RuntimeError("The number of labels must match the number of vectors")
            if not np.issubdtype(labels.dtype, np.integer):
                raise RuntimeError("Labels must be integers")
            if np.any(labels < 0):
                raise OverflowError("Labels must be non-negative integers")
            labels = labels.astype(np.uint64, copy=False)
        if len(set(map(int, labels))) != len(labels):
            raise RuntimeError("Labels must be unique")
        self.num_threads = int(num_threads)

        for vector, raw_label in zip(vectors, labels):
            label = int(raw_label)
            existing = self._label_to_pos.get(label)
            if existing is not None:
                updated = self._data[: self.element_count].copy()
                updated[existing] = vector
                deleted_labels = {
                    int(self._labels[position])
                    for position in self._deleted
                    if position != existing
                }
                self._rebuild(updated, self._labels.copy(), deleted_labels)
                continue
            if replace_deleted:
                if not self._allow_replace_deleted:
                    raise RuntimeError("Replacement of deleted elements is disabled")
                if self._deleted:
                    replaced = min(self._deleted)
                    keep = np.arange(self.element_count) != replaced
                    updated = np.concatenate(
                        [self._data[: self.element_count][keep], vector.reshape(1, -1)]
                    )
                    updated_labels = np.concatenate(
                        [self._labels[keep], np.asarray([label], dtype=np.uint64)]
                    )
                    deleted_labels = {
                        int(self._labels[position])
                        for position in self._deleted
                        if position != replaced
                    }
                    self._rebuild(updated, updated_labels, deleted_labels)
                    continue
            if self.element_count >= self.max_elements:
                raise RuntimeError("The number of elements exceeds the specified limit")
            self._insert(vector, label)

    def _query_one(
        self, query: np.ndarray, k: int, filter: Callable[[int], bool] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        entry = self._entrypoint
        for level in range(self._max_level, 0, -1):
            entry, _ = self._greedy(query, entry, level)
        ef = max(self.ef, k)
        candidates = self._search_layer(query, [entry], ef, 0)
        eligible = [
            (distance, position)
            for distance, position in candidates
            if position not in self._deleted
            and (filter is None or bool(filter(int(self._labels[position]))))
        ]

        if len(eligible) < k:
            all_distances = np.empty(self.element_count, dtype=np.float32)
            distance_all(
                self._data, query, all_distances, self.element_count, self._space_code
            )
            seen = {position for _, position in eligible}
            missing = [
                (float(all_distances[position]), position)
                for position in range(self.element_count)
                if position not in seen
                and position not in self._deleted
                and (filter is None or bool(filter(int(self._labels[position]))))
            ]
            eligible.extend(missing)
        eligible.sort()
        chosen = eligible[:k]
        if len(chosen) < k:
            raise RuntimeError("Cannot return the results in a contiguous 2D array. Probably ef or M is too small")
        labels = np.asarray([self._labels[position] for _, position in chosen], dtype=np.uint64)
        distances = np.asarray([distance for distance, _ in chosen], dtype=np.float32)
        return labels, distances

    def knn_query(self, data, k: int = 1, num_threads: int = -1, filter=None):
        self._require_initialized()
        if not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer")
        if not isinstance(num_threads, (int, np.integer)):
            raise TypeError("num_threads must be an integer")
        if k <= 0:
            raise ValueError("k must be positive")
        available = self.element_count - len(self._deleted)
        if k > available:
            raise RuntimeError("Cannot return the results in a contiguous 2D array")
        queries = self._prepare_vectors(data)
        self.num_threads = int(num_threads)
        labels = np.empty((len(queries), k), dtype=np.uint64)
        distances = np.empty((len(queries), k), dtype=np.float32)
        for row, query in enumerate(queries):
            labels[row], distances[row] = self._query_one(query, int(k), filter)
        return labels, distances

    def get_items(self, ids=None, return_type: str = "numpy"):
        self._require_initialized()
        if ids is None:
            positions = np.arange(self.element_count, dtype=np.int64)
        else:
            raw = np.asarray(ids)
            if raw.ndim == 0:
                raw = raw.reshape(1)
            try:
                positions = np.asarray(
                    [self._label_to_pos[int(label)] for label in raw], dtype=np.int64
                )
            except KeyError as error:
                raise RuntimeError(f"Label not found: {error.args[0]}") from None
        result = self._data[positions].copy()
        if return_type == "numpy":
            return result
        if return_type == "list":
            return result.tolist()
        raise ValueError("return_type must be 'numpy' or 'list'")

    def get_ids_list(self) -> list[int]:
        return [int(label) for label in self._labels]

    def mark_deleted(self, label: int) -> None:
        try:
            position = self._label_to_pos[int(label)]
        except KeyError:
            raise RuntimeError(f"Label not found: {label}") from None
        if position in self._deleted:
            raise RuntimeError("The requested label is already deleted")
        self._deleted.add(position)

    def unmark_deleted(self, label: int) -> None:
        try:
            position = self._label_to_pos[int(label)]
        except KeyError:
            raise RuntimeError(f"Label not found: {label}") from None
        if position not in self._deleted:
            raise RuntimeError("The requested label is not deleted")
        self._deleted.remove(position)

    def save_index(self, path_to_index: str) -> None:
        self._require_initialized()
        state = np.empty((), dtype=object)
        state[()] = {
            "version": 1,
            "space": self.space,
            "dim": self.dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef": self.ef,
            "num_threads": self.num_threads,
            "max_elements": self.max_elements,
            "random_seed": self._random_seed,
            "allow_replace_deleted": self._allow_replace_deleted,
            "data": self._data[: self.element_count].copy(),
            "labels": self._labels,
            "levels": self._levels,
            "links": self._links,
            "deleted": self._deleted,
            "entrypoint": self._entrypoint,
            "max_level": self._max_level,
        }
        with open(os.fspath(path_to_index), "wb") as stream:
            np.save(stream, state, allow_pickle=True)

    def load_index(
        self,
        path_to_index: str,
        max_elements: int = 0,
        allow_replace_deleted: bool = False,
    ) -> None:
        if not isinstance(max_elements, (int, np.integer)):
            raise TypeError("max_elements must be an integer")
        if max_elements < 0:
            raise ValueError("max_elements must be non-negative")
        with open(os.fspath(path_to_index), "rb") as stream:
            state = np.load(stream, allow_pickle=True).item()
        if not isinstance(state, dict):
            raise RuntimeError("Invalid mojo-hnswlib index state")
        if state.get("version") != 1:
            raise RuntimeError("Unsupported mojo-hnswlib index version")
        required = {
            "space", "dim", "M", "ef_construction", "ef", "num_threads",
            "max_elements", "random_seed", "data", "labels", "levels", "links",
            "deleted", "entrypoint", "max_level",
        }
        if not required.issubset(state):
            raise RuntimeError("Incomplete mojo-hnswlib index state")
        if state["space"] != self.space or state["dim"] != self.dim:
            raise RuntimeError("Index space or dimensionality does not match")
        data = state["data"]
        labels = state["labels"]
        levels = state["levels"]
        links = state["links"]
        if (
            not isinstance(data, np.ndarray)
            or data.dtype != np.float32
            or data.ndim != 2
            or data.shape[1] != self.dim
            or not isinstance(labels, np.ndarray)
            or labels.dtype != np.uint64
            or labels.ndim != 1
            or not isinstance(levels, np.ndarray)
            or levels.dtype != np.int16
            or levels.ndim != 1
        ):
            raise RuntimeError("Invalid array layout in saved index")
        count = len(labels)
        if data.shape[0] != count or len(levels) != count:
            raise RuntimeError("Inconsistent array lengths in saved index")
        if len(set(map(int, labels))) != count:
            raise RuntimeError("Saved index contains duplicate labels")
        M = int(state["M"])
        if not isinstance(links, list) or len(links) != count or M <= 1:
            raise RuntimeError("Invalid graph in saved index")
        for position, node_levels in enumerate(links):
            if not isinstance(node_levels, list) or len(node_levels) != int(levels[position]) + 1:
                raise RuntimeError("Invalid graph levels in saved index")
            for level, neighbors in enumerate(node_levels):
                limit = 2 * M if level == 0 else M
                if (
                    not isinstance(neighbors, list)
                    or len(neighbors) > limit
                    or len(set(neighbors)) != len(neighbors)
                    or any(
                        not isinstance(neighbor, (int, np.integer))
                        or not 0 <= int(neighbor) < count
                        for neighbor in neighbors
                    )
                ):
                    raise RuntimeError("Invalid graph links in saved index")
        deleted = state["deleted"]
        if not isinstance(deleted, set) or any(
            not isinstance(position, (int, np.integer))
            or not 0 <= int(position) < count
            for position in deleted
        ):
            raise RuntimeError("Invalid deleted set in saved index")
        entrypoint = int(state["entrypoint"])
        max_level = int(state["max_level"])
        if (count == 0 and (entrypoint != -1 or max_level != -1)) or (
            count and (not 0 <= entrypoint < count or max_level != int(levels[entrypoint]))
        ):
            raise RuntimeError("Invalid entry point in saved index")
        capacity = int(max_elements) if max_elements else int(state["max_elements"])
        if capacity < count:
            raise RuntimeError("max_elements is smaller than the saved index")
        self.init_index(
            capacity,
            M=M,
            ef_construction=int(state["ef_construction"]),
            random_seed=int(state["random_seed"]),
            allow_replace_deleted=allow_replace_deleted,
        )
        self._data[:count] = data
        self._labels = labels.copy()
        self._levels = levels.copy()
        self._links = links
        self._deleted = set(deleted)
        self._entrypoint = entrypoint
        self._max_level = max_level
        self._label_to_pos = {
            int(label): position for position, label in enumerate(self._labels)
        }
        self._ensure_graph_capacity(count)
        for position in range(count):
            self._sync_link0(position)
        self.ef = int(state["ef"])
        self.num_threads = int(state["num_threads"])

    def __repr__(self) -> str:
        return f"<mojo_hnswlib.Index(space={self.space!r}, dim={self.dim})>"
