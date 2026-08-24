"""Numerical and behavioural parity with the real hnswlib 0.8 Python package."""

from __future__ import annotations

import numpy as np
import pytest

import hnswlib as upstream
import mojo_hnswlib as mojo
from mojo_hnswlib._lib import (
    distance_all,
    distance_indexed,
    pair_distance,
    search_layer_zero,
)


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(240, 12)).astype(np.float32)
    queries = rng.normal(size=(24, 12)).astype(np.float32)
    labels = np.arange(1000, 1240, dtype=np.uint64)
    return data, queries, labels


@pytest.fixture(scope="module")
def indexes(corpus):
    data, _, labels = corpus
    result = {}
    for space in ("l2", "ip", "cosine"):
        ours = mojo.Index(space=space, dim=data.shape[1])
        theirs = upstream.Index(space=space, dim=data.shape[1])
        for index in (ours, theirs):
            index.init_index(
                max_elements=len(data), M=12, ef_construction=100, random_seed=7
            )
            index.add_items(data, labels)
            index.set_ef(120)
        result[space] = ours, theirs
    return result


def exact_distances(space, data, queries):
    if space == "l2":
        return np.sum((queries[:, None] - data[None]) ** 2, axis=2)
    if space == "ip":
        return 1.0 - queries @ data.T
    data_norms = np.linalg.norm(data, axis=1, keepdims=True)
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    normalized_data = np.divide(data, data_norms, out=np.zeros_like(data), where=data_norms != 0)
    normalized_queries = np.divide(
        queries, query_norms, out=np.zeros_like(queries), where=query_norms != 0
    )
    return 1.0 - normalized_queries @ normalized_data.T


def recall_at(labels, truth, external_labels):
    expected = external_labels[truth]
    return np.mean(
        [len(set(map(int, actual)) & set(map(int, wanted))) / truth.shape[1]
         for actual, wanted in zip(labels, expected)]
    )


def test_fresh_index_properties_match_upstream():
    ours = mojo.Index("l2", 5)
    theirs = upstream.Index("l2", 5)
    for name in ("space", "dim", "M", "ef_construction", "max_elements", "element_count", "ef"):
        assert getattr(ours, name) == getattr(theirs, name)


@pytest.mark.parametrize("space", ["l2", "ip", "cosine"])
def test_knn_recall_and_distance_parity(space, corpus, indexes):
    data, queries, labels = corpus
    ours, theirs = indexes[space]
    ours_labels, ours_distances = ours.knn_query(queries, k=10)
    their_labels, their_distances = theirs.knn_query(queries, k=10)
    exact = exact_distances(space, data, queries)
    truth = np.argsort(exact, axis=1)[:, :10]

    our_recall = recall_at(ours_labels, truth, labels)
    their_recall = recall_at(their_labels, truth, labels)
    assert our_recall >= 0.98
    assert abs(our_recall - their_recall) <= 0.02

    positions = ours_labels.astype(np.int64) - 1000
    expected_distances = np.take_along_axis(exact, positions, axis=1)
    assert np.allclose(ours_distances, expected_distances, atol=4e-5)
    assert ours_labels.dtype == their_labels.dtype == np.uint64
    assert ours_distances.dtype == their_distances.dtype == np.float32


@pytest.mark.parametrize("space", ["l2", "ip", "cosine"])
def test_mojo_distance_kernels_match_numpy(space, corpus):
    data, queries, _ = corpus
    if space == "cosine":
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        query = queries[0] / np.linalg.norm(queries[0])
    else:
        query = queries[0]
    data = np.ascontiguousarray(data, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    code = {"l2": 0, "ip": 1, "cosine": 2}[space]
    expected = exact_distances(space, data, query[None])[0]

    all_result = np.empty(len(data), dtype=np.float32)
    distance_all(data, query, all_result, len(data), code)
    assert np.allclose(all_result, expected, atol=4e-5)

    selected = np.ascontiguousarray([9, 2, 170, 33], dtype=np.int64)
    indexed_result = np.empty(len(selected), dtype=np.float32)
    distance_indexed(data, query, selected, indexed_result, code)
    assert np.allclose(indexed_result, expected[selected], atol=4e-5)


@pytest.mark.parametrize("count", [4095, 4096])
def test_distance_simd_tail_across_parallel_threshold(count):
    rng = np.random.default_rng(count)
    data = rng.normal(size=(count, 13)).astype(np.float32)
    query = rng.normal(size=13).astype(np.float32)
    expected = np.sum((data - query) ** 2, axis=1)

    all_result = np.empty(count, dtype=np.float32)
    distance_all(data, query, all_result, count, 0)
    assert np.allclose(all_result, expected, atol=4e-5)

    indices = np.ascontiguousarray(rng.permutation(count), dtype=np.int64)
    indexed_result = np.empty(count, dtype=np.float32)
    distance_indexed(data, query, indices, indexed_result, 0)
    assert np.allclose(indexed_result, expected[indices], atol=4e-5)


def test_ffi_rejects_unsafe_buffer_descriptions():
    data = np.zeros((4, 3), dtype=np.float32)
    query = np.zeros(3, dtype=np.float32)
    result = np.empty(4, dtype=np.float32)

    with pytest.raises(TypeError):
        distance_all(data.astype(np.float64), query, result, 4, 0)
    with pytest.raises(ValueError):
        distance_all(data[:, ::-1], query, result, 4, 0)
    with pytest.raises(ValueError):
        distance_all(data, query, result[:2], 4, 0)
    with pytest.raises(ValueError):
        distance_all(data, query, result, 5, 0)
    with pytest.raises(IndexError):
        distance_indexed(
            data, query, np.asarray([4], dtype=np.int64), result[:1], 0
        )
    with pytest.raises(ValueError):
        pair_distance(query, query[:2], 0)


def test_search_ffi_rejects_invalid_graph_bounds():
    data = np.zeros((2, 3), dtype=np.float32)
    query = np.zeros(3, dtype=np.float32)
    links = np.zeros((2, 2), dtype=np.int64)
    counts = np.asarray([1, 0], dtype=np.int64)
    scratch_i = np.empty(2, dtype=np.int64)
    scratch_f = np.empty(2, dtype=np.float32)
    visited = np.zeros(2, dtype=np.int64)
    links[0, 0] = 2
    with pytest.raises(IndexError):
        search_layer_zero(
            data,
            query,
            links,
            counts,
            scratch_i,
            scratch_f,
            scratch_i.copy(),
            scratch_f.copy(),
            visited,
            0,
            1,
            2,
            0,
        )


def test_level_zero_search_returns_ordered_simd_tail_results():
    data = np.asarray(
        [[4.0, 0.0, 1.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    query = np.zeros(3, dtype=np.float32)
    links = np.asarray(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64
    )
    counts = np.full(4, 3, dtype=np.int64)
    candidate_ids = np.empty(4, dtype=np.int64)
    candidate_distances = np.empty(4, dtype=np.float32)
    best_ids = np.empty(4, dtype=np.int64)
    best_distances = np.empty(4, dtype=np.float32)
    visited = np.zeros(4, dtype=np.int64)

    count = search_layer_zero(
        data,
        query,
        links,
        counts,
        candidate_ids,
        candidate_distances,
        best_ids,
        best_distances,
        visited,
        0,
        1,
        4,
        0,
    )

    assert count == 4
    assert np.array_equal(best_ids, np.asarray([1, 3, 2, 0], dtype=np.int64))
    assert np.allclose(best_distances, np.asarray([1.0, 4.0, 9.0, 17.0]))


def test_cosine_stores_normalized_vectors(corpus, indexes):
    data, _, labels = corpus
    ours, theirs = indexes["cosine"]
    chosen = labels[[0, 19, 103]]
    assert np.allclose(ours.get_items(chosen), theirs.get_items(chosen), atol=1e-6)
    assert np.allclose(np.linalg.norm(ours.get_items(chosen), axis=1), 1.0)
    assert isinstance(ours.get_items(chosen, return_type="list"), list)


def test_delete_unmark_and_filter_match_upstream(corpus):
    data, queries, labels = corpus
    instances = []
    for cls in (mojo.Index, upstream.Index):
        index = cls("l2", data.shape[1])
        index.init_index(len(data), M=12, ef_construction=100)
        index.add_items(data, labels)
        index.set_ef(120)
        instances.append(index)
    nearest = int(instances[1].knn_query(queries[0], k=1)[0][0, 0])
    for index in instances:
        index.mark_deleted(nearest)
    for index in instances:
        found = index.knn_query(queries[0], k=20)[0][0]
        assert nearest not in found
        filtered = index.knn_query(queries[:3], k=6, filter=lambda label: label % 3 == 0)[0]
        assert np.all(filtered % 3 == 0)
        index.unmark_deleted(nearest)
        assert nearest in index.knn_query(queries[0], k=20)[0][0]


def test_update_existing_label_changes_search_result():
    data = np.asarray([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    for cls in (mojo.Index, upstream.Index):
        index = cls("l2", 2)
        index.init_index(3)
        index.add_items(data, [7, 8, 9])
        index.add_items(np.asarray([[9.9, 10.1]], dtype=np.float32), [7])
        labels, distances = index.knn_query([[10.0, 10.0]], k=2)
        assert labels[0, 0] == 8
        assert labels[0, 1] == 7
        assert distances[0, 1] == pytest.approx(0.02, abs=1e-5)


def test_replace_deleted_and_capacity():
    index = mojo.Index("l2", 2)
    index.init_index(3, allow_replace_deleted=True)
    index.add_items([[0, 0], [1, 1], [2, 2]], [10, 11, 12])
    index.mark_deleted(11)
    index.add_items([[1.1, 1.1]], [99], replace_deleted=True)
    assert index.element_count == 3
    assert set(index.get_ids_list()) == {10, 12, 99}
    assert index.knn_query([[1, 1]], k=1)[0][0, 0] == 99


def test_resize_and_maximum_capacity():
    index = mojo.Index("l2", 2)
    index.init_index(2)
    index.add_items([[0, 0], [1, 1]])
    with pytest.raises(RuntimeError):
        index.add_items([[2, 2]])
    index.resize_index(3)
    index.add_items([[2, 2]])
    assert index.get_current_count() == 3
    assert index.get_max_elements() == 3
    with pytest.raises(RuntimeError):
        index.resize_index(2)


def test_save_load_round_trip(tmp_path, corpus):
    data, queries, labels = corpus
    original = mojo.Index("cosine", data.shape[1])
    original.init_index(300, M=10, ef_construction=80, random_seed=19)
    original.add_items(data[:80], labels[:80])
    original.mark_deleted(int(labels[7]))
    original.set_ef(60)
    before = original.knn_query(queries[:4], k=8)
    path = tmp_path / "index.bin"
    original.save_index(path)

    restored = mojo.Index("cosine", data.shape[1])
    restored.load_index(path)
    after = restored.knn_query(queries[:4], k=8)
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])
    assert restored.max_elements == 300
    assert restored.ef == 60
    assert int(labels[7]) not in restored.knn_query(queries[:4], k=40)[0]


def test_load_rejects_graph_that_could_escape_ffi_buffers(tmp_path):
    index = mojo.Index("l2", 2)
    index.init_index(2)
    index.add_items([[0, 0], [1, 1]])
    path = tmp_path / "valid.bin"
    index.save_index(path)
    with path.open("rb") as stream:
        state = np.load(stream, allow_pickle=True).item()
    state["links"][0][0] = [len(state["labels"])]
    corrupt = tmp_path / "corrupt.bin"
    wrapped = np.empty((), dtype=object)
    wrapped[()] = state
    with corrupt.open("wb") as stream:
        np.save(stream, wrapped, allow_pickle=True)

    restored = mojo.Index("l2", 2)
    with pytest.raises(RuntimeError, match="Invalid graph links"):
        restored.load_index(corrupt)


def test_graph_has_bounded_bidirectional_multilayer_structure(corpus, indexes):
    ours, _ = indexes["l2"]
    assert ours._max_level >= 1
    assert any(len(levels) > 1 for levels in ours._links)
    for position, levels in enumerate(ours._links):
        for level, neighbors in enumerate(levels):
            limit = 2 * ours.M if level == 0 else ours.M
            assert len(neighbors) <= limit
            assert len(neighbors) == len(set(neighbors))
            for neighbor in neighbors:
                assert position in ours._links[neighbor][level]


def test_validation_and_shapes():
    with pytest.raises(RuntimeError):
        mojo.Index("angular", 3)
    index = mojo.Index("l2", 3)
    with pytest.raises(RuntimeError):
        index.add_items([[1, 2, 3]])
    index.init_index(4)
    with pytest.raises(RuntimeError):
        index.add_items([[1, 2]])
    with pytest.raises(OverflowError):
        index.add_items([[1, 2, 3]], [-1])
    index.add_items([[1, 2, 3], [3, 2, 1]], [10, 11])
    labels, distances = index.knn_query([1, 2, 3], k=1)
    assert labels.shape == distances.shape == (1, 1)
    with pytest.raises(RuntimeError):
        index.knn_query([1, 2, 3], k=3)
    with pytest.raises(TypeError):
        index.resize_index(4.5)
    with pytest.raises(TypeError):
        index.set_ef(4.5)


def test_thread_controls_are_accepted_and_reported(corpus):
    data, queries, _ = corpus
    index = mojo.Index("l2", data.shape[1])
    index.init_index(8)
    index.set_num_threads(2)
    assert index.num_threads == 2
    index.add_items(data[:8], num_threads=1)
    assert index.num_threads == 1
    index.knn_query(queries[:1], num_threads=3)
    assert index.num_threads == 3
