"""SIMD distance kernels used by the Python HNSW graph implementation."""

from std.algorithm import parallelize
from std.sys.info import simd_width_of as simdwidthof

comptime W = simdwidthof[DType.float64]()
comptime PARALLEL_DISTANCE_THRESHOLD = 4096
comptime PARALLEL_DISTANCE_WORKERS = 8
comptime FPtr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def vector_distance(a: FPtr, b: FPtr, dim: Int, space: Int) -> Float32:
    var acc0 = SIMD[DType.float32, W](0.0)
    var acc1 = SIMD[DType.float32, W](0.0)
    var i = 0
    if space == 0:
        while i + 2 * W <= dim:
            var delta0 = a.load[width=W](i) - b.load[width=W](i)
            var delta1 = a.load[width=W](i + W) - b.load[width=W](i + W)
            acc0 += delta0 * delta0
            acc1 += delta1 * delta1
            i += 2 * W
        while i + W <= dim:
            var delta = a.load[width=W](i) - b.load[width=W](i)
            acc0 += delta * delta
            i += W
        var total = (acc0 + acc1).reduce_add()
        while i < dim:
            var delta = a[i] - b[i]
            total += delta * delta
            i += 1
        return total

    while i + 2 * W <= dim:
        acc0 += a.load[width=W](i) * b.load[width=W](i)
        acc1 += a.load[width=W](i + W) * b.load[width=W](i + W)
        i += 2 * W
    while i + W <= dim:
        acc0 += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = (acc0 + acc1).reduce_add()
    while i < dim:
        total += a[i] * b[i]
        i += 1
    return 1.0 - total


@export("mh_distance_indexed")
def mh_distance_indexed(
    data_addr: Int,
    query_addr: Int,
    indices_addr: Int,
    result_addr: Int,
    count: Int,
    dim: Int,
    space: Int,
) abi("C"):
    var data = FPtr(unsafe_from_address=data_addr)
    var query = FPtr(unsafe_from_address=query_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    if count < PARALLEL_DISTANCE_THRESHOLD:
        for i in range(count):
            result[i] = vector_distance(
                data + Int(indices[i]) * dim, query, dim, space
            )
        return

    @parameter
    def work(task: Int):
        var task_data = FPtr(unsafe_from_address=data_addr)
        var task_query = FPtr(unsafe_from_address=query_addr)
        var task_indices = IPtr(unsafe_from_address=indices_addr)
        var task_result = FPtr(unsafe_from_address=result_addr)
        var chunk = (
            count + PARALLEL_DISTANCE_WORKERS - 1
        ) // PARALLEL_DISTANCE_WORKERS
        var start = task * chunk
        var end = min(start + chunk, count)
        for i in range(start, end):
            task_result[i] = vector_distance(
                task_data + Int(task_indices[i]) * dim,
                task_query,
                dim,
                space,
            )

    parallelize[work](PARALLEL_DISTANCE_WORKERS)


@export("mh_distance_all")
def mh_distance_all(
    data_addr: Int,
    query_addr: Int,
    result_addr: Int,
    count: Int,
    dim: Int,
    space: Int,
) abi("C"):
    var data = FPtr(unsafe_from_address=data_addr)
    var query = FPtr(unsafe_from_address=query_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    if count < PARALLEL_DISTANCE_THRESHOLD:
        for i in range(count):
            result[i] = vector_distance(data + i * dim, query, dim, space)
        return

    @parameter
    def work(task: Int):
        var task_data = FPtr(unsafe_from_address=data_addr)
        var task_query = FPtr(unsafe_from_address=query_addr)
        var task_result = FPtr(unsafe_from_address=result_addr)
        var chunk = (
            count + PARALLEL_DISTANCE_WORKERS - 1
        ) // PARALLEL_DISTANCE_WORKERS
        var start = task * chunk
        var end = min(start + chunk, count)
        for i in range(start, end):
            task_result[i] = vector_distance(
                task_data + i * dim, task_query, dim, space
            )

    parallelize[work](PARALLEL_DISTANCE_WORKERS)


@export("mh_pair_distance")
def mh_pair_distance(
    a_addr: Int, b_addr: Int, dim: Int, space: Int
) abi("C") -> Float32:
    return vector_distance(
        FPtr(unsafe_from_address=a_addr),
        FPtr(unsafe_from_address=b_addr),
        dim,
        space,
    )


def candidate_less(
    left_distance: Float32,
    left_id: Int64,
    right_distance: Float32,
    right_id: Int64,
) -> Bool:
    return (
        left_distance < right_distance
        or (left_distance == right_distance and left_id < right_id)
    )


def best_less(
    left_distance: Float32,
    left_id: Int64,
    right_distance: Float32,
    right_id: Int64,
) -> Bool:
    return (
        left_distance > right_distance
        or (left_distance == right_distance and left_id < right_id)
    )


def candidate_push(
    ids: IPtr, distances: FPtr, size: Int, node: Int64, distance: Float32
):
    var child = size
    while child > 0:
        var parent = (child - 1) // 2
        if not candidate_less(distance, node, distances[parent], ids[parent]):
            break
        ids[child] = ids[parent]
        distances[child] = distances[parent]
        child = parent
    ids[child] = node
    distances[child] = distance


def best_push(
    ids: IPtr, distances: FPtr, size: Int, node: Int64, distance: Float32
):
    var child = size
    while child > 0:
        var parent = (child - 1) // 2
        if not best_less(distance, node, distances[parent], ids[parent]):
            break
        ids[child] = ids[parent]
        distances[child] = distances[parent]
        child = parent
    ids[child] = node
    distances[child] = distance


def candidate_pop(ids: IPtr, distances: FPtr, size: Int):
    var last = size - 1
    if last == 0:
        return
    var node = ids[last]
    var distance = distances[last]
    var parent = 0
    while True:
        var left = 2 * parent + 1
        if left >= last:
            break
        var child = left
        var right = left + 1
        if (
            right < last
            and candidate_less(
                distances[right], ids[right], distances[left], ids[left]
            )
        ):
            child = right
        if not candidate_less(
            distances[child], ids[child], distance, node
        ):
            break
        ids[parent] = ids[child]
        distances[parent] = distances[child]
        parent = child
    ids[parent] = node
    distances[parent] = distance


def best_pop(ids: IPtr, distances: FPtr, size: Int):
    var last = size - 1
    if last == 0:
        return
    var node = ids[last]
    var distance = distances[last]
    var parent = 0
    while True:
        var left = 2 * parent + 1
        if left >= last:
            break
        var child = left
        var right = left + 1
        if (
            right < last
            and best_less(
                distances[right], ids[right], distances[left], ids[left]
            )
        ):
            child = right
        if not best_less(distances[child], ids[child], distance, node):
            break
        ids[parent] = ids[child]
        distances[parent] = distances[child]
        parent = child
    ids[parent] = node
    distances[parent] = distance


@export("mh_search_layer_zero")
def mh_search_layer_zero(
    data_addr: Int,
    query_addr: Int,
    links_addr: Int,
    link_counts_addr: Int,
    candidate_ids_addr: Int,
    candidate_distances_addr: Int,
    best_ids_addr: Int,
    best_distances_addr: Int,
    visited_addr: Int,
    entry: Int,
    visit_token: Int64,
    ef: Int,
    degree: Int,
    dim: Int,
    space: Int,
) abi("C") -> Int:
    var data = FPtr(unsafe_from_address=data_addr)
    var query = FPtr(unsafe_from_address=query_addr)
    var links = IPtr(unsafe_from_address=links_addr)
    var link_counts = IPtr(unsafe_from_address=link_counts_addr)
    var candidate_ids = IPtr(unsafe_from_address=candidate_ids_addr)
    var candidate_distances = FPtr(
        unsafe_from_address=candidate_distances_addr
    )
    var best_ids = IPtr(unsafe_from_address=best_ids_addr)
    var best_distances = FPtr(unsafe_from_address=best_distances_addr)
    var visited = IPtr(unsafe_from_address=visited_addr)

    var entry_id = Int64(entry)
    var entry_distance = vector_distance(
        data + entry * dim, query, dim, space
    )
    visited[entry] = visit_token
    candidate_push(
        candidate_ids, candidate_distances, 0, entry_id, entry_distance
    )
    best_push(best_ids, best_distances, 0, entry_id, entry_distance)
    var candidate_count = 1
    var best_count = 1

    while candidate_count > 0:
        var candidate = Int(candidate_ids[0])
        var candidate_distance = candidate_distances[0]
        candidate_pop(
            candidate_ids, candidate_distances, candidate_count
        )
        candidate_count -= 1
        if (
            best_count >= ef
            and candidate_distance > best_distances[0]
        ):
            break

        var base = candidate * degree
        for offset in range(Int(link_counts[candidate])):
            var neighbor_id = links[base + offset]
            var neighbor = Int(neighbor_id)
            if visited[neighbor] == visit_token:
                continue
            visited[neighbor] = visit_token
            var distance = vector_distance(
                data + neighbor * dim, query, dim, space
            )
            if best_count < ef or distance < best_distances[0]:
                candidate_push(
                    candidate_ids,
                    candidate_distances,
                    candidate_count,
                    neighbor_id,
                    distance,
                )
                candidate_count += 1
                best_push(
                    best_ids,
                    best_distances,
                    best_count,
                    neighbor_id,
                    distance,
                )
                best_count += 1
                if best_count > ef:
                    best_pop(best_ids, best_distances, best_count)
                    best_count -= 1
    return best_count


def select_neighbors_core(
    data: FPtr,
    candidates: IPtr,
    distances: FPtr,
    selected: IPtr,
    count: Int,
    limit: Int,
    dim: Int,
    space: Int,
) -> Int:
    var selected_count = 0
    for i in range(count):
        if selected_count == limit:
            break
        var candidate = Int(candidates[i])
        var good = True
        for j in range(selected_count):
            var other = Int(selected[j])
            if (
                vector_distance(
                    data + candidate * dim, data + other * dim, dim, space
                )
                < distances[i]
            ):
                good = False
                break
        if good:
            selected[selected_count] = candidates[i]
            selected_count += 1

    if selected_count < limit:
        for i in range(count):
            if selected_count == limit:
                break
            var already_selected = False
            for j in range(selected_count):
                if selected[j] == candidates[i]:
                    already_selected = True
                    break
            if not already_selected:
                selected[selected_count] = candidates[i]
                selected_count += 1
    return selected_count


@export("mh_select_neighbors")
def mh_select_neighbors(
    data_addr: Int,
    candidates_addr: Int,
    distances_addr: Int,
    selected_addr: Int,
    count: Int,
    limit: Int,
    dim: Int,
    space: Int,
) abi("C") -> Int:
    return select_neighbors_core(
        FPtr(unsafe_from_address=data_addr),
        IPtr(unsafe_from_address=candidates_addr),
        FPtr(unsafe_from_address=distances_addr),
        IPtr(unsafe_from_address=selected_addr),
        count,
        limit,
        dim,
        space,
    )


@export("mh_prune_neighbors")
def mh_prune_neighbors(
    data_addr: Int,
    candidates_addr: Int,
    distances_addr: Int,
    selected_addr: Int,
    count: Int,
    limit: Int,
    query_position: Int,
    dim: Int,
    space: Int,
) abi("C") -> Int:
    var data = FPtr(unsafe_from_address=data_addr)
    var candidates = IPtr(unsafe_from_address=candidates_addr)
    var distances = FPtr(unsafe_from_address=distances_addr)
    var selected = IPtr(unsafe_from_address=selected_addr)
    var query = data + query_position * dim
    for i in range(count):
        distances[i] = vector_distance(
            query, data + Int(candidates[i]) * dim, dim, space
        )

    for i in range(1, count):
        var candidate = candidates[i]
        var distance = distances[i]
        var j = i
        while (
            j > 0
            and (
                distance < distances[j - 1]
                or (
                    distance == distances[j - 1]
                    and candidate < candidates[j - 1]
                )
            )
        ):
            candidates[j] = candidates[j - 1]
            distances[j] = distances[j - 1]
            j -= 1
        candidates[j] = candidate
        distances[j] = distance

    return select_neighbors_core(
        data,
        candidates,
        distances,
        selected,
        count,
        limit,
        dim,
        space,
    )
