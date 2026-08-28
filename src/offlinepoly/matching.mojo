"""Pairwise assignment solving and the single-tracker STWO module.

The paper solves the symmetric tracklet-pairing problem with the blossom
algorithm. Here the thresholded cost graph is split into connected
components; small components are solved exactly (exhaustive maximum-weight
matching — components are almost always 2-4 nodes), large ones greedily.
"""

from .config import OfflinePolyConfig
from .geometry import iou_3d
from .motion import interpolate_gaps, predict_at
from .types import BoxState, Tracklet

comptime INVALID_COST = 1.0e18
comptime PAIR_BONUS = 10.0  # ensures maximum cardinality before cost


@fieldwise_init
struct Pair(ImplicitlyCopyable, Movable):
    var a: Int
    var b: Int


def _component_nodes(
    n: Int, cost: List[Float64], threshold: Float64
) -> List[List[Int]]:
    """Connected components of the graph with edges cost[i,j] < threshold."""
    var comp = List[Int]()
    for _ in range(n):
        comp.append(-1)
    var components = List[List[Int]]()
    for start in range(n):
        if comp[start] >= 0:
            continue
        var cid = len(components)
        var stack: List[Int] = [start]
        comp[start] = cid
        var members = List[Int]()
        while len(stack) > 0:
            var u = stack.pop()
            members.append(u)
            for v in range(n):
                if comp[v] < 0 and cost[u * n + v] < threshold:
                    comp[v] = cid
                    stack.append(v)
        components.append(members^)
    return components^


def _search_exact(
    pos: Int,
    nodes: List[Int],
    n: Int,
    cost: List[Float64],
    threshold: Float64,
    mut used: List[Bool],
    mut current: List[Pair],
    current_score: Float64,
    mut best_score: Float64,
    mut best: List[Pair],
):
    """Exhaustive max-weight matching over one component."""
    var k = pos
    while k < len(nodes) and used[k]:
        k += 1
    if k >= len(nodes):
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
        return
    # option 1: leave nodes[k] unmatched
    used[k] = True
    _search_exact(
        k + 1, nodes, n, cost, threshold, used, current, current_score,
        best_score, best,
    )
    # option 2: match nodes[k] with a later unused neighbor
    for j in range(k + 1, len(nodes)):
        if used[j]:
            continue
        var c = cost[nodes[k] * n + nodes[j]]
        if c >= threshold:
            continue
        used[j] = True
        current.append(Pair(nodes[k], nodes[j]))
        _search_exact(
            k + 1, nodes, n, cost, threshold, used, current,
            current_score + PAIR_BONUS - c, best_score, best,
        )
        _ = current.pop()
        used[j] = False
    used[k] = False


def _greedy_matching(
    nodes: List[Int], n: Int, cost: List[Float64], threshold: Float64
) -> List[Pair]:
    """Fallback for large components: pick edges by ascending cost."""
    var edges = List[Pair]()
    var weights = List[Float64]()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            var c = cost[nodes[i] * n + nodes[j]]
            if c < threshold:
                edges.append(Pair(nodes[i], nodes[j]))
                weights.append(c)
    # insertion sort edges by cost
    for i in range(1, len(edges)):
        var e = edges[i]
        var w = weights[i]
        var j = i - 1
        while j >= 0 and weights[j] > w:
            edges[j + 1] = edges[j]
            weights[j + 1] = weights[j]
            j -= 1
        edges[j + 1] = e
        weights[j + 1] = w
    var taken = Dict[Int, Bool]()
    var out = List[Pair]()
    for e in edges:
        if e.a in taken or e.b in taken:
            continue
        taken[e.a] = True
        taken[e.b] = True
        out.append(e)
    return out^


def solve_matching(
    n: Int, cost: List[Float64], threshold: Float64, max_exact: Int
) -> List[Pair]:
    """One-to-one pairs (i, j), i < j, with cost < threshold, maximizing
    matched pairs first and total affinity second."""
    var out = List[Pair]()
    for members in _component_nodes(n, cost, threshold):
        if len(members) < 2:
            continue
        if len(members) <= max_exact:
            var used = List[Bool]()
            for _ in range(len(members)):
                used.append(False)
            var current = List[Pair]()
            var best = List[Pair]()
            var best_score = -1.0
            _search_exact(
                0, members, n, cost, threshold, used, current, 0.0,
                best_score, best,
            )
            for p in best:  # _search_exact stores global node ids
                out.append(p)
        else:
            for p in _greedy_matching(members, n, cost, threshold):
                out.append(p)
    return out^


def merge_pair(
    a: Tracklet,
    b: Tracklet,
    frame_times: List[Float64],
    mut next_id: Int,
) -> Tracklet:
    """Fuse two fragments of the same object into one tracklet with a new
    identity, interpolating the interior gap (Sec. IV-C, Fusion)."""
    var merged = Tracklet(next_id, a.cls)
    next_id += 1
    for s in a.states:
        merged.insert_state(s)
    for s in b.states:
        merged.insert_state(s)
    merged.retag(merged.tid)
    interpolate_gaps(merged, frame_times)
    return merged^


def stwo(
    tracklets: List[Tracklet],
    frame_times: List[Float64],
    cfg: OfflinePolyConfig,
    mut next_id: Int,
) -> List[Tracklet]:
    """Single Tracker, Without Overlapping lifecycles: iteratively re-links
    temporally disjoint fragments via motion extrapolation + IoU matching."""
    var current = tracklets.copy()
    var num_frames = len(frame_times)
    for _ in range(cfg.stwo_max_passes):
        var merged_any = False
        for frame in range(num_frames):
            var n = len(current)
            if n < 2:
                break
            # node construction (Eq. 2)
            var nodes = List[Optional[BoxState]]()
            for tr in current:
                nodes.append(
                    predict_at(
                        tr, frame, frame_times[frame], cfg.max_pred_seconds
                    )
                )
            # cost matrix (Eq. 3): only disjoint-lifecycle, same-class pairs
            var cost = List[Float64]()
            for _ in range(n * n):
                cost.append(INVALID_COST)
            var any_edge = False
            for i in range(n):
                if not nodes[i]:
                    continue
                for j in range(i + 1, n):
                    if not nodes[j]:
                        continue
                    if current[i].cls != current[j].cls:
                        continue
                    if current[i].overlaps(current[j]):
                        continue
                    var c = 1.0 - iou_3d(nodes[i].value(), nodes[j].value())
                    if c < cfg.theta_blo:
                        cost[i * n + j] = c
                        cost[j * n + i] = c
                        any_edge = True
            if not any_edge:
                continue
            var pairs = solve_matching(
                n, cost, cfg.theta_blo, cfg.max_exact_component
            )
            if len(pairs) == 0:
                continue
            merged_any = True
            var consumed = List[Bool]()
            for _ in range(n):
                consumed.append(False)
            var updated = List[Tracklet]()
            for p in pairs:
                consumed[p.a] = True
                consumed[p.b] = True
                updated.append(
                    merge_pair(current[p.a], current[p.b], frame_times, next_id)
                )
            for i in range(n):
                if not consumed[i]:
                    updated.append(current[i].copy())
            current = updated^
        if not merged_any:
            break
    return current^
