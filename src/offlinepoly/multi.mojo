"""Hierarchical matching and fusion across multiple trackers (Sec. IV-D).

Scene-level connectivity (Eqs. 6-8) links two tracklets when their observed
states overlap geometrically in ANY frame (max-pooling over time); DFS over
that adjacency (Eq. 9) yields one-to-many clusters, which are fused by
confidence-weighted averaging of all valid per-frame observations.
"""

from std.math import min, max

from .config import OfflinePolyConfig
from .geometry import giou_3d, iou_3d, wrap_angle
from .types import BoxState, Tracklet


def frame_table(tr: Tracklet, num_frames: Int) -> List[Int]:
    """Per-frame index into tr.states (-1 where unobserved)."""
    var table = List[Int]()
    for _ in range(num_frames):
        table.append(-1)
    for i in range(len(tr.states)):
        var f = tr.states[i].frame
        if f >= 0 and f < num_frames:
            table[f] = i
    return table^


def pairwise_connected(
    a: Tracklet,
    ta: List[Int],
    b: Tracklet,
    tb: List[Int],
    num_frames: Int,
    threshold: Float64,
    use_giou: Bool,
) -> Bool:
    """Scene-level adjacency: cost < threshold in at least one frame."""
    if a.cls != b.cls:
        return False
    var lo = max(a.start_frame(), b.start_frame())
    var hi = min(a.end_frame(), b.end_frame())
    for f in range(lo, hi + 1):
        if ta[f] < 0 or tb[f] < 0:
            continue
        var cost: Float64
        if use_giou:
            cost = 1.0 - giou_3d(a.states[ta[f]], b.states[tb[f]])
        else:
            cost = 1.0 - iou_3d(a.states[ta[f]], b.states[tb[f]])
        if cost < threshold:
            return True
    return False


def cluster_tracklets(
    tracklets: List[Tracklet],
    num_frames: Int,
    threshold: Float64,
    use_giou: Bool,
) -> List[List[Int]]:
    """DFS connected components over the scene-level adjacency (Eq. 9)."""
    var n = len(tracklets)
    var tables = List[List[Int]]()
    for tr in tracklets:
        tables.append(frame_table(tr, num_frames))
    var adj = List[Bool]()
    for _ in range(n * n):
        adj.append(False)
    for i in range(n):
        for j in range(i + 1, n):
            if pairwise_connected(
                tracklets[i], tables[i], tracklets[j], tables[j],
                num_frames, threshold, use_giou,
            ):
                adj[i * n + j] = True
                adj[j * n + i] = True
    var comp = List[Int]()
    for _ in range(n):
        comp.append(-1)
    var clusters = List[List[Int]]()
    for start in range(n):
        if comp[start] >= 0:
            continue
        var cid = len(clusters)
        var stack: List[Int] = [start]
        comp[start] = cid
        var members = List[Int]()
        while len(stack) > 0:
            var u = stack.pop()
            members.append(u)
            for v in range(n):
                if comp[v] < 0 and adj[u * n + v]:
                    comp[v] = cid
                    stack.append(v)
        clusters.append(members^)
    return clusters^


def fuse_states(states: List[BoxState], tid: Int) -> BoxState:
    """Attribute-agnostic fusion: confidence-weighted average of all valid
    observations; heading uses a weighted circular mean around the first
    observation's heading."""
    var out = states[0]
    if len(states) == 1:
        out.tid = tid
        return out
    var total_w = 0.0
    for s in states:
        total_w += max(s.conf, 1e-6)
    var x = 0.0
    var y = 0.0
    var z = 0.0
    var w = 0.0
    var l = 0.0
    var h = 0.0
    var vx = 0.0
    var vy = 0.0
    var conf = 0.0
    var theta_off = 0.0
    var ref_theta = states[0].theta
    for s in states:
        var wt = max(s.conf, 1e-6) / total_w
        x += wt * s.x
        y += wt * s.y
        z += wt * s.z
        w += wt * s.w
        l += wt * s.l
        h += wt * s.h
        vx += wt * s.vx
        vy += wt * s.vy
        theta_off += wt * wrap_angle(s.theta - ref_theta)
        conf += s.conf
    out.x = x
    out.y = y
    out.z = z
    out.w = w
    out.l = l
    out.h = h
    out.vx = vx
    out.vy = vy
    out.theta = wrap_angle(ref_theta + theta_off)
    out.conf = conf / Float64(len(states))
    out.tid = tid
    return out


def mtm(
    sources: List[List[Tracklet]],
    frame_times: List[Float64],
    cfg: OfflinePolyConfig,
    mut next_id: Int,
) -> List[Tracklet]:
    """Multiple Trackers Matching and fusion: consolidate the outputs of all
    upstream trackers into a single set of trajectories."""
    var num_frames = len(frame_times)
    var flat = List[Tracklet]()
    for src in sources:
        for tr in src:
            flat.append(tr.copy())
    if len(flat) == 0:
        return flat^
    var clusters = cluster_tracklets(
        flat, num_frames, cfg.theta_multi, use_giou=False
    )
    var tables = List[List[Int]]()
    for tr in flat:
        tables.append(frame_table(tr, num_frames))
    var out = List[Tracklet]()
    for members in clusters:
        var fused = Tracklet(next_id, flat[members[0]].cls)
        next_id += 1
        for f in range(num_frames):
            var obs = List[BoxState]()
            for m in members:
                var idx = tables[m][f]
                if idx >= 0:
                    obs.append(flat[m].states[idx])
            if len(obs) > 0:
                fused.insert_state(fuse_states(obs, fused.tid))
        out.append(fused^)
    return out^
