"""Single Tracker, With overlapping lifecycles (STW, Sec. IV-C).

Tracklets that were erroneously fused from several physical objects are
clustered by geometric similarity (gIoU adjacency, reusing the multi-tracker
machinery), split at the entangled frames into reliable pre-/post-segments,
the entangled per-frame components fused into age-1 tracklets, and the
resulting pieces re-linked with the STWO module.
"""

from .config import OfflinePolyConfig
from .geometry import giou_3d
from .matching import stwo
from .multi import cluster_tracklets, frame_table, fuse_states
from .types import BoxState, Tracklet


def stw(
    tracklets: List[Tracklet],
    frame_times: List[Float64],
    cfg: OfflinePolyConfig,
    mut next_id: Int,
) -> List[Tracklet]:
    var num_frames = len(frame_times)
    var clusters = cluster_tracklets(
        tracklets, num_frames, cfg.theta_stw, use_giou=True
    )
    var out = List[Tracklet]()
    for members in clusters:
        if len(members) == 1:
            out.append(tracklets[members[0]].copy())
            continue
        var tables = List[List[Int]]()
        for m in members:
            tables.append(frame_table(tracklets[m], num_frames))
        # entangled[k * num_frames + f]: member k is in a connected component
        # of size >= 2 at frame f
        var k = len(members)
        var entangled = List[Bool]()
        for _ in range(k * num_frames):
            entangled.append(False)
        var pieces = List[Tracklet]()
        for f in range(num_frames):
            var present = List[Int]()  # indices into `members`
            for i in range(k):
                if tables[i][f] >= 0:
                    present.append(i)
            var p = len(present)
            if p < 2:
                continue
            # frame-level adjacency among the present states
            var adj = List[Bool]()
            for _ in range(p * p):
                adj.append(False)
            for i in range(p):
                var si = tracklets[members[present[i]]].states[
                    tables[present[i]][f]
                ]
                for j in range(i + 1, p):
                    var sj = tracklets[members[present[j]]].states[
                        tables[present[j]][f]
                    ]
                    if 1.0 - giou_3d(si, sj) < cfg.theta_stw:
                        adj[i * p + j] = True
                        adj[j * p + i] = True
            # connected components; size >= 2 -> entangled, fuse into one node
            var comp = List[Int]()
            for _ in range(p):
                comp.append(-1)
            for start in range(p):
                if comp[start] >= 0:
                    continue
                var stack: List[Int] = [start]
                comp[start] = start
                var group = List[Int]()
                while len(stack) > 0:
                    var u = stack.pop()
                    group.append(u)
                    for v in range(p):
                        if comp[v] < 0 and adj[u * p + v]:
                            comp[v] = start
                            stack.append(v)
                if len(group) < 2:
                    continue
                var obs = List[BoxState]()
                for g in group:
                    entangled[present[g] * num_frames + f] = True
                    obs.append(
                        tracklets[members[present[g]]].states[
                            tables[present[g]][f]
                        ]
                    )
                var node = Tracklet(next_id, tracklets[members[0]].cls)
                next_id += 1
                node.insert_state(fuse_states(obs, node.tid))
                pieces.append(node^)
        # split each member at its entangled frames into reliable segments
        for i in range(k):
            var tr = tracklets[members[i]].copy()
            var segment = Tracklet(next_id, tr.cls)
            for s in tr.states:
                if entangled[i * num_frames + s.frame]:
                    if segment.age() > 0:
                        segment.retag(next_id)
                        next_id += 1
                        pieces.append(segment^)
                        segment = Tracklet(next_id, tr.cls)
                else:
                    segment.insert_state(s)
            if segment.age() > 0:
                segment.retag(next_id)
                next_id += 1
                pieces.append(segment^)
        # reorganize: bidirectional-motion re-linking of the pieces
        for tr in stwo(pieces, frame_times, cfg, next_id):
            out.append(tr.copy())
    return out^
