"""Multi-perspective trajectory refinement (Sec. IV-E).

Global perspective: rigid objects have a constant size, so the Top-K most
confident observations vote (softmax-weighted) on one 3D size, and each
frame's center is corrected with the corner-alignment strategy (the corner
nearest the ego is the reliable L-shape reference) when an ego trajectory is
available; the bottom face is kept fixed vertically.

Local perspective: a sliding-window least-squares fit of each state's
motion-related attributes (center, velocity, heading) against its past AND
future neighbors under the constant-velocity model. The paper solves this
with Levenberg-Marquardt; under CV the problem is linear, so a damped
Gauss-Newton (identical update) converges in two iterations.
"""

from std.math import cos, exp, min, max, sin

from .config import OfflinePolyConfig
from .geometry import Vec2, bev_corners, wrap_angle
from .types import BoxState, Tracklet


def _top_k_indices(tr: Tracklet, k: Int) -> List[Int]:
    """Indices of the k most confident states (selection by confidence)."""
    var n = len(tr.states)
    var picked = List[Bool]()
    for _ in range(n):
        picked.append(False)
    var out = List[Int]()
    for _ in range(min(k, n)):
        var best = -1
        var best_conf = -1.0
        for i in range(n):
            if not picked[i] and tr.states[i].conf > best_conf:
                best_conf = tr.states[i].conf
                best = i
        picked[best] = True
        out.append(best)
    return out^


def global_refine(
    tr: Tracklet, cfg: OfflinePolyConfig, ego_xy: List[Vec2]
) -> Tracklet:
    """Size correction (TopK + softmax) and corner-aligned center correction.

    `ego_xy` holds the ego BEV position per frame; pass an empty list when the
    ego trajectory is unknown, which keeps centers BEV-unchanged
    (center-aligned) and only anchors the bottom face vertically.
    """
    var out = tr.copy()
    if len(out.states) == 0 or not cfg.is_rigid(out.cls):
        return out^
    var top = _top_k_indices(out, cfg.top_k)
    var max_conf = -1.0e18
    for i in top:
        max_conf = max(max_conf, out.states[i].conf)
    var total = 0.0
    var new_w = 0.0
    var new_l = 0.0
    var new_h = 0.0
    for i in top:
        var wt = exp(out.states[i].conf - max_conf)
        total += wt
        new_w += wt * out.states[i].w
        new_l += wt * out.states[i].l
        new_h += wt * out.states[i].h
    new_w /= total
    new_l /= total
    new_h /= total
    var have_ego = len(ego_xy) > 0
    # object-frame corner offsets, matching bev_corners() ordering
    var sxs: List[Float64] = [1.0, -1.0, -1.0, 1.0]
    var sys: List[Float64] = [1.0, 1.0, -1.0, -1.0]
    for ref s in out.states:
        if have_ego and s.frame >= 0 and s.frame < len(ego_xy):
            var corners = bev_corners(s)
            var ego = ego_xy[s.frame]
            var nearest = 0
            var best_d = 1.0e18
            for c in range(4):
                var d = (corners[c] - ego) * (corners[c] - ego)
                var dist = d[0] + d[1]
                if dist < best_d:
                    best_d = dist
                    nearest = c
            # keep the nearest corner fixed while the size changes
            var cth = cos(s.theta)
            var sth = sin(s.theta)
            var dx = sxs[nearest] * new_l * 0.5
            var dy = sys[nearest] * new_w * 0.5
            s.x = corners[nearest][0] - (cth * dx - sth * dy)
            s.y = corners[nearest][1] - (sth * dx + cth * dy)
        s.z = s.z - s.h * 0.5 + new_h * 0.5  # bottom face fixed
        s.w = new_w
        s.l = new_l
        s.h = new_h
    return out^


def _solve6(mut a: List[Float64], mut g: List[Float64]) -> List[Float64]:
    """Solve the 6x6 system a * x = g in place (partial pivoting)."""
    comptime N = 6
    for col in range(N):
        var piv = col
        for r in range(col + 1, N):
            if abs(a[r * N + col]) > abs(a[piv * N + col]):
                piv = r
        if piv != col:
            for c in range(N):
                var tmp = a[col * N + c]
                a[col * N + c] = a[piv * N + c]
                a[piv * N + c] = tmp
            var tg = g[col]
            g[col] = g[piv]
            g[piv] = tg
        var d = a[col * N + col]
        if abs(d) < 1e-12:
            continue
        for r in range(N):
            if r == col:
                continue
            var factor = a[r * N + col] / d
            for c in range(col, N):
                a[r * N + c] -= factor * a[col * N + c]
            g[r] -= factor * g[col]
    var x = List[Float64]()
    for i in range(N):
        var d = a[i * N + i]
        if abs(d) < 1e-12:
            x.append(0.0)
        else:
            x.append(g[i] / d)
    return x^


def local_refine(tr: Tracklet, cfg: OfflinePolyConfig) -> Tracklet:
    """Sliding-window optimization (Eq. 10) of (x, y, z, vx, vy, theta)."""
    var out = tr.copy()
    var n = len(out.states)
    if n < 2:
        return out^
    var half = max(cfg.window // 2, 1)
    var refined = List[BoxState]()
    for i in range(n):
        var center = out.states[i]
        # collect neighbors (targets stay the pre-refinement observations)
        var window = List[BoxState]()
        for j in range(n):
            var s = out.states[j]
            if abs(s.frame - center.frame) > half:
                continue
            if abs(s.t - center.t) > cfg.max_window_seconds * 0.5:
                continue
            window.append(s)
        if len(window) < 2:
            refined.append(center)
            continue
        # params p = (x, y, z, vx, vy, theta), Gauss-Newton with tiny damping
        var p: List[Float64] = [
            center.x, center.y, center.z, center.vx, center.vy, center.theta
        ]
        for _ in range(2):
            var a = List[Float64]()
            for _ in range(36):
                a.append(0.0)
            var g = List[Float64]()
            for _ in range(6):
                g.append(0.0)
            for s in window:
                var dts = s.t - center.t
                # residual rows: (value, gradient indices/values)
                # r0 = p0 + p3*dts - s.x       d/dp0=1, d/dp3=dts
                # r1 = p1 + p4*dts - s.y       d/dp1=1, d/dp4=dts
                # r2 = p2 - s.z                d/dp2=1
                # r3 = p3 - s.vx               d/dp3=1
                # r4 = p4 - s.vy               d/dp4=1
                # r5 = wrap(p5 - s.theta)      d/dp5=1
                var r0 = p[0] + p[3] * dts - s.x
                a[0 * 6 + 0] += 1.0
                a[0 * 6 + 3] += dts
                a[3 * 6 + 0] += dts
                a[3 * 6 + 3] += dts * dts
                g[0] -= r0
                g[3] -= r0 * dts
                var r1 = p[1] + p[4] * dts - s.y
                a[1 * 6 + 1] += 1.0
                a[1 * 6 + 4] += dts
                a[4 * 6 + 1] += dts
                a[4 * 6 + 4] += dts * dts
                g[1] -= r1
                g[4] -= r1 * dts
                var r2 = p[2] - s.z
                a[2 * 6 + 2] += 1.0
                g[2] -= r2
                var r3 = p[3] - s.vx
                a[3 * 6 + 3] += 1.0
                g[3] -= r3
                var r4 = p[4] - s.vy
                a[4 * 6 + 4] += 1.0
                g[4] -= r4
                var r5 = wrap_angle(p[5] - s.theta)
                a[5 * 6 + 5] += 1.0
                g[5] -= r5
            for d in range(6):
                a[d * 6 + d] += 1e-9  # damping
            var delta = _solve6(a, g)
            for d in range(6):
                p[d] += delta[d]
        var s_out = center
        s_out.x = p[0]
        s_out.y = p[1]
        s_out.z = p[2]
        s_out.vx = p[3]
        s_out.vy = p[4]
        s_out.theta = wrap_angle(p[5])
        refined.append(s_out)
    out.states = refined^
    return out^
