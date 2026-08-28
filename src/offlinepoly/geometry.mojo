"""Rotated-box overlap metrics: BEV IoU, 3D IoU, and generalized 3D IoU.

BEV boxes are convex quadrilaterals; intersection uses Sutherland-Hodgman
clipping and the enclosing hull for gIoU uses a monotone-chain convex hull.
"""

from std.math import cos, sin, pi, min, max

from .types import BoxState

comptime Vec2 = SIMD[DType.float64, 2]


def wrap_angle(a: Float64) -> Float64:
    """Wrap an angle to [-pi, pi]."""
    var x = a
    while x > pi:
        x -= 2.0 * pi
    while x < -pi:
        x += 2.0 * pi
    return x


def _cross(o: Vec2, a: Vec2, b: Vec2) -> Float64:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def bev_corners(b: BoxState) -> List[Vec2]:
    """The 4 BEV corners, counter-clockwise; `l` runs along the heading."""
    var c = cos(b.theta)
    var s = sin(b.theta)
    var hl = b.l * 0.5
    var hw = b.w * 0.5
    var out = List[Vec2]()
    # object-frame corners (+l,+w), (-l,+w), (-l,-w), (+l,-w)
    var dxs: List[Float64] = [hl, -hl, -hl, hl]
    var dys: List[Float64] = [hw, hw, -hw, -hw]
    for i in range(4):
        var dx = dxs[i]
        var dy = dys[i]
        out.append(Vec2(b.x + c * dx - s * dy, b.y + s * dx + c * dy))
    return out^


def polygon_area(poly: List[Vec2]) -> Float64:
    """Shoelace area (positive for counter-clockwise winding)."""
    var n = len(poly)
    if n < 3:
        return 0.0
    var acc = 0.0
    for i in range(n):
        var j = (i + 1) % n
        acc += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(acc) * 0.5


def clip_polygon(subject: List[Vec2], clip: List[Vec2]) -> List[Vec2]:
    """Sutherland-Hodgman: clip `subject` by convex CCW polygon `clip`."""
    var output = subject.copy()
    var m = len(clip)
    for i in range(m):
        if len(output) == 0:
            break
        var a = clip[i]
        var b = clip[(i + 1) % m]
        var input = output.copy()
        output = List[Vec2]()
        var n = len(input)
        for j in range(n):
            var p = input[j]
            var q = input[(j + 1) % n]
            var p_in = _cross(a, b, p) >= -1e-12
            var q_in = _cross(a, b, q) >= -1e-12
            if p_in:
                output.append(p)
                if not q_in:
                    output.append(_intersect(a, b, p, q))
            elif q_in:
                output.append(_intersect(a, b, p, q))
    return output^


def _intersect(a: Vec2, b: Vec2, p: Vec2, q: Vec2) -> Vec2:
    """Intersection of line (a,b) with segment (p,q)."""
    var d1 = b - a
    var d2 = q - p
    var denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-15:
        return p
    var w = p - a
    var s = (w[0] * d2[1] - w[1] * d2[0]) / denom
    return a + d1 * s


def convex_hull(points: List[Vec2]) -> List[Vec2]:
    """Monotone-chain convex hull (CCW). Fine for the small point sets here."""
    var pts = points.copy()
    var n = len(pts)
    if n < 3:
        return pts^
    # insertion sort by (x, y)
    for i in range(1, n):
        var key = pts[i]
        var j = i - 1
        while j >= 0 and (
            pts[j][0] > key[0] or (pts[j][0] == key[0] and pts[j][1] > key[1])
        ):
            pts[j + 1] = pts[j]
            j -= 1
        pts[j + 1] = key
    var hull = List[Vec2]()
    for i in range(n):  # lower
        while len(hull) >= 2 and _cross(
            hull[len(hull) - 2], hull[len(hull) - 1], pts[i]
        ) <= 0:
            _ = hull.pop()
        hull.append(pts[i])
    var lower_len = len(hull) + 1
    var i2 = n - 2
    while i2 >= 0:  # upper
        while len(hull) >= lower_len and _cross(
            hull[len(hull) - 2], hull[len(hull) - 1], pts[i2]
        ) <= 0:
            _ = hull.pop()
        hull.append(pts[i2])
        i2 -= 1
    _ = hull.pop()  # last point equals the first
    return hull^


def bev_intersection_area(a: BoxState, b: BoxState) -> Float64:
    return polygon_area(clip_polygon(bev_corners(a), bev_corners(b)))


def iou_bev(a: BoxState, b: BoxState) -> Float64:
    var inter = bev_intersection_area(a, b)
    var union = a.w * a.l + b.w * b.l - inter
    if union <= 1e-12:
        return 0.0
    return inter / union


def _z_overlap(a: BoxState, b: BoxState) -> Float64:
    var lo = max(a.z - a.h * 0.5, b.z - b.h * 0.5)
    var hi = min(a.z + a.h * 0.5, b.z + b.h * 0.5)
    return max(0.0, hi - lo)


def iou_3d(a: BoxState, b: BoxState) -> Float64:
    var inter = bev_intersection_area(a, b) * _z_overlap(a, b)
    var union = a.w * a.l * a.h + b.w * b.l * b.h - inter
    if union <= 1e-12:
        return 0.0
    return inter / union


def giou_3d(a: BoxState, b: BoxState) -> Float64:
    """Generalized 3D IoU in [-1, 1]; enclosure is hull area x z-extent."""
    var inter = bev_intersection_area(a, b) * _z_overlap(a, b)
    var vol_a = a.w * a.l * a.h
    var vol_b = b.w * b.l * b.h
    var union = vol_a + vol_b - inter
    var pts = bev_corners(a)
    for p in bev_corners(b):
        pts.append(p)
    var hull_area = polygon_area(convex_hull(pts))
    var z_lo = min(a.z - a.h * 0.5, b.z - b.h * 0.5)
    var z_hi = max(a.z + a.h * 0.5, b.z + b.h * 0.5)
    var enclosure = hull_area * (z_hi - z_lo)
    if union <= 1e-12 or enclosure <= 1e-12:
        return 0.0
    return inter / union - (enclosure - union) / enclosure
