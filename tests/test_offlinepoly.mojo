"""Unit tests for the OfflinePoly pipeline (synthetic scenes)."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from offlinepoly.config import OfflinePolyConfig
from offlinepoly.geometry import Vec2, giou_3d, iou_3d, iou_bev
from offlinepoly.io import parse_f64, parse_int
from offlinepoly.matching import Pair, solve_matching, stwo
from offlinepoly.motion import predict_at
from offlinepoly.multi import fuse_states, mtm
from offlinepoly.pipeline import preprocess, run_pipeline
from offlinepoly.refine import global_refine, local_refine
from offlinepoly.types import BoxState, Tracklet

comptime DT = 0.5  # seconds per frame in all synthetic scenes


def mk_state(
    frame: Int,
    x: Float64,
    y: Float64,
    vx: Float64 = 0.0,
    vy: Float64 = 0.0,
    theta: Float64 = 0.0,
    conf: Float64 = 0.9,
    w: Float64 = 2.0,
    l: Float64 = 4.0,
    h: Float64 = 1.5,
    cls: Int = 0,
    tid: Int = 0,
) -> BoxState:
    return BoxState(
        frame=frame, t=Float64(frame) * DT, x=x, y=y, z=0.75,
        w=w, l=l, h=h, vx=vx, vy=vy, theta=theta, conf=conf,
        cls=cls, tid=tid,
    )


def mk_timeline(num_frames: Int) -> List[Float64]:
    var out = List[Float64]()
    for f in range(num_frames):
        out.append(Float64(f) * DT)
    return out^


def test_iou_identical() raises:
    var a = mk_state(0, 0.0, 0.0)
    assert_almost_equal(iou_bev(a, a), 1.0, atol=Float64(1e-9))
    assert_almost_equal(iou_3d(a, a), 1.0, atol=Float64(1e-9))
    assert_almost_equal(giou_3d(a, a), 1.0, atol=Float64(1e-9))


def test_iou_half_shift() raises:
    # 4x2 boxes shifted by half the length: inter 2*2=4, union 8+8-4=12
    var a = mk_state(0, 0.0, 0.0)
    var b = mk_state(0, 2.0, 0.0)
    assert_almost_equal(iou_bev(a, b), 4.0 / 12.0, atol=Float64(1e-9))


def test_iou_disjoint_and_giou() raises:
    var a = mk_state(0, 0.0, 0.0)
    var b = mk_state(0, 100.0, 0.0)
    assert_almost_equal(iou_3d(a, b), 0.0, atol=Float64(1e-12))
    assert_true(giou_3d(a, b) < -0.5)  # far apart -> strongly negative


def test_iou_rotated() raises:
    # same square box rotated by 90 degrees: BEV IoU of a 2x2 square = 1
    var a = mk_state(0, 0.0, 0.0, w=2.0, l=2.0)
    var b = mk_state(0, 0.0, 0.0, w=2.0, l=2.0, theta=1.5707963267948966)
    assert_almost_equal(iou_bev(a, b), 1.0, atol=Float64(1e-6))


def test_parse_numbers() raises:
    assert_almost_equal(parse_f64("3.5"), 3.5, atol=Float64(1e-12))
    assert_almost_equal(parse_f64(" -1e-2 "), -0.01, atol=Float64(1e-12))
    assert_almost_equal(parse_f64("+2.25"), 2.25, atol=Float64(1e-12))
    assert_equal(parse_int("7"), 7)
    assert_equal(parse_int("-3"), -3)


def test_preprocess_filter() raises:
    var cfg = OfflinePolyConfig()  # age < 3 AND conf < 0.3 -> drop
    var ghost = Tracklet(1, 0)
    ghost.states.append(mk_state(0, 0.0, 0.0, conf=0.1, tid=1))
    var short_confident = Tracklet(2, 0)
    short_confident.states.append(mk_state(0, 5.0, 0.0, conf=0.9, tid=2))
    var long_weak = Tracklet(3, 0)
    for f in range(5):
        long_weak.states.append(mk_state(f, 10.0, 0.0, conf=0.1, tid=3))
    var src = List[Tracklet]()
    src.append(ghost^)
    src.append(short_confident^)
    src.append(long_weak^)
    var kept = preprocess(src, cfg)
    assert_equal(len(kept), 2)  # only the ghost is removed


def _set_edge(mut c: List[Float64], n: Int, i: Int, j: Int, v: Float64):
    c[i * n + j] = v
    c[j * n + i] = v


def test_solve_matching_prefers_cardinality() raises:
    # chain 0-1 (0.5), 1-2 (0.1), 2-3 (0.5): two pairs beat the single best
    comptime BIG = 1.0e18
    var n = 4
    var cost = List[Float64]()
    for _ in range(n * n):
        cost.append(BIG)
    _set_edge(cost, n, 0, 1, 0.5)
    _set_edge(cost, n, 1, 2, 0.1)
    _set_edge(cost, n, 2, 3, 0.5)
    var pairs = solve_matching(n, cost, 0.9, 10)
    assert_equal(len(pairs), 2)
    for p in pairs:
        assert_true((p.a == 0 and p.b == 1) or (p.a == 2 and p.b == 3))


def test_predict_in_gap() raises:
    var tr = Tracklet(1, 0)
    tr.states.append(mk_state(0, 0.0, 0.0, vx=2.0, tid=1))
    tr.states.append(mk_state(4, 4.0, 0.0, vx=2.0, tid=1))
    var pred = predict_at(tr, 2, 2 * DT, 1.0)
    assert_true(Bool(pred))
    assert_almost_equal(pred.value().x, 2.0, atol=Float64(1e-9))
    # horizon: frame 12 is 4s past the end -> discarded
    var none = predict_at(tr, 12, 12 * DT, 1.0)
    assert_true(not Bool(none))


def test_stwo_relinks_fragments() raises:
    # one object at constant velocity, observed frames 0-4 and 8-12
    var cfg = OfflinePolyConfig()
    var frame_times = mk_timeline(13)
    var early = Tracklet(1, 0)
    var late = Tracklet(2, 0)
    for f in range(5):
        early.states.append(
            mk_state(f, Float64(f) * 1.0, 0.0, vx=2.0, tid=1)
        )
    for f in range(8, 13):
        late.states.append(
            mk_state(f, Float64(f) * 1.0, 0.0, vx=2.0, tid=2)
        )
    var src = List[Tracklet]()
    src.append(early^)
    src.append(late^)
    var next_id = 10
    var linked = stwo(src, frame_times, cfg, next_id)
    assert_equal(len(linked), 1)
    assert_equal(linked[0].age(), 13)  # gap interpolated
    # interpolated midpoint should sit on the true trajectory
    var mid = linked[0].state_at(6)
    assert_true(Bool(mid))
    assert_almost_equal(mid.value().x, 6.0, atol=Float64(0.2))


def test_mtm_fuses_two_trackers() raises:
    var cfg = OfflinePolyConfig()
    var frame_times = mk_timeline(6)
    var next_id = 100
    var sources = List[List[Tracklet]]()
    var offsets: List[Float64] = [-0.2, 0.2]
    for offset in offsets:
        var tr = Tracklet(1, 0)
        for f in range(6):
            tr.states.append(
                mk_state(f, Float64(f) * 1.0, offset, vx=2.0, tid=1)
            )
        var src = List[Tracklet]()
        src.append(tr^)
        sources.append(src^)
    var fused = mtm(sources, frame_times, cfg, next_id)
    assert_equal(len(fused), 1)
    assert_equal(fused[0].age(), 6)
    # equal confidences -> fused y is the mean of the two offsets
    assert_almost_equal(fused[0].states[0].y, 0.0, atol=Float64(1e-9))


def test_fuse_states_weights_by_confidence() raises:
    var a = mk_state(0, 0.0, 0.0, conf=0.9)
    var b = mk_state(0, 1.0, 0.0, conf=0.3)
    var obs = List[BoxState]()
    obs.append(a)
    obs.append(b)
    var fused = fuse_states(obs, 7)
    assert_equal(fused.tid, 7)
    assert_almost_equal(fused.x, 0.25, atol=Float64(1e-9))  # 0.3/1.2 weight


def test_global_refine_size_vote() raises:
    var cfg = OfflinePolyConfig()
    var tr = Tracklet(1, 0)
    for f in range(10):
        var s = mk_state(f, Float64(f), 0.0, vx=2.0, tid=1)
        s.w = 2.0 + (0.5 if f == 0 else 0.0)  # one low-conf outlier
        s.conf = 0.1 if f == 0 else 0.9
        tr.states.append(s)
    var refined = global_refine(tr, cfg, List[Vec2]())
    # all frames share one size, close to the high-confidence 2.0
    for s in refined.states:
        assert_almost_equal(s.w, refined.states[0].w, atol=Float64(1e-12))
    assert_true(abs(refined.states[0].w - 2.0) < 0.05)
    # bottom face is preserved (h unchanged here -> z unchanged)
    assert_almost_equal(refined.states[0].z, 0.75, atol=Float64(1e-9))


def test_local_refine_smooths_noise() raises:
    var cfg = OfflinePolyConfig()
    var tr = Tracklet(1, 0)
    # straight line with alternating +/-0.3 noise in y
    for f in range(20):
        var noise = 0.3 if f % 2 == 0 else -0.3
        tr.states.append(
            mk_state(f, Float64(f) * 1.0, noise, vx=2.0, tid=1)
        )
    var refined = local_refine(tr, cfg)
    var raw_err = 0.0
    var ref_err = 0.0
    for f in range(20):
        raw_err += abs(tr.states[f].y)
        ref_err += abs(refined.states[f].y)
    assert_true(ref_err < raw_err * 0.5)


def test_pipeline_end_to_end() raises:
    var cfg = OfflinePolyConfig()
    var frame_times = mk_timeline(13)
    # tracker A: fragmented object + a ghost; tracker B: complete but offset
    var a1 = Tracklet(1, 0)
    var a2 = Tracklet(2, 0)
    for f in range(5):
        a1.states.append(mk_state(f, Float64(f), 0.0, vx=2.0, tid=1))
    for f in range(8, 13):
        a2.states.append(mk_state(f, Float64(f), 0.0, vx=2.0, tid=2))
    var ghost = Tracklet(3, 0)
    ghost.states.append(mk_state(4, 50.0, 50.0, conf=0.05, tid=3))
    var src_a = List[Tracklet]()
    src_a.append(a1^)
    src_a.append(a2^)
    src_a.append(ghost^)
    var b1 = Tracklet(1, 0)
    for f in range(13):
        b1.states.append(mk_state(f, Float64(f), 0.15, vx=2.0, tid=1))
    var src_b = List[Tracklet]()
    src_b.append(b1^)
    var sources = List[List[Tracklet]]()
    sources.append(src_a^)
    sources.append(src_b^)
    var out = run_pipeline(sources, frame_times, cfg, List[Vec2]())
    assert_equal(len(out), 1)  # ghost dropped, fragments + trackers fused
    assert_equal(out[0].age(), 13)
    # fused trajectory stays near the true line y ~ 0.075
    for s in out[0].states:
        assert_true(abs(s.y - 0.075) < 0.2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
