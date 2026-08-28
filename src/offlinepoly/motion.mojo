"""Explicit motion models and the forward-backward prediction of Eq. 2.

The default model is Constant Velocity (CV): positions advance by the state's
own velocity; size, heading, and velocity are carried over unchanged.
"""

from .types import BoxState, Tracklet


def predict_cv(b: BoxState, target_frame: Int, target_time: Float64) -> BoxState:
    """Propagate a state to `target_time` under constant velocity."""
    var dt = target_time - b.t
    var out = b
    out.frame = target_frame
    out.t = target_time
    out.x = b.x + b.vx * dt
    out.y = b.y + b.vy * dt
    return out


def _mean_state(a: BoxState, b: BoxState) -> BoxState:
    var out = a
    out.x = (a.x + b.x) * 0.5
    out.y = (a.y + b.y) * 0.5
    out.z = (a.z + b.z) * 0.5
    out.vx = (a.vx + b.vx) * 0.5
    out.vy = (a.vy + b.vy) * 0.5
    out.conf = (a.conf + b.conf) * 0.5
    # sizes/heading come from `a`; heading averaging is intentionally avoided
    return out


def predict_at(
    tr: Tracklet,
    frame: Int,
    target_time: Float64,
    max_pred_seconds: Float64,
) -> Optional[BoxState]:
    """Eq. 2: forward/backward extrapolation outside the lifecycle, averaged
    bidirectional prediction inside gaps. Predictions whose horizon exceeds
    `max_pred_seconds` are discarded (returns None)."""
    if len(tr.states) == 0:
        return None
    var observed = tr.state_at(frame)
    if observed:
        return observed
    var first = tr.states[0]
    var last = tr.states[len(tr.states) - 1]
    if frame > last.frame:
        if target_time - last.t > max_pred_seconds:
            return None
        return predict_cv(last, frame, target_time)
    if frame < first.frame:
        if first.t - target_time > max_pred_seconds:
            return None
        return predict_cv(first, frame, target_time)
    # inside a gap: nearest observed states on both sides
    var before = first
    var after = last
    for s in tr.states:
        if s.frame < frame:
            before = s
        elif s.frame > frame:
            after = s
            break
    var fwd_ok = target_time - before.t <= max_pred_seconds
    var bwd_ok = after.t - target_time <= max_pred_seconds
    if fwd_ok and bwd_ok:
        return _mean_state(
            predict_cv(before, frame, target_time),
            predict_cv(after, frame, target_time),
        )
    if fwd_ok:
        return predict_cv(before, frame, target_time)
    if bwd_ok:
        return predict_cv(after, frame, target_time)
    return None


def interpolate_gaps(mut tr: Tracklet, frame_times: List[Float64]):
    """Fill every missing interior frame of a merged tracklet with the
    averaged bidirectional prediction (no horizon limit for interior gaps)."""
    if len(tr.states) < 2:
        return
    var filled = List[BoxState]()
    for i in range(len(tr.states) - 1):
        var a = tr.states[i]
        var b = tr.states[i + 1]
        filled.append(a)
        for f in range(a.frame + 1, b.frame):
            var t = frame_times[f]
            var s = _mean_state(predict_cv(a, f, t), predict_cv(b, f, t))
            s.tid = tr.tid
            filled.append(s)
    filled.append(tr.states[len(tr.states) - 1])
    tr.states = filled^
