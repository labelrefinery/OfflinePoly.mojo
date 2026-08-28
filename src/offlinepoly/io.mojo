"""CSV I/O for the TBT contract.

One CSV per upstream tracker, one row per state:

    track_id,cls,t,x,y,z,w,l,h,vx,vy,theta,conf

Header lines and blank lines are skipped. `frame` indices are assigned later
by `assign_frames` from the union of timestamps across all trackers.
"""

from .types import BoxState, Tracklet


def _is_ws(c: UInt8) -> Bool:
    return c == 32 or c == 9 or c == 13 or c == 10


def parse_f64(s: String) raises -> Float64:
    """Minimal float parser: sign, digits, fraction, exponent."""
    var bytes = List[UInt8]()
    for b in s.as_bytes():
        bytes.append(b)
    var n = len(bytes)
    var i = 0
    while i < n and _is_ws(bytes[i]):
        i += 1
    while n > i and _is_ws(bytes[n - 1]):
        n -= 1
    if i >= n:
        raise Error("empty numeric field")
    var sign = 1.0
    if bytes[i] == 45:  # '-'
        sign = -1.0
        i += 1
    elif bytes[i] == 43:  # '+'
        i += 1
    var mantissa = 0.0
    var seen_digit = False
    while i < n and bytes[i] >= 48 and bytes[i] <= 57:
        mantissa = mantissa * 10.0 + Float64(Int(bytes[i]) - 48)
        seen_digit = True
        i += 1
    var frac_scale = 1.0
    if i < n and bytes[i] == 46:  # '.'
        i += 1
        while i < n and bytes[i] >= 48 and bytes[i] <= 57:
            mantissa = mantissa * 10.0 + Float64(Int(bytes[i]) - 48)
            frac_scale *= 10.0
            seen_digit = True
            i += 1
    if not seen_digit:
        raise Error("invalid numeric field: " + s)
    var value = sign * mantissa / frac_scale
    if i < n and (bytes[i] == 101 or bytes[i] == 69):  # 'e' / 'E'
        i += 1
        var esign = 1
        if i < n and bytes[i] == 45:
            esign = -1
            i += 1
        elif i < n and bytes[i] == 43:
            i += 1
        var e = 0
        var seen_exp = False
        while i < n and bytes[i] >= 48 and bytes[i] <= 57:
            e = e * 10 + Int(bytes[i]) - 48
            seen_exp = True
            i += 1
        if not seen_exp:
            raise Error("invalid exponent: " + s)
        var scale = 1.0
        for _ in range(e):
            scale *= 10.0
        if esign > 0:
            value *= scale
        else:
            value /= scale
    if i != n:
        raise Error("trailing characters in numeric field: " + s)
    return value


def parse_int(s: String) raises -> Int:
    var v = parse_f64(s)
    if v >= 0:
        return Int(v + 0.5)
    return -Int(0.5 - v)


def _is_header(line: String) -> Bool:
    for b in line.as_bytes():
        if _is_ws(b) or b == 44:  # skip whitespace and commas
            continue
        # a data line must start with a digit, sign, or dot
        return not (
            (b >= 48 and b <= 57) or b == 45 or b == 43 or b == 46
        )
    return True  # blank line


def read_tracker_csv(path: String) raises -> List[Tracklet]:
    var content = open(path, "r").read()
    var by_id = Dict[Int, Int]()
    var out = List[Tracklet]()
    for line_slice in content.split("\n"):
        var line = String(line_slice)
        if _is_header(line):
            continue
        var fields = line.split(",")
        if len(fields) != 13:
            raise Error(
                "expected 13 fields, got "
                + String(len(fields))
                + " in line: "
                + String(line)
            )
        var tid = parse_int(String(fields[0]))
        var cls = parse_int(String(fields[1]))
        var s = BoxState(
            frame=-1,
            t=parse_f64(String(fields[2])),
            x=parse_f64(String(fields[3])),
            y=parse_f64(String(fields[4])),
            z=parse_f64(String(fields[5])),
            w=parse_f64(String(fields[6])),
            l=parse_f64(String(fields[7])),
            h=parse_f64(String(fields[8])),
            vx=parse_f64(String(fields[9])),
            vy=parse_f64(String(fields[10])),
            theta=parse_f64(String(fields[11])),
            conf=parse_f64(String(fields[12])),
            cls=cls,
            tid=tid,
        )
        if tid not in by_id:
            by_id[tid] = len(out)
            out.append(Tracklet(tid, cls))
        out[by_id[tid]].states.append(s)
    # sort each tracklet's states by time
    for ref tr in out:
        for i in range(1, len(tr.states)):
            var key = tr.states[i]
            var j = i - 1
            while j >= 0 and tr.states[j].t > key.t:
                tr.states[j + 1] = tr.states[j]
                j -= 1
            tr.states[j + 1] = key
    return out^


def assign_frames(
    mut sources: List[List[Tracklet]], tolerance: Float64 = 1e-6
) -> List[Float64]:
    """Build the global frame timeline from the union of timestamps across
    all trackers and stamp every state's `frame` index."""
    var times = List[Float64]()
    for src in sources:
        for tr in src:
            for s in tr.states:
                times.append(s.t)
    for i in range(1, len(times)):  # insertion sort
        var key = times[i]
        var j = i - 1
        while j >= 0 and times[j] > key:
            times[j + 1] = times[j]
            j -= 1
        times[j + 1] = key
    var frame_times = List[Float64]()
    for t in times:
        if len(frame_times) == 0 or t - frame_times[len(frame_times) - 1] > tolerance:
            frame_times.append(t)
    for ref src in sources:
        for ref tr in src:
            for ref s in tr.states:
                # binary search for the nearest frame time
                var lo = 0
                var hi = len(frame_times) - 1
                while lo < hi:
                    var mid = (lo + hi) // 2
                    if frame_times[mid] < s.t - tolerance:
                        lo = mid + 1
                    else:
                        hi = mid
                s.frame = lo
    return frame_times^


def write_tracker_csv(path: String, tracklets: List[Tracklet]) raises:
    var f = open(path, "w")
    f.write("track_id,cls,t,x,y,z,w,l,h,vx,vy,theta,conf\n")
    for tr in tracklets:
        for s in tr.states:
            var row = String()
            row.write(
                tr.tid, ",", tr.cls, ",", s.t, ",",
                s.x, ",", s.y, ",", s.z, ",",
                s.w, ",", s.l, ",", s.h, ",",
                s.vx, ",", s.vy, ",", s.theta, ",",
                s.conf, "\n",
            )
            f.write(row)
    f.close()
