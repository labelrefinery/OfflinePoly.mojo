"""Core data types for the Tracking-By-Tracking (TBT) paradigm.

A tracklet is a temporally ordered sequence of box states produced by an
arbitrary upstream tracker (Offline-Poly, Sec. III). All states live in the
global coordinate frame.
"""


@fieldwise_init
struct BoxState(ImplicitlyCopyable, Movable, Writable):
    """One per-frame observation: (x,y,z,w,l,h,vx,vy,theta,conf,t,cls,id).

    `l` extends along the heading `theta`, `w` perpendicular to it, `h`
    vertically; `(x, y, z)` is the box center.
    """

    var frame: Int
    var t: Float64
    var x: Float64
    var y: Float64
    var z: Float64
    var w: Float64
    var l: Float64
    var h: Float64
    var vx: Float64
    var vy: Float64
    var theta: Float64
    var conf: Float64
    var cls: Int
    var tid: Int


struct Tracklet(Copyable, Movable):
    """A tracklet: states sorted by frame, at most one state per frame."""

    var tid: Int
    var cls: Int
    var states: List[BoxState]

    def __init__(out self, tid: Int, cls: Int):
        self.tid = tid
        self.cls = cls
        self.states = List[BoxState]()

    def age(self) -> Int:
        return len(self.states)

    def start_frame(self) -> Int:
        return self.states[0].frame

    def end_frame(self) -> Int:
        return self.states[len(self.states) - 1].frame

    def mean_conf(self) -> Float64:
        if len(self.states) == 0:
            return 0.0
        var total = 0.0
        for s in self.states:
            total += s.conf
        return total / Float64(len(self.states))

    def overlaps(self, other: Tracklet) -> Bool:
        """Whether the two lifecycles share any part of their frame ranges."""
        if len(self.states) == 0 or len(other.states) == 0:
            return False
        return not (
            self.end_frame() < other.start_frame()
            or other.end_frame() < self.start_frame()
        )

    def state_index(self, frame: Int) -> Int:
        """Index of the state observed at `frame`, or -1."""
        var lo = 0
        var hi = len(self.states) - 1
        while lo <= hi:
            var mid = (lo + hi) // 2
            var f = self.states[mid].frame
            if f == frame:
                return mid
            if f < frame:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    def state_at(self, frame: Int) -> Optional[BoxState]:
        var idx = self.state_index(frame)
        if idx < 0:
            return None
        return self.states[idx]

    def insert_state(mut self, state: BoxState):
        """Insert keeping frame order; replaces an existing state at the frame."""
        var i = 0
        while i < len(self.states) and self.states[i].frame < state.frame:
            i += 1
        if i < len(self.states) and self.states[i].frame == state.frame:
            self.states[i] = state
        else:
            self.states.insert(i, state)

    def retag(mut self, tid: Int):
        """Assign a new identity to the tracklet and all of its states."""
        self.tid = tid
        for ref s in self.states:
            s.tid = tid
