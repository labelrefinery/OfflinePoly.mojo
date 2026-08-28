"""Offline-Poly CLI.

Usage:
    mojo run -I src src/main.mojo OUT.csv IN1.csv [IN2.csv ...] [--ego EGO.csv]

Each IN CSV is the output of one upstream tracker
(track_id,cls,t,x,y,z,w,l,h,vx,vy,theta,conf). The optional EGO csv
(t,x,y rows) enables corner-aligned position correction in the global
refinement stage. OUT.csv receives the refined trajectories.
"""

from std.sys import argv

from offlinepoly.config import OfflinePolyConfig
from offlinepoly.geometry import Vec2
from offlinepoly.io import (
    assign_frames,
    parse_f64,
    read_tracker_csv,
    write_tracker_csv,
)
from offlinepoly.pipeline import run_pipeline
from offlinepoly.types import Tracklet


def _read_ego(path: String, frame_times: List[Float64]) raises -> List[Vec2]:
    """Map t,x,y rows to per-frame ego positions (nearest-neighbor fill)."""
    var ts = List[Float64]()
    var xy = List[Vec2]()
    var content = open(path, "r").read()
    for line_slice in content.split("\n"):
        var line = String(line_slice)
        var fields = line.split(",")
        if len(fields) != 3:
            continue
        try:
            var t = parse_f64(String(fields[0]))
            var x = parse_f64(String(fields[1]))
            var y = parse_f64(String(fields[2]))
            ts.append(t)
            xy.append(Vec2(x, y))
        except:
            continue  # header or malformed line
    var out = List[Vec2]()
    for ft in frame_times:
        var best = 0
        var best_d = 1.0e18
        for i in range(len(ts)):
            var d = abs(ts[i] - ft)
            if d < best_d:
                best_d = d
                best = i
        if len(xy) > 0:
            out.append(xy[best])
        else:
            out.append(Vec2(0.0, 0.0))
    if len(ts) == 0:
        return List[Vec2]()
    return out^


def main() raises:
    var args = argv()
    var out_path = String("")
    var in_paths = List[String]()
    var ego_path = String("")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--ego":
            if i + 1 >= len(args):
                raise Error("--ego requires a path")
            ego_path = String(args[i + 1])
            i += 2
            continue
        if out_path == "":
            out_path = a
        else:
            in_paths.append(a)
        i += 1
    if out_path == "" or len(in_paths) == 0:
        print(
            "usage: mojo run -I src src/main.mojo OUT.csv IN1.csv"
            " [IN2.csv ...] [--ego EGO.csv]"
        )
        return
    var sources = List[List[Tracklet]]()
    for p in in_paths:
        var trs = read_tracker_csv(p)
        var n_states = 0
        for tr in trs:
            n_states += tr.age()
        print("read", p, "->", len(trs), "tracklets,", n_states, "states")
        sources.append(trs^)
    var frame_times = assign_frames(sources)
    print("timeline:", len(frame_times), "frames")
    var ego = List[Vec2]()
    if ego_path != "":
        ego = _read_ego(ego_path, frame_times)
        print("ego trajectory loaded (corner-aligned refinement enabled)")
    var cfg = OfflinePolyConfig()
    var refined = run_pipeline(sources, frame_times, cfg, ego)
    var n_out_states = 0
    for tr in refined:
        n_out_states += tr.age()
    print(
        "refined ->", len(refined), "trajectories,", n_out_states, "states"
    )
    write_tracker_csv(out_path, refined)
    print("wrote", out_path)
