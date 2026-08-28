"""The four-stage Offline-Poly pipeline (Fig. 2):

pre-processing -> single-tracker matching/fusion (STWO + STW, per source)
-> multi-tracker matching/fusion -> multi-perspective refinement.
"""

from .config import OfflinePolyConfig
from .geometry import Vec2
from .matching import stwo
from .multi import mtm
from .overlap import stw
from .refine import global_refine, local_refine
from .types import Tracklet


def preprocess(
    source: List[Tracklet], cfg: OfflinePolyConfig
) -> List[Tracklet]:
    """Tracklet-level filter (Sec. IV-B): drop trajectories whose age AND
    mean confidence are both below their thresholds."""
    var out = List[Tracklet]()
    for tr in source:
        if tr.age() < cfg.age_threshold and tr.mean_conf() < cfg.score_threshold:
            continue
        out.append(tr.copy())
    return out^


def run_pipeline(
    sources: List[List[Tracklet]],
    frame_times: List[Float64],
    cfg: OfflinePolyConfig,
    ego_xy: List[Vec2],
) -> List[Tracklet]:
    """Refine one or more upstream tracking results into final trajectories.

    `sources` must already carry frame indices (see `io.assign_frames`);
    `ego_xy` is the per-frame ego BEV position, or empty when unknown.
    """
    var next_id = 1
    for src in sources:
        for tr in src:
            if tr.tid >= next_id:
                next_id = tr.tid + 1
    var singles = List[List[Tracklet]]()
    for src in sources:
        var pre = preprocess(src, cfg)
        var linked = stwo(pre, frame_times, cfg, next_id)
        var untangled = stw(linked, frame_times, cfg, next_id)
        singles.append(untangled^)
    var fused = mtm(singles, frame_times, cfg, next_id)
    var out = List[Tracklet]()
    for tr in fused:
        var glo = global_refine(tr, cfg, ego_xy)
        out.append(local_refine(glo, cfg))
    return out^
