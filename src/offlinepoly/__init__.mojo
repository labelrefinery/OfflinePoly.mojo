"""OfflinePoly.mojo — a pure-Mojo implementation of Offline-Poly
(arXiv:2602.13772): learning-free offline 3D multi-object tracking under the
Tracking-By-Tracking paradigm, for 4D auto-labeling pipelines."""

from .config import OfflinePolyConfig
from .geometry import Vec2, giou_3d, iou_3d, iou_bev, wrap_angle
from .io import assign_frames, read_tracker_csv, write_tracker_csv
from .matching import solve_matching, stwo
from .motion import predict_at, predict_cv
from .multi import cluster_tracklets, fuse_states, mtm
from .overlap import stw
from .pipeline import preprocess, run_pipeline
from .refine import global_refine, local_refine
from .types import BoxState, Tracklet
