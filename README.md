# OfflinePoly.mojo

[![mojoshelf](https://mojoshelf.org/badge/offlinepoly.svg)](https://mojoshelf.org/tins/offlinepoly) [![mojo nightly](https://mojoshelf.org/badge/offlinepoly/nightly.svg)](https://mojoshelf.org/tins/offlinepoly)

A pure-[Mojo](https://www.modular.com/mojo), learning-free implementation of
**Offline-Poly** ([arXiv:2602.13772](https://arxiv.org/abs/2602.13772), Li et
al. 2026): offline 3D multi-object tracking under the **Tracking-By-Tracking
(TBT)** paradigm, built for 4D auto-labeling (4DAL) pipelines.

TBT consumes the *final tracklets* of one or more arbitrary upstream trackers
— no frame-wise detections, no association metadata, no maps, no learned
weights — and emits globally refined trajectories. Because the method is
entirely learning-free, there is no checkpoint and therefore no dataset
license attached to one: the implementation is usable as-is.

> The authors' official code ([K544-AD/Offline-Poly](https://github.com/K544-AD/Offline-Poly),
> MIT) had not been released as of Aug 2026; this is an independent
> from-paper implementation and has **not** been validated against the
> paper's nuScenes/KITTI numbers.

## Pipeline

Four stages, mirroring the paper (Fig. 2):

1. **Tracklet-level pre-processing** (Sec. IV-B) — drops ghost tracklets
   whose age *and* mean confidence are both below threshold.
2. **Single-tracker matching & fusion** (Sec. IV-C) — per source:
   - **STWO** re-links temporally disjoint fragments of the same object via
     forward/backward constant-velocity extrapolation (Eq. 2), 3D-IoU cost
     (Eq. 3), and one-to-one assignment, iterated frame-by-frame; merged
     gaps are interpolated bidirectionally.
   - **STW** disentangles tracklets erroneously merged from several objects:
     gIoU clustering, per-frame connected-component separation into reliable
     segments + entangled nodes, then STWO reorganization.
3. **Multi-tracker matching & fusion** (Sec. IV-D) — scene-level adjacency
   (per-frame IoU threshold, max-pooled over time, Eqs. 6–8), DFS clustering
   (Eq. 9), and confidence-weighted per-frame fusion of each cluster.
4. **Multi-perspective refinement** (Sec. IV-E) —
   - **Global**: rigid-object size from a softmax-weighted vote of the Top-K
     most confident observations; corner-aligned center correction (needs an
     ego trajectory; bottom face is always kept fixed vertically).
   - **Local**: sliding-window least-squares of (center, velocity, heading)
     against past *and* future neighbors under the CV model (Eq. 10).

### Deviations from the paper

- The blossom assignment solver is replaced by *exact* maximum-weight
  matching within each connected component of the thresholded cost graph
  (components are tiny in practice), with a greedy fallback for components
  larger than `max_exact_component`. Same optimum in effect.
- Only the Constant Velocity motion model is implemented (the paper's KITTI
  configuration); CTRA is not.
- Eq. 10 is solved with damped Gauss-Newton rather than
  Levenberg-Marquardt — under CV the problem is linear, so the update is
  identical.
- Heading fusion uses a circular weighted mean without 180° flip handling.

## Install as a mojoshelf tin

Published on [mojoshelf](https://mojoshelf.org/tins/offlinepoly) as `offlinepoly`:

```sh
pixi shelf add offlinepoly     # pixi mode (git source dependency)
shelf add offlinepoly          # or as a git submodule
```

Maintainers release new versions with `shelf publish` from the repo root
(see [getting started](https://mojoshelf.org/getting-started)).

## Usage

```sh
pixi run test    # unit tests
pixi run demo    # 2 synthetic trackers -> examples/refined.csv
```

CLI — one CSV per upstream tracker, rows
`track_id,cls,t,x,y,z,w,l,h,vx,vy,theta,conf` in global coordinates
(`l` along heading `theta`):

```sh
mojo run -I src src/main.mojo OUT.csv IN1.csv [IN2.csv ...] [--ego EGO.csv]
```

`--ego` (rows `t,x,y`) enables corner-aligned position correction.

As a library:

```mojo
from offlinepoly import (
    OfflinePolyConfig, assign_frames, read_tracker_csv, run_pipeline,
)

def main() raises:
    var sources = [read_tracker_csv("fp_forward.csv"),
                   read_tracker_csv("fp_backward.csv")]
    var frame_times = assign_frames(sources)
    var refined = run_pipeline(
        sources, frame_times, OfflinePolyConfig(), ego_xy=[]
    )
```

All hyper-parameters (thresholds θ_age, θ_score, θ_blo, θ_stw, θ_multi,
Top-K, window M, prediction horizon) live in `OfflinePolyConfig` with
defaults following the paper's sensitivity analysis (Sec. V-E).

## Why Mojo

Offline-Poly is the rare 4DAL component that is *entirely* geometry and
combinatorics — rotated-box IoU, motion extrapolation, graph matching,
least squares — with no training and no tensor-framework dependency. That
makes it a natural fit for a fast, dependency-free Mojo library. It pairs
with [LabelFormer.mojo](https://github.com/labelrefinery/LabelFormer.mojo):
Offline-Poly consolidates upstream tracks (learning-free), LabelFormer
refines the resulting trajectories (learned) — together a complete tier-3
offboard refinement stack in pure Mojo.

## Citation

```bibtex
@article{li2026offlinepoly,
  title={Offline-Poly: A Polyhedral Framework For Offline 3D Multi-Object Tracking},
  author={Li, Xiaoyu and Wu, Yitao and Wu, Xian and Zhuo, Haolin and Zhao, Lijun and Sun, Lining},
  journal={arXiv preprint arXiv:2602.13772},
  year={2026}
}
```

## License

MIT (this implementation). The Offline-Poly paper is CC BY-NC-SA 4.0; no
text, figures, or code from the authors are included here.
