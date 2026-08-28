"""Hyper-parameters of the Offline-Poly pipeline (paper Sec. V-B / Fig. 8)."""


struct OfflinePolyConfig(Copyable, Movable):
    # Pre-processing: drop tracklets with age < age_threshold AND
    # mean confidence < score_threshold (both must fail).
    var age_threshold: Int
    var score_threshold: Float64

    # STWO re-identification: IoU3d cost threshold (theta_blo) and iteration cap.
    var theta_blo: Float64
    var stwo_max_passes: Int

    # STW disentanglement: gIoU3d cost threshold for the frame-level adjacency.
    var theta_stw: Float64

    # Multi-tracker matching: IoU3d cost threshold (theta_multi).
    var theta_multi: Float64

    # Motion prediction horizon (Eq. 2); predictions beyond this are discarded.
    var max_pred_seconds: Float64

    # Assignment solving: components up to this size are matched exactly,
    # larger ones greedily (stand-in for the paper's blossom algorithm).
    var max_exact_component: Int

    # Global refinement: number of most-confident observations for the
    # softmax-weighted size estimate; classes whose size is NOT constant.
    var top_k: Int
    var nonrigid_classes: List[Int]

    # Local refinement: sliding window length M (frames) and its time cap.
    var window: Int
    var max_window_seconds: Float64

    def __init__(out self):
        self.age_threshold = 3
        self.score_threshold = 0.3
        self.theta_blo = 0.9
        self.stwo_max_passes = 3
        self.theta_stw = 0.7
        self.theta_multi = 0.8
        self.max_pred_seconds = 1.0
        self.max_exact_component = 10
        self.top_k = 10
        self.nonrigid_classes = List[Int]()
        self.window = 8
        self.max_window_seconds = 4.0

    def is_rigid(self, cls: Int) -> Bool:
        for c in self.nonrigid_classes:
            if c == cls:
                return False
        return True
