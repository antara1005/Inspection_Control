"""Sparse focus volume: the feature front end of the hybrid autofocus.

Implements the front end that the Hybrid Autofocus Design (sections 3 and 9.2 to
9.5) pinned down empirically: **AKAZE-acquire, KLT-sustain, FB-gate-retire**,
with a grid spawn manager, and a local focus measure per tracked feature. Each
feature's focus history is a column of the sparse focus volume, and its
running-max-normalised focus (P1) is scale-free, so it means the same thing on
steel as on a textured part.

No ROS imports: focus_volume_node runs it on the live camera, and it runs the
same way offline on recorded frames (test plan S2).

Coordinates: detection and tracking run on a downscaled copy of each frame
(``track_scale``); every position this module returns is in pixels of the
full-resolution frame, so it lines up with the camera image.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import time

import cv2
import numpy as np

# Track states. The values match inspection_control_interfaces/FocusFeature.
TENTATIVE = 0   # seeded, not trusted yet: excluded from every estimate
ACTIVE = 1      # tracked and confirmed
COASTING = 2    # lost by KLT; held at a predicted position, re-acquired or retired
STATE_NAMES = {TENTATIVE: 'tentative', ACTIVE: 'active', COASTING: 'coasting'}

FOCUS_OPERATORS = ('laplacian', 'tenengrad')


@dataclass
class FocusVolumeConfig:
    """Tunables. Pixel sizes are at track scale unless they say full res."""

    # Detection and KLT run on the frame scaled by this factor.
    track_scale: float = 0.5
    # Grid spawn manager (design 9.3). Cells cover detect_region, given as
    # fractions of the frame (x0, y0, x1, y1).
    grid_cols: int = 6
    grid_rows: int = 6
    detect_region: tuple = (0.0, 0.0, 1.0, 1.0)
    per_cell: int = 8               # target live tracks per cell
    refill_below: float = 0.5       # detect in a cell once it holds < this * per_cell
    max_cells_per_frame: int = 6    # detection budget, cells per frame
    # Above this fraction of cells needing detection, detect over the whole
    # region in one pass (cheaper than many overlapping crops, e.g. at start).
    full_detect_fraction: float = 0.5
    # A cell where detection found nothing new (no texture there) is not
    # searched again for this many frames.
    empty_cell_backoff: int = 15
    # AKAZE. A low threshold finds structure in blur (design 9.2): calibrate it
    # on blurred frames, not sharp ones. Lower = more, weaker features. On the
    # macro camera (2028x1520, track_scale 0.5) AKAZE's usual 0.001 finds next to
    # nothing on a 3D-printed part; 1e-4 finds ~200 over the frame.
    akaze_threshold: float = 0.0001
    akaze_octaves: int = 4
    akaze_octave_layers: int = 4
    min_distance: float = 8.0       # px between features
    # Pyramidal Lucas-Kanade with a forward-backward check.
    klt_window: int = 21
    klt_levels: int = 3
    fb_max_error: float = 1.0       # px
    # Track states (design 9.5).
    confirm_frames: int = 3         # consecutive good frames to promote tentative -> active
    coast_frames: int = 5           # frames a lost track coasts before it is retired
    # Local focus measure per feature, on the full-resolution frame.
    focus_operator: str = 'laplacian'   # laplacian (energy) | tenengrad (gradient energy)
    # px, full res: side of the window around each feature. Measured on the
    # macro camera, still scene: 31 px -> 14 % frame-to-frame noise, 61 px -> 8 %.
    focus_window: int = 61
    focus_presmooth: float = 0.0    # Gaussian sigma (px, full res) before the operator
    # s, EMA of each feature's focus. The running max and the normalised focus
    # use the smoothed value: the max of the raw value keeps creeping up with
    # noise, so a still feature would drift below its "peak". Smoothing delays
    # the peak by a little under this long; z_at_max compensates.
    focus_tau: float = 0.1
    focus_rate_tau: float = 0.3     # s, EMA of d(ln focus)/dt
    # A feature whose normalised focus falls below this after its max has
    # passed its peak: its max is then a real peak of its focus curve.
    peak_drop: float = 0.7
    trail_frames: int = 8           # positions kept per track, for drawing motion


@dataclass
class Track:
    """One tracked feature: a column of the focus volume."""

    track_id: int
    x: float                        # track-scale pixel position
    y: float
    born_t: float
    state: int = TENTATIVE
    age: int = 0                    # frames since spawn
    good_frames: int = 0            # consecutive KLT successes
    lost_frames: int = 0            # consecutive KLT failures (coasting)
    fb_error: float = 0.0           # px, track scale
    focus_raw: float = math.nan     # phi_i this frame, operator units
    focus: float = math.nan         # phi_i smoothed (focus_tau)
    focus_max: float = 0.0          # running max of the smoothed phi_i
    focus_norm: float = math.nan    # phi_i / running max, 0..1
    focus_rate: float = 0.0         # EMA of d(ln phi_i)/dt, 1/s (> 0: sharpening)
    z_at_max: float = math.nan      # focal distance when focus_max was seen, m
    passed_peak: bool = False
    last_focus_t: float = math.nan
    trail: deque = field(default_factory=deque)


@dataclass
class FeatureView:
    """Read-only snapshot of a track for publishing, in full-res pixels."""

    track_id: int
    state: int
    u: float
    v: float
    focus_raw: float
    focus: float
    focus_max: float
    focus_norm: float
    focus_rate: float
    z_at_max: float
    passed_peak: bool
    age: int
    fb_error: float
    trail_u: float                  # oldest kept position, for a motion tail
    trail_v: float
    cell: int


@dataclass
class FrameResult:
    """Everything one processed frame produced."""

    t: float
    z: float                        # focal distance used for this frame, NaN if none
    width: int
    height: int
    features: list
    n_tentative: int
    n_active: int
    n_coasting: int
    n_spawned: int
    n_retired: int
    retired_by: dict                # reason -> count, this frame
    coverage: float                 # fraction of cells with an active track
    median_focus_norm: float        # over active tracks
    median_flow_px: float           # full-res px per frame, active tracks
    cells: list                     # (x0, y0, x1, y1) full-res rect per cell
    cell_active: list               # active tracks per cell
    cell_focus_norm: list           # median normalised focus per cell (NaN if none)
    cells_detected: list            # cells AKAZE ran in this frame
    full_detect: bool
    process_ms: float


class FocusVolumeTracker:
    """Detect, track and measure features frame by frame.

    Feed full-resolution grayscale frames to :meth:`process` in order. ``z`` is
    the focal distance (m) at that frame if known: it is what turns each
    feature's focus history into a curve over metres (``z_at_max``).
    """

    def __init__(self, config: FocusVolumeConfig = None):
        self.cfg = config or FocusVolumeConfig()
        if self.cfg.focus_operator not in FOCUS_OPERATORS:
            raise ValueError(f'focus_operator must be one of {FOCUS_OPERATORS}, '
                             f"not '{self.cfg.focus_operator}'")
        self._akaze = cv2.AKAZE_create(
            threshold=self.cfg.akaze_threshold,
            nOctaves=self.cfg.akaze_octaves,
            nOctaveLayers=self.cfg.akaze_octave_layers)
        self._lk = {
            'winSize': (self.cfg.klt_window, self.cfg.klt_window),
            'maxLevel': self.cfg.klt_levels,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        self.reset()

    def reset(self):
        """Forget every track (e.g. after the image size changes)."""
        self.tracks = []
        self._next_id = 0
        self._prev_small = None
        self._shape = None
        self._cells = None          # track-scale (x0, y0, x1, y1) per cell
        self._next_try = None       # per cell: first frame it may be searched again
        self._frame = 0
        self._rotate = 0            # round-robin offset among equally needy cells
        self._z_hist = deque(maxlen=120)    # (t, z): to undo the smoothing delay
        self._last_t = None

    # ------------------------------------------------------------------ main
    def process(self, gray, t, z=math.nan) -> FrameResult:
        """Process one full-resolution grayscale frame taken at time ``t`` (s)."""
        t0 = time.perf_counter()
        if gray.ndim != 2:
            raise ValueError('process() takes a single-channel (grayscale) frame')
        if self._shape != gray.shape:
            self.reset()
            self._shape = gray.shape
            self._cells = self._make_cells(gray.shape)
            self._next_try = np.zeros(len(self._cells), int)
        self._frame += 1
        dt = t - self._last_t if self._last_t is not None else math.nan
        self._last_t = t
        if math.isfinite(z):
            self._z_hist.append((t, float(z)))
        s = self.cfg.track_scale
        small = gray if s == 1.0 else cv2.resize(
            gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        retired = {}
        flow = self._track(small, retired)
        self._suppress_duplicates(retired)
        cells_detected, full_detect, n_spawned = self._spawn(small, t)
        self._measure_focus(gray, t, dt, z)
        self._prev_small = small

        for tr in self.tracks:
            tr.age += 1
            tr.trail.append((tr.x, tr.y))
            while len(tr.trail) > self.cfg.trail_frames:
                tr.trail.popleft()

        return self._result(t, z, gray.shape, flow, retired, n_spawned,
                            cells_detected, full_detect,
                            (time.perf_counter() - t0) * 1000.0)

    # ----------------------------------------------------------- tracking
    def _track(self, small, retired):
        """KLT every live track from the previous frame; FB-gate the result.

        Returns the median per-frame flow of the active tracks (track-scale px,
        as (dx, dy)), which also moves the coasting tracks.
        """
        if self._prev_small is None or not self.tracks:
            return np.zeros(2)
        p0 = np.array([[tr.x, tr.y] for tr in self.tracks], np.float32).reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self._prev_small, small, p0, None, **self._lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(small, self._prev_small, p1, None, **self._lk)
        p0, p1, p0r = p0.reshape(-1, 2), p1.reshape(-1, 2), p0r.reshape(-1, 2)
        fb = np.linalg.norm(p0 - p0r, axis=1)
        h, w = small.shape
        inside = ((p1[:, 0] >= 0) & (p1[:, 0] < w - 1)
                  & (p1[:, 1] >= 0) & (p1[:, 1] < h - 1))
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.cfg.fb_max_error) & inside

        active = np.array([tr.state == ACTIVE for tr in self.tracks])
        sel = ok & active
        flow = np.median(p1[sel] - p0[sel], axis=0) if sel.any() else np.zeros(2)

        keep = []
        for i, tr in enumerate(self.tracks):
            tr.fb_error = float(fb[i])
            if ok[i]:
                tr.x, tr.y = float(p1[i, 0]), float(p1[i, 1])
                tr.good_frames += 1
                tr.lost_frames = 0
                if tr.state == COASTING:
                    tr.state = ACTIVE           # re-acquired: keeps its history
                elif tr.state == TENTATIVE and tr.good_frames >= self.cfg.confirm_frames:
                    tr.state = ACTIVE
                keep.append(tr)
                continue
            # Lost this frame.
            if tr.state == TENTATIVE:
                retired['tentative_lost'] = retired.get('tentative_lost', 0) + 1
                continue
            tr.state = COASTING
            tr.good_frames = 0
            tr.lost_frames += 1
            tr.x += float(flow[0])
            tr.y += float(flow[1])
            if not (0 <= tr.x < w - 1 and 0 <= tr.y < h - 1):
                retired['left_frame'] = retired.get('left_frame', 0) + 1
            elif tr.lost_frames > self.cfg.coast_frames:
                retired['coast_timeout'] = retired.get('coast_timeout', 0) + 1
            else:
                keep.append(tr)
        self.tracks = keep
        return flow

    def _suppress_duplicates(self, retired):
        """Tracks that converge onto one point: keep the most established."""
        if len(self.tracks) < 2:
            return
        xy = np.array([[tr.x, tr.y] for tr in self.tracks])
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        limit = (0.5 * self.cfg.min_distance) ** 2
        if not (d2 < limit).any():
            return
        # Rank: active over coasting over tentative, then older first.
        rank = {ACTIVE: 0, COASTING: 1, TENTATIVE: 2}
        order = sorted(range(len(self.tracks)),
                       key=lambda i: (rank[self.tracks[i].state], -self.tracks[i].age))
        dead = set()
        for i in order:
            if i in dead:
                continue
            for j in np.nonzero(d2[i] < limit)[0]:
                if j not in dead and j != i:
                    dead.add(int(j))
        if dead:
            retired['duplicate'] = retired.get('duplicate', 0) + len(dead)
            self.tracks = [tr for k, tr in enumerate(self.tracks) if k not in dead]

    # ------------------------------------------------------------ spawning
    def _make_cells(self, shape):
        h, w = shape
        s = self.cfg.track_scale
        sw, sh = w * s, h * s
        x0, y0, x1, y1 = self.cfg.detect_region
        xs = np.linspace(x0 * sw, x1 * sw, self.cfg.grid_cols + 1)
        ys = np.linspace(y0 * sh, y1 * sh, self.cfg.grid_rows + 1)
        return [(float(xs[c]), float(ys[r]), float(xs[c + 1]), float(ys[r + 1]))
                for r in range(self.cfg.grid_rows) for c in range(self.cfg.grid_cols)]

    def _cell_of(self, x, y):
        """Index of the cell containing (x, y) at track scale, or -1."""
        x0, y0 = self._cells[0][0], self._cells[0][1]
        x1, y1 = self._cells[-1][2], self._cells[-1][3]
        if not (x0 <= x < x1 and y0 <= y < y1):
            return -1
        c = int((x - x0) / (x1 - x0) * self.cfg.grid_cols)
        r = int((y - y0) / (y1 - y0) * self.cfg.grid_rows)
        return (min(r, self.cfg.grid_rows - 1) * self.cfg.grid_cols
                + min(c, self.cfg.grid_cols - 1))

    def _spawn(self, small, t):
        """Seed AKAZE features in the cells that have too few live tracks."""
        n = len(self._cells)
        count = np.zeros(n, int)
        for tr in self.tracks:
            c = self._cell_of(tr.x, tr.y)
            if c >= 0:
                count[c] += 1
        need = [c for c in range(n)
                if count[c] < self.cfg.refill_below * self.cfg.per_cell
                and self._next_try[c] <= self._frame]
        if not need:
            return [], False, 0
        full = len(need) > self.cfg.full_detect_fraction * n
        if full:
            chosen = need
            keypoints = self._detect(small, self._region_rect())
        else:
            # Emptiest first; rotate among ties so no cell starves.
            self._rotate = (self._rotate + 1) % n
            need.sort(key=lambda c: (count[c], (c - self._rotate) % n))
            chosen = need[:self.cfg.max_cells_per_frame]
            keypoints = []
            for c in chosen:
                keypoints += self._detect(small, self._cells[c])
        chosen_set = set(chosen)

        existing = np.array([[tr.x, tr.y] for tr in self.tracks]).reshape(-1, 2)
        min_d2 = self.cfg.min_distance ** 2
        keypoints.sort(key=lambda kp: -kp[2])         # strongest first
        added = []
        spawned_in = np.zeros(n, int)
        for x, y, _ in keypoints:
            c = self._cell_of(x, y)
            if c not in chosen_set or count[c] >= self.cfg.per_cell:
                continue
            if existing.size and (((existing - (x, y)) ** 2).sum(1) < min_d2).any():
                continue
            if added and (((np.array(added) - (x, y)) ** 2).sum(1) < min_d2).any():
                continue
            self.tracks.append(Track(track_id=self._next_id, x=float(x), y=float(y), born_t=t))
            self._next_id += 1
            added.append((x, y))
            count[c] += 1
            spawned_in[c] += 1
        for c in chosen:
            if spawned_in[c] == 0:
                self._next_try[c] = self._frame + self.cfg.empty_cell_backoff
        return chosen, full, int(spawned_in.sum())

    def _region_rect(self):
        return (self._cells[0][0], self._cells[0][1], self._cells[-1][2], self._cells[-1][3])

    def _detect(self, small, rect, margin=16):
        """AKAZE keypoints (x, y, response) inside ``rect``, track-scale px.

        Detects on a crop with a margin around the rect, so the cost scales with
        the area searched (a mask would not save AKAZE's scale-space build).
        """
        h, w = small.shape
        x0, y0, x1, y1 = rect
        cx0, cy0 = max(0, int(x0) - margin), max(0, int(y0) - margin)
        cx1, cy1 = min(w, int(math.ceil(x1)) + margin), min(h, int(math.ceil(y1)) + margin)
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            return []
        kps = self._akaze.detect(small[cy0:cy1, cx0:cx1], None)
        out = []
        for kp in kps:
            x, y = kp.pt[0] + cx0, kp.pt[1] + cy0
            if x0 <= x < x1 and y0 <= y < y1:
                out.append((x, y, kp.response))
        return out

    # --------------------------------------------------------------- focus
    def _measure_focus(self, gray, t, dt, z):
        """Local focus measure phi_i around each non-coasting track.

        The operator's energy image is integrated once per frame, so each
        feature's windowed mean costs four lookups whatever the window size.
        """
        live = [tr for tr in self.tracks if tr.state != COASTING]
        if not live:
            return
        g = gray.astype(np.float32)
        if self.cfg.focus_presmooth > 0:
            g = cv2.GaussianBlur(g, (0, 0), self.cfg.focus_presmooth)
        if self.cfg.focus_operator == 'laplacian':
            r = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
            energy = r * r
        else:   # tenengrad
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            energy = gx * gx + gy * gy
        ii = cv2.integral(energy, sdepth=cv2.CV_64F)

        h, w = gray.shape
        half = self.cfg.focus_window // 2
        inv_s = 1.0 / self.cfg.track_scale
        u = np.array([tr.x * inv_s for tr in live])
        v = np.array([tr.y * inv_s for tr in live])
        x0 = np.clip(np.round(u).astype(int) - half, 0, w - 1)
        x1 = np.clip(np.round(u).astype(int) + half + 1, 1, w)
        y0 = np.clip(np.round(v).astype(int) - half, 0, h - 1)
        y1 = np.clip(np.round(v).astype(int) + half + 1, 1, h)
        area = np.maximum((x1 - x0) * (y1 - y0), 1)
        phi = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / area

        tau = max(self.cfg.focus_tau, 1e-6)
        rate_tau = max(self.cfg.focus_rate_tau, 1e-6)
        # The smoothed focus lags the raw one by the filter's delay, so its max
        # is reached that much after the true peak: pair it with the distance
        # from that long ago. Per-frame EMA delay: dt * d / (1 - d), d = e^(-dt/tau).
        if math.isfinite(dt) and dt > 0:
            d = math.exp(-dt / tau)
            delay = dt * d / (1.0 - d)
        else:
            delay = tau
        z_lagged = self._z_at(t - delay) if math.isfinite(z) else math.nan
        for tr, f in zip(live, phi):
            f = float(f)
            if not math.isfinite(f) or f <= 0.0:
                continue
            tr.focus_raw = f
            if not math.isfinite(tr.focus):
                tr.focus = f
            else:
                dt = t - tr.last_focus_t
                if dt > 1e-6:
                    smoothed = tr.focus + (1.0 - math.exp(-dt / tau)) * (f - tr.focus)
                    rate = (math.log(smoothed) - math.log(tr.focus)) / dt
                    tr.focus_rate += (1.0 - math.exp(-dt / rate_tau)) * (rate - tr.focus_rate)
                    tr.focus = smoothed
            tr.last_focus_t = t
            if tr.focus > tr.focus_max:
                tr.focus_max = tr.focus
                tr.z_at_max = z_lagged
                tr.passed_peak = False
            tr.focus_norm = tr.focus / tr.focus_max
            if tr.focus_norm < self.cfg.peak_drop:
                tr.passed_peak = True

    def _z_at(self, t):
        """Focal distance at time ``t``, interpolated between recorded frames."""
        if not self._z_hist:
            return math.nan
        ts, zs = zip(*self._z_hist)
        return float(np.interp(t, ts, zs))

    # -------------------------------------------------------------- output
    def _result(self, t, z, shape, flow, retired, n_spawned, cells_detected,
                full_detect, process_ms) -> FrameResult:
        inv_s = 1.0 / self.cfg.track_scale
        n = len(self._cells)
        cell_active = [0] * n
        cell_norms = [[] for _ in range(n)]
        views = []
        counts = {TENTATIVE: 0, ACTIVE: 0, COASTING: 0}
        for tr in self.tracks:
            counts[tr.state] += 1
            c = self._cell_of(tr.x, tr.y)
            if tr.state == ACTIVE and c >= 0:
                cell_active[c] += 1
                if math.isfinite(tr.focus_norm):
                    cell_norms[c].append(tr.focus_norm)
            tx, ty = tr.trail[0] if tr.trail else (tr.x, tr.y)
            views.append(FeatureView(
                track_id=tr.track_id, state=tr.state, u=tr.x * inv_s, v=tr.y * inv_s,
                focus_raw=tr.focus_raw, focus=tr.focus, focus_max=tr.focus_max,
                focus_norm=tr.focus_norm,
                focus_rate=tr.focus_rate, z_at_max=tr.z_at_max,
                passed_peak=tr.passed_peak, age=tr.age, fb_error=tr.fb_error * inv_s,
                trail_u=tx * inv_s, trail_v=ty * inv_s, cell=c))
        active_norms = [tr.focus_norm for tr in self.tracks
                        if tr.state == ACTIVE and math.isfinite(tr.focus_norm)]
        return FrameResult(
            t=t, z=float(z), width=int(shape[1]), height=int(shape[0]), features=views,
            n_tentative=counts[TENTATIVE], n_active=counts[ACTIVE],
            n_coasting=counts[COASTING], n_spawned=n_spawned,
            n_retired=sum(retired.values()), retired_by=retired,
            coverage=sum(1 for a in cell_active if a > 0) / n,
            median_focus_norm=float(np.median(active_norms)) if active_norms else math.nan,
            median_flow_px=float(np.hypot(*flow)) * inv_s,
            cells=[(x0 * inv_s, y0 * inv_s, x1 * inv_s, y1 * inv_s)
                   for x0, y0, x1, y1 in self._cells],
            cell_active=cell_active,
            cell_focus_norm=[float(np.median(c)) if c else math.nan for c in cell_norms],
            cells_detected=list(cells_detected), full_detect=full_detect,
            process_ms=process_ms)
