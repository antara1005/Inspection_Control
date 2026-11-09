# One tracked feature of the sparse focus volume (focus_volume_node): a column
# of the volume, i.e. one feature's focus measured frame after frame.

uint8 TENTATIVE=0   # seeded, not trusted yet; excluded from every estimate
uint8 ACTIVE=1      # tracked and confirmed
uint8 COASTING=2    # lost by the tracker; held at a predicted position

uint32 id           # stable for the track's life
uint8 state

# Position in pixels of the full-resolution camera image.
float32 u
float32 v

# Local focus measure phi_i in a window around the feature (operator units,
# see the node's focus_operator): this frame's value, the value smoothed over
# the node's focus_tau, the running max of the smoothed value, and smoothed /
# max (0..1, scale-free). Coasting tracks keep their last values.
float32 focus_raw
float32 focus
float32 focus_max
float32 focus_norm
# Smoothed d(ln phi_i)/dt, 1/s: > 0 sharpening, < 0 blurring.
float32 focus_rate

# Focal distance (m) when focus_max was seen, corrected for the smoothing
# delay; NaN without a distance input.
float32 z_at_max
# focus_norm has fallen below the node's peak_drop since focus_max: the max is
# a peak of this feature's focus curve, not just the sharpest so far.
bool passed_peak

uint32 age          # frames since the feature was seeded
float32 fb_error    # forward-backward tracking error, px (full resolution)
