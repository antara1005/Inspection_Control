# The sparse focus volume at one camera frame (focus_volume_node).

# Stamp and frame_id of the camera image these features were measured in.
std_msgs/Header header
uint32 image_width
uint32 image_height

# Focal distance (m) paired with this frame; NaN if none or stale.
float32 distance

FocusFeature[] features

# Summary.
uint32 n_tentative
uint32 n_active
uint32 n_coasting
uint32 n_spawned            # seeded this frame
uint32 n_retired            # retired this frame
float32 coverage            # fraction of grid cells holding an active feature
float32 median_focus_norm   # over active features; NaN if none
float32 median_flow_px      # per frame, active features, px (full resolution)
uint32 cells_detected       # grid cells the detector ran in this frame
float32 process_ms          # processing time of this frame
