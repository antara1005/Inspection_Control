# The virtual sphere the cell's assistive controllers share: the plant
# admittance_control integrates, and the model the orientation controller and
# autofocus place their poles on (inspection_control/virtual_sphere.py).
# admittance_control owns the values (its sphere_mass, sphere_radius and
# fluid_viscosity parameters) and publishes this, latched, on ~/virtual_sphere
# at startup and whenever they change.
std_msgs/Header header

float64 mass          # kg
float64 radius        # m
float64 viscosity     # Pa·s

# Derived from the three above, as the nodes compute them:
float64 inertia       # kg·m², 2/5 m r²
float64 linear_drag   # N·s/m, 6 π μ r (Stokes)
float64 angular_drag  # N·m·s/rad, 2.4 π μ r³
