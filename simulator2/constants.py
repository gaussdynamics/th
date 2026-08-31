"""Physical and default numerical constants aligned with the LTD buildup notebook."""

# Gravity [m/s^2] (notebook: G_MPS2)
GRAVITY_MPS2 = 9.81

# Stage 2 / illustrative masses and Davis (locomotive)
DEFAULT_M_LOCO_KG = 130_000.0
DEFAULT_DAVIS_A_LOCO = 800.0
DEFAULT_DAVIS_B_LOCO = 15.0
DEFAULT_DAVIS_C_LOCO = 0.8

# Freight car defaults
DEFAULT_M_CAR_KG = 100_000.0
DEFAULT_DAVIS_A_CAR = 600.0
DEFAULT_DAVIS_B_CAR = 12.0
DEFAULT_DAVIS_C_CAR = 0.7

# Default coupler stiffness / damping / slack (Stage 3 notebook)
DEFAULT_SLACK_HALF_M = 0.02
DEFAULT_K_DRAFT = 9.0e6
DEFAULT_C_DRAFT = 5.0e5
DEFAULT_K_BUFF = 12.0e6
DEFAULT_C_BUFF = 7.0e5

# Stage 7 extended dynamics
DEFAULT_TAU_BRK_S = 3.0
DEFAULT_TAU_TRAC_S = 5.0
DEFAULT_P_MAX_W = 3.5e6
DEFAULT_V_EPS = 1.0

# Blend width [m/s] for the direction-of-travel sign applied to brake force
# when ``brake_opposes_motion`` is enabled (see rhs.train_rhs_tensorized).
# Small enough that it does not soften braking at operating speed, large enough
# to keep the RHS smooth through v = 0.
DEFAULT_V_BRAKE_EPS = 0.1

# Default simulation tolerances (notebook TrainSimulationConfig)
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-8
DEFAULT_METHOD = "RK45"
