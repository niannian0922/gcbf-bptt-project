from typing import Optional

# Expose the refactored environment modules
from .base_env import BaseEnv, EnvState, StepResult
from .multi_agent_env import MultiAgentEnv, MultiAgentState
from .double_integrator import DoubleIntegratorEnv, DoubleIntegratorState
from .crazyflie_env import CrazyFlieEnv, CrazyFlieState
from .gcbf_safety_layer import GCBFSafetyLayer, GCBFPlusAgent

# For backward compatibility, keep the old imports
# These will be gradually phased out as we refactor the codebase
# Commented out to avoid dependency issues during testing
# from .base import MultiAgentEnv as OldMultiAgentEnv
# from .double_integrator import DoubleIntegrator
# from .crazyflie import CrazyFlie
# from .dubins_car import DubinsCar
# from .linear_drone import LinearDrone
# from .single_integrator import SingleIntegrator
# from .obstacle import Obstacle, Rectangle, Sphere
