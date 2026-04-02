"""Top-level solver exports and compatibility aliases."""

from nnc_joint_solver.base import (
    CliJointScheduleSolver,
    DEFAULT_SOLVER_TIMEOUT_SECONDS,
    JointScheduleSolver,
    JointSolverTransportError,
)
from nnc_joint_solver.v0.solver import BaselineJointScheduleSolver, V0JointScheduleSolver
from nnc_joint_solver.v1.solver import V1JointScheduleSolver


LatestJointScheduleSolver = V1JointScheduleSolver


__all__ = [
    "BaselineJointScheduleSolver",
    "CliJointScheduleSolver",
    "DEFAULT_SOLVER_TIMEOUT_SECONDS",
    "JointScheduleSolver",
    "JointSolverTransportError",
    "LatestJointScheduleSolver",
    "V0JointScheduleSolver",
    "V1JointScheduleSolver",
]
