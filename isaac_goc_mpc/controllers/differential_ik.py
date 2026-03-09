import torch
from dataclasses import dataclass


@dataclass
class CartesianVelocityControllerCfg:
    """Configuration for the Cartesian velocity controller."""
    device: str
    damping: float = 0.05  # Standard damping (lambda) for DLS


class CartesianVelocityController:
    """A Jacobian-based velocity controller for Isaac Lab."""

    def __init__(self, cfg: CartesianVelocityControllerCfg):
        self.cfg = cfg
        # identity damping term (6x6 for Task Space)
        self._damping = torch.eye(6, device=self.cfg.device)

    def compute(self, v_desired: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """
        Computes joint velocities using Damped Least Squares.

        Args:
            v_desired: Target EE velocity [N, 6]  (linear + angular)
            jacobian:  The EE Jacobian    [N, 6, num_joints]

        Returns:
            Joint velocities [N, num_joints]
        """
        # J * J^T
        jj_t = torch.matmul(jacobian, jacobian.transpose(1, 2))

        # Damping term: lambda^2 * I
        damping_term = (self.cfg.damping ** 2) * self._damping

        # Solve: dot_q = J^T * inv(J*J^T + lambda^2*I) * v_desired
        # Using linalg.solve is faster and more stable than explicit inverse
        v_col = v_desired.unsqueeze(-1)
        inv_term = torch.linalg.solve(jj_t + damping_term, v_col)

        q_dot = torch.matmul(jacobian.transpose(1, 2), inv_term).squeeze(-1)

        return q_dot
