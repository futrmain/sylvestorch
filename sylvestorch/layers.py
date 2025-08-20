import torch
from typing import Dict, Any, Tuple, Optional

from .parametrizations import Triu, DiagAct, Orthogonalize


class SylvesterBlock(torch.nn.Module):
    """
    Sylvester normalizing flow block.
    
    This module implements a Sylvester normalizing flow transformation that
    provides an invertible mapping with tractable Jacobian determinant.
    The transformation uses orthogonal and triangular matrix parametrizations
    to ensure numerical stability and efficient computation.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - hidden_size (int): Dimension of hidden representations
            - input_size (int): Dimension of input data
            - ortho_eps (float, optional): Orthogonalization tolerance
    
    Attributes:
        R1 (torch.nn.Linear): First triangular transformation matrix
        R2 (torch.nn.Linear): Second triangular transformation matrix  
        Q (torch.nn.Linear): Orthogonal transformation matrix
        bias (torch.nn.Parameter): Bias parameter
        act (callable): Activation function (tanh)
    
    Example:
        >>> config = {
        ...     'hidden_size': 64,
        ...     'input_size': 32,
        ...     'ortho_eps': 1e-6
        ... }
        >>> block = SylvesterBlock(config)
        >>> x = torch.randn(100, 32)  # batch_size=100, input_size=32
        >>> y, log_det = block(x)
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        ortho_eps: float = 1e-6
    ) -> None:
        super(SylvesterBlock, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ortho_eps = ortho_eps

        # Activation function
        # Should be tanh or else needs to replace der_act(self, x)
        self.act = torch.tanh

        # First triangular matrix R1
        self.R1 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        torch.nn.utils.parametrize.register_parametrization(
            self.R1, "weight", Triu(0)
        )
        torch.nn.utils.parametrize.register_parametrization(
            self.R1, "weight", DiagAct(hidden_size, self.act)
        )

        # Second triangular matrix R2
        self.R2 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        torch.nn.utils.parametrize.register_parametrization(
            self.R2, "weight", Triu(0)
        )
        torch.nn.utils.parametrize.register_parametrization(
            self.R2, "weight", DiagAct(hidden_size, self.act)
        )

        # Orthogonal matrix Q 
        self.Q  = torch.nn.Linear(hidden_size, input_size, bias=False)
        self._initialize_orthogonal_matrix(hidden_size)
        torch.nn.utils.parametrize.register_parametrization(
            self.Q, "weight", 
            Orthogonalize(input_size, hidden_size, eps=ortho_eps))

        # Bias parameter (batch last because of x.T in forward)
        self.bias = torch.nn.Parameter(torch.randn(hidden_size, 1)) 

    def _initialize_orthogonal_matrix(self, hidden_size: int) -> None:
        with torch.no_grad():
            # Compute Q^T @ Q - I and get largest singular value
            q_gram = torch.matmul(
                self.Q.weight.transpose(0, 1), 
                self.Q.weight
            ) - torch.eye(hidden_size)
            largest_sv = torch.linalg.svdvals(q_gram)[0]
            
            # Scale weights to improve init
            self.Q.weight *= 0.9 / largest_sv

    def der_act(self, x):
        return 1 - torch.square(self.act(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Sylvester residual block.
        
        Applies the Sylvester transformation: x + Q @ R1 @ tanh(R2 @ Q^T @ x + b)
        and computes the log determinant of the Jacobian.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Transformed tensor of shape (batch_size, input_size)
                - Log determinant of Jacobian of shape (batch_size,)
        """
        # Extract weight matrices with parameterization
        Q = self.Q.weight  # (input_size, hidden_size)
        R1 = self.R1.weight  # (hidden_size, hidden_size)
        R2 = self.R2.weight  # (hidden_size, hidden_size)

        y = torch.matmul(Q.T, x.T)
        y1 = torch.matmul(R2, y) + self.bias
        activated_y1 = self.act(y1)

        y = torch.matmul(R1, activated_y1)
        y = torch.matmul(Q, y)

        dy1 = self.der_act(y1) # (hidden_size, batch_size)

        # Jacobian determinant contribution
        r1_diag = torch.diag(R1).unsqueeze(1)  # (hidden_size, 1)
        r2_diag = torch.diag(R2).unsqueeze(1)  # (hidden_size, 1)
        
        # det(I + diag(R1) * diag(R2) * der_act(y1))
        det_term = 1 + dy1 * r1_diag * r2_diag  # (hidden_size, batch_size)
        
        # Log determinant
        log_det = torch.sum(torch.log(det_term), dim=0)  # (batch_size,)

        return x + y.T, log_det
    
    def extra_repr(self) -> str:
        """
        Return extra representation string for the module.
        
        Returns:
            str: String representation of module parameters.
        """
        return (
            f'hidden_size={self.R1.weight.shape[0]}, '
            f'input_size={self.Q.weight.shape[0]}'
        )