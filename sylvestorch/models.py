"""Sylvester normalizing flow network implementation."""

import torch
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional, List

from .layers import SylvesterBlock


class SylvesterNet(torch.nn.Module):
    def __init__(
        self, 
        n_layers: int, 
        input_size: int, 
        hidden_size: int, 
        ortho_eps: float = 1e-6
    ) -> None:
        """
        Sylvester normalizing flow network.
        
        This module implements a stack of Sylvester normalizing flow blocks to create
        a more expressive normalizing flow model. Each block applies an invertible
        transformation with tractable Jacobian determinant, and the total log determinant
        is computed as the sum of individual block contributions.
        
        Args:
            n_layers (int): Number of Sylvester blocks to stack.
            input_size (int): Dimension of input data.
            hidden_size (int): Dimension of hidden representations.
            ortho_eps (float, optional): Orthogonalization tolerance. 
                Defaults to 1e-6.
        
        Example:
            >>> net = SylvesterNet(n_layers=4, input_size=32, hidden_size=64)
            >>> x = torch.randn(100, 32)  # batch_size=100, input_size=32
            >>> y, log_det = net(x)
        """
        
        super(SylvesterNet, self).__init__()
        
        self.blocks = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                SylvesterBlock(input_size, hidden_size, ortho_eps)
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize log determinant accumulator
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        
        # Apply each block sequentially
        for block in self.blocks:
            x, block_logdet = block(x)
            logdet = logdet + block_logdet
        
        return x, logdet
    
    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass through the Sylvester network.
        
        Note: This method is not implemented yet. For a complete normalizing flow,
        the inverse transformation would apply the blocks in reverse order.
        
        Args:
            y (torch.Tensor): Output tensor to invert.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input tensor and log determinant.
            
        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError(
            "Inverse transformation not implemented. "
            "This would require implementing inverse operations for each block."
        )

    def fit(self, x: torch.Tensor, 
            epochs: int = 10,
            optimizer: Optional[torch.optim.Optimizer] = None,
            lr: float = 0.01,
            ) -> Dict[str, List[float]]:
        """
        Simple example of a training loop provided as reference.

        Parameters:
            x (torch.Tensor): Input data tensor to train on.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            optimizer (torch.optim.Optimizer, optional): PyTorch optimizer to use for training.
            lr (float, optional): Learning rate to use if creating a new optimizer. Defaults to 0.01.
    
        Returns:
            Dict[str, List[float]]: Epoch-wise mean square & logdet.
        """
        
        # Set default optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            optimizer = optimizer(self.parameters(), lr=lr)

        training_logs = {
            'meansquare' : torch.zeros((epochs,)),
            'logdet' : torch.zeros((epochs,))
        }
        
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()

            y, ld = self.forward(x)

            meansquare = y.square().mean()
            logdet = ld.square().mean()

            loss = meansquare + logdet
            loss.backward()
            optimizer.step()

            training_logs['meansquare'][epoch] = meansquare.item()
            training_logs['logdet'][epoch] = logdet.item()

        return training_logs