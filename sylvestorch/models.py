"""Sylvester normalizing flow network implementation."""
import warnings

import torch
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List

from .layers import SylvesterBlock, GeneralizedSylvesterBlock


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
    
    def inverse(self, 
                y: torch.Tensor,
                max_iterations: int = 100,
                tolerance: float = 1e-6
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass through the Sylvester network.
        
        Args:
            y (torch.Tensor): Output tensor to invert.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input tensor and log determinant.
        """
        with torch.no_grad():

            inverse_logs = {
                'residual_history': torch.zeros((len(self.blocks), max_iterations,)) * float('nan'),
                'max_iteration_history': torch.zeros((len(self.blocks),))
            }

            x = y

            for b, block in enumerate(self.blocks[::-1]):
                for iteration in range(max_iterations):
                    # Fixed point iteration: x = x + (y - f(x))
                    fx, _ = block.forward(x)
                    error = fx - y
                    error_max = error.abs().max().item()
                    inverse_logs['residual_history'][b, iteration] = error_max

                    if error_max <= tolerance:
                        break
                    else:
                        x = x - error

                inverse_logs['max_iteration_history'][b] = iteration

                if iteration == (max_iterations - 1):
                    warnings.warn(
                        f"Inverse reached maximum iterations {max_iterations} "
                        f"in the {b+1}-th block from the end."
                        f"Final residual: {error.abs().max():.3e}",
                        RuntimeWarning
                    )
                y = x
        return x, inverse_logs

    def fit(self, x: torch.Tensor, 
            epochs: int = 10,
            optimizer: Optional[torch.optim.Optimizer] = None,
            lr: float = 0.01,
            ) -> Dict[str, List[float]]:
        """
        Simple example of a training loop provided as reference.

        Args:
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
    

class GeneralizedSylvesterNet(torch.nn.Module):
  def __init__(self, input_size, n_blocks):
    super(GeneralizedSylvesterNet, self).__init__()
    
    self.blocks = torch.nn.ModuleList()

    for b in range(n_blocks):
      self.blocks.append(GeneralizedSylvesterBlock(input_size))

  def forward(self, x):
    logdet = 0.0
    for block in self.blocks:
      x, ldj = block(x)
      logdet = logdet + ldj
    return x, logdet
  
  def inverse(self, x):
    for block in self.blocks[::-1]:
      x = block.inverse(x)
    return x 