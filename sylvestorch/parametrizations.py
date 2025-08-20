"""
Parametrization modules for the Sylvester library.
"""

import warnings

import torch
from typing import Callable


class Triu(torch.nn.Module):
    """
    Applies upper triangular masking to the weight tensors.
    """
    
    def __init__(self, diagonal: int) -> None:
        super(Triu, self).__init__()
        self.diagonal = diagonal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.triu(self.diagonal)


class Tril(torch.nn.Module):
    """
    Applies lower triangular masking to the weight tensors.
    """
    
    def __init__(self, diagonal: int) -> None:
        super(Tril, self).__init__()
        self.diagonal = diagonal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.tril(self.diagonal)


class Upper(torch.nn.Module):
    """
    A multi-head upper triangular transformation module.
    
    This module applies upper triangular masking to different sections of the
    input tensor, treating each section as a separate head in a multi-head
    architecture. Each head processes a slice of the input tensor independently.
    
    Args:
        out_s (int): Output size per head (number of rows per head slice).
        n_heads (int): Number of attention heads.
        d (int): Diagonal offset for triangular masking. d=0 is the main diagonal,
                d>0 is above the main diagonal, d<0 is below the main diagonal.
    
    Example:
        >>> upper_layer = Upper(out_s=64, n_heads=8, d=1)
        >>> x = torch.randn(512, 64)  # 8 heads * 64 = 512 rows
        >>> output = upper_layer(x)
    """
    
    def __init__(self, out_s: int, n_heads: int, d: int) -> None:
        super(Upper, self).__init__()
        self.n_heads = n_heads
        self.out_s = out_s
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head upper triangular transformation.
        
        The input tensor is split into n_heads sections, each of size out_s.
        Upper triangular masking is applied to each section independently,
        then all sections are concatenated back together.
        
        Args:
            x (torch.Tensor): Input tensor of shape (n_heads * out_s, features).
            
        Returns:
            torch.Tensor: Transformed tensor with upper triangular masking
                         applied to each head section, same shape as input.
        """
        w_list = []
        
        for h in range(self.n_heads):
            # Extract the slice for head h
            start_idx = h * self.out_s
            end_idx = (h + 1) * self.out_s
            head_matrix = x[start_idx:end_idx, :]
            
            # Apply upper triangular masking
            masked_matrix = head_matrix.triu(self.d)
            
            # Note: The commented line below could be used for alternating signs
            # masked_matrix = masked_matrix * (2 * (h % 2) - 1)
            
            w_list.append(masked_matrix)
        
        # Concatenate all heads back together
        w = torch.cat(w_list, dim=0)
        
        # Note: The commented line below could be used to set zeros to large negative values
        # This is sometimes useful for attention mechanisms
        # w[w == 0.0] = -1000
        
        return w


class DiagAct(torch.nn.Module):
    """
    Diagonal activation module that applies the activation function only to the 
    diagonal elements of the input tensor.
    
    Args:
        length (int): Dimension of the square matrix (number of rows/columns).
    
    Attributes:
        activation (callable): Activation function (torch.tanh).
        diag_idx (torch.Tensor): Indices for diagonal elements.
    
    Example:
        >>> diag_act = DiagAct(d=128)
        >>> x = torch.randn(128, 128)
        >>> output = diag_act(x)  # Diagonal elements transformed, others unchanged
    """
    
    def __init__(
            self, 
            length: int, 
            activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh
    ) -> None:
        super(DiagAct, self).__init__()
        self.act = activation
        self.diag_idx = torch.arange(0, length, dtype=torch.long)

    def forward(self, x):
        x[self.diag_idx, self.diag_idx] = self.act(torch.diag(x))
        return x
    
class Orthogonalize(torch.nn.Module):
    """
    Orthogonalization module using iterative Newton-Schulz method.
    
    This module orthogonalizes input matrices using the Newton-Schulz iteration,
    which iteratively refines a matrix to make its columns orthonormal.
    The algorithm stops when the orthogonality constraint is satisfied within
    the specified tolerance or when the maximum number of steps is reached.
    
    Args:
        d (int): Number of rows in the input matrix.
        m (int): Number of columns in the input matrix.
        steps (int, optional): Maximum number of iteration steps. Defaults to 200.
        eps (float, optional): Convergence tolerance. Defaults to 1e-6.
    
    Attributes:
        m (int): Number of columns.
        d (int): Number of rows.
        I (torch.Tensor): Identity matrix of size (m, m), registered as buffer.
        eps (float): Convergence tolerance.
        steps (int): Maximum number of iteration steps.
    
    Example:
        >>> ortho = Orthogonalize(d=128, m=64)
        >>> x = torch.randn(128, 64)
        >>> x_ortho = ortho(x)  # Orthogonalized matrix
    """
    
    def __init__(
        self, 
        d: int, 
        m: int, 
        steps: int = 200, 
        eps: float = 1e-6
    ) -> None:
        super(Orthogonalize, self).__init__()
        self.m = m
        self.d = d
        self.eps = eps
        self.steps = steps
        self.register_buffer('I', torch.eye(m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalize the input matrix using Newton-Schulz iteration.
        
        The Newton-Schulz method iteratively refines the matrix to satisfy
        the orthogonality constraint X^T @ X = I, where I is the identity matrix.
        
        Args:
            x (torch.Tensor): Input matrix of shape (d, m) to be orthogonalized.
            
        Returns:
            torch.Tensor: Orthogonalized matrix of the same shape as input.
            
        Raises:
            RuntimeError: If input tensor dimensions don't match expected shape.
            
        Note:
            If convergence is not achieved within the specified number of steps,
            a warning will be issued showing the final residual.
        """
        # Validate input dimensions
        if x.shape != (self.d, self.m):
            raise RuntimeError(
                f"Expected input shape ({self.d}, {self.m}), "
                f"but got {x.shape}"
            )
        
        # Optional: Normalize columns initially (currently commented out)
        # x_ortho = x_ortho / torch.linalg.norm(x_ortho, dim=0, keepdim=True)
        
        for step in range(self.steps):
            final_step = step
            
            # Newton-Schulz iteration step
            x_t = x.transpose(0, 1)
            gram_matrix = torch.matmul(x_t, x)
            correction = self.I + 0.5 * (self.I - gram_matrix)
            x = torch.matmul(x, correction)
            
            # Check convergence
            with torch.no_grad():
                orthogonality_error = self.I - torch.matmul(
                    x.transpose(0, 1), x
                )
                residual = torch.sum(torch.square(orthogonality_error)).item()
                
                if residual < self.eps:
                    break
        
        # Issue warning if convergence was not achieved
        if residual > self.eps:
            warnings.warn(
                f"Orthogonalization did not converge after {final_step + 1} "
                f"steps. Final residual: {residual:.2e}",
                RuntimeWarning
            )
        
        return x

    def extra_repr(self) -> str:
        """
        Return extra representation string for the module.
        
        Returns:
            str: String representation of module parameters.
        """
        return f'd={self.d}, m={self.m}, steps={self.steps}, eps={self.eps}'