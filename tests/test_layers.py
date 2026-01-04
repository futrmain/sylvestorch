import pytest
import torch

from sylvestorch.layers import SylvesterBlock

@pytest.mark.parametrize("input_size,hidden_size,ortho_eps", [(4, 3, 1e-6), (8, 8, 1e-3), (123, 13, 1e-4)])
def test_initialization(input_size, hidden_size, ortho_eps):
    block = SylvesterBlock(input_size=input_size, hidden_size=hidden_size, ortho_eps=ortho_eps)
    
    assert block.input_size == input_size
    assert block.hidden_size == hidden_size
    assert block.ortho_eps == ortho_eps
    assert block.act == torch.tanh
    
    # Check layer dimensions
    assert block.R1.weight.shape == (hidden_size, hidden_size)
    assert block.R2.weight.shape == (hidden_size, hidden_size)
    assert block.Q.weight.shape == (input_size, hidden_size)
    assert block.bias.shape == (hidden_size, 1)


@pytest.mark.parametrize("batch,input_size,hidden_size,ortho_eps", [(1000, 4, 3, 1e-6), (10, 8, 8, 1e-3), (30, 123, 13, 1e-4)])
def test_forward_shapes_and_finite(batch, input_size, hidden_size, ortho_eps):
    block = SylvesterBlock(input_size=input_size, hidden_size=hidden_size, ortho_eps=ortho_eps)
    x = torch.randn(batch, input_size)

    y, log_det = block(x)

    assert y.shape == (batch, input_size)
    assert log_det.shape == (batch,)
    assert torch.isfinite(y).all()
    assert torch.isfinite(log_det).all()
    assert not torch.isnan(y).any()
    assert not torch.isnan(log_det).any()

    assert y.requires_grad
    assert log_det.requires_grad

    # Check that it's not identity (with high probability)
    assert not torch.allclose(y, x)


@pytest.mark.parametrize("batch,input_size,hidden_size,ortho_eps", [(1000, 4, 3, 1e-6), (10, 8, 8, 1e-6), (30, 123, 13, 1e-6)])
def test_log_det_matches_autograd_jacobian_small_dim(batch, input_size, hidden_size, ortho_eps):
    def jacobian(function, input):
        x = input.detach().requires_grad_(True)
        y, logdet = function(x)
            
        J = torch.zeros(x.shape[0], x.shape[1], x.shape[1], dtype=x.dtype) # batch x out_s x in_s
        auto_grad = torch.autograd.functional.jacobian(function, x)[0]
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                J[i,j,:] = auto_grad[i, j, i, :]
        return J, logdet
    
    
    block = SylvesterBlock(input_size=input_size, hidden_size=hidden_size, ortho_eps=ortho_eps).double()

    x = torch.randn(batch, input_size, dtype=torch.double, requires_grad=True)

    # Analytic output
    J, logdet = jacobian(block.forward, x)
    detJ = torch.log(torch.abs(torch.det(J)))

    # The flow should be orientation-preserving often, but abs keeps this robust.
    assert torch.isfinite(detJ).all()
    assert torch.allclose(detJ, logdet, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("batch,input_size,hidden_size,ortho_eps", [(1000, 4, 3, 1e-6), (100, 8, 8, 1e-3), (300, 123, 13, 1e-4)])
def test_inverse(batch, input_size, hidden_size, ortho_eps):
    block = SylvesterBlock(input_size=input_size, hidden_size=hidden_size, ortho_eps=ortho_eps)
    x = torch.randn(batch, input_size)

    y, log_det = block(x)
    z = y
    for iteration in range(100):
        fx, _ = block.forward(z)
        error = fx - y
        error_max = error.abs().max().item()
        
        if error_max <= 1e-6:
            break
        else:
            z = z - error

    assert z.shape == x.shape
    assert torch.isfinite(z).all()
    assert not torch.isnan(z).any()
    assert error_max <= 1e-6, f"Error max: {error_max}"
    assert iteration < 99, f"Number of iterations: {iteration}"
    assert torch.allclose(x, z, atol=1e-6), f"Inverse not equal to original, Number of iterations: {iteration}, Error max: {error_max}"