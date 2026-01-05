import pytest
import torch

from sylvestorch.parametrizations import (
    Triu, 
    Tril, 
    Upper, 
    DiagAct, 
    Orthogonalize
)


@pytest.mark.parametrize("diag", [-1, 0, 1])
def test_triu_matches_tensor_triu(diag):
    x = torch.randn(5, 5)
    module = Triu(diagonal=diag)

    out = module(x.clone())
    expected = x.triu(diag)

    assert torch.allclose(out, expected)

@pytest.mark.parametrize("diag", [-1, 0, 1])
def test_tril_matches_tensor_tril(diag):
    x = torch.randn(5, 5)
    module = Tril(diagonal=diag)

    out = module(x.clone())
    expected = x.tril(diag)

    assert torch.allclose(out, expected)



# ----------------------------
# DiagAct
# ----------------------------

@pytest.mark.parametrize("length", [1, 4, 6, 13])
def test_diagact_only_modifies_diagonal(length):
    torch.manual_seed(0)
    module = DiagAct(length=length, activation=torch.tanh)

    x = torch.randn(length, length)
    x_original = x.clone()

    out = module(x)

    # Off-diagonal entries should be unchanged
    mask_offdiag = ~torch.eye(length, dtype=torch.bool)
    assert torch.allclose(out[mask_offdiag], x_original[mask_offdiag])

    # Diagonal entries should be activation(diag(x_original))
    diag_before = torch.diag(x_original)
    diag_after = torch.diag(out)
    expected_diag = torch.tanh(diag_before)

    assert torch.allclose(diag_after, expected_diag, atol=1e-6, rtol=1e-6)

@pytest.mark.parametrize("length", [1, 4, 6, 13])
def test_diagact_custom_activation(length):
    # Use a simple linear activation for test
    def custom_act(t):
        return 2 * t

    length = 4
    module = DiagAct(length=length, activation=custom_act)

    x = torch.randn(length, length)
    x_original = x.clone()

    out = module(x)

    expected_diag = 2 * torch.diag(x_original)
    assert torch.allclose(torch.diag(out), expected_diag, atol=1e-6, rtol=1e-6)

@pytest.mark.parametrize("length", [1, 4, 6, 13])
def test_diag_act_inplace_behavior(length):
    """
    The provided implementation modifies x in-place:
    `x[self.diag_idx, self.diag_idx] = ...`
    This test verifies that behavior.
    """
    module = DiagAct(length=length)
    x = torch.randn(length, length)
    original_x = x.clone()
    
    out = module(x)
    
    # Ensure it returns the modified tensor
    assert out is x 
    # Ensure diagonal changed
    assert not torch.equal(torch.diag(out), torch.diag(original_x))


# ----------------------------
# Orthogonalize
# ----------------------------

def gram_residual(x: torch.Tensor) -> float:
    """Helper: ||X^T X - I||_F^2."""
    m = x.shape[1]
    I = torch.eye(m, dtype=x.dtype, device=x.device)
    diff = torch.matmul(x.T , x) - I
    return torch.sum(diff ** 2).item()


@pytest.mark.parametrize("d,m", [(3, 3), (4, 3), (1000, 133), (1000, 1000)])
def test_orthogonalize_output_shape_and_orthogonality(d, m):
    torch.manual_seed(0)
    ortho = Orthogonalize(d=d, m=m, steps=200, eps=1e-6)

    x = torch.randn(d, m)
    x_gram = torch.matmul(
                x.transpose(0, 1), 
                x
            ) - torch.eye(m)
    largest_sv = torch.linalg.svdvals(x_gram)[0]

    # Scale weights to improve init
    x *= 0.9 / largest_sv

    x_ortho = ortho(x)

    # Shape preserved
    assert x_ortho.shape == (d, m)

    # Check that columns are close to orthonormal: X^T X ≈ I
    res = gram_residual(x_ortho)
    assert res < ortho.eps

def test_orthogonalize_invalid_shape_raises():
    d, m = 5, 3
    ortho = Orthogonalize(d=d, m=m)

    bad_x = torch.randn(d + 1, m)  # wrong first dim

    with pytest.raises(RuntimeError):
        _ = ortho(bad_x)

@pytest.mark.parametrize("d,m", [(3, 3), (4, 3), (1000, 900), (1000, 1000)])
def test_orthogonalize_gradients(d, m):
    """Test that we can backprop through the iterative process."""
    ortho = Orthogonalize(d=d, m=m)

    x = torch.randn(d, m, requires_grad=True)
    with torch.no_grad():
        x_gram = torch.matmul(
                    x.transpose(0, 1), 
                    x
                ) - torch.eye(m)
        largest_sv = torch.linalg.svdvals(x_gram)[0]

        # Scale weights to improve init
        x *= 0.9 / largest_sv
        
    out = ortho(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()