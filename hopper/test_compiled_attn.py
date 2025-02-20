import torch
import pytest
from padding import unpad_input
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func


# # We parameterize over a few options: batch_size, seqlen, num_heads, feature dimension (d) and causal mode.
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seqlen", [16, 32, 64])
@pytest.mark.parametrize("num_heads", [2, 8, 16])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flash_attn_compiled_vs_uncompiled(batch_size, seqlen, num_heads, d, causal, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    device = "cuda"

    # Construct random query, key and value tensors.
    q = torch.randn(batch_size, seqlen, num_heads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, num_heads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, num_heads, d, device=device, dtype=dtype, requires_grad=True)

    q_compiled = q.detach().clone()
    k_compiled = k.detach().clone()
    v_compiled = v.detach().clone()
    q_compiled.requires_grad = True
    k_compiled.requires_grad = True
    v_compiled.requires_grad = True

    # Run the flash attention (uncompiled) function.
    out_uncompiled, _ = flash_attn_func(q, k, v, causal=causal)

    # Compile the flash attention function.
    compiled_flash_attn_func = torch.compile(flash_attn_func)
    out_compiled, _ = compiled_flash_attn_func(q_compiled, k_compiled, v_compiled, causal=causal)

    # Compare the forward outputs.
    forward_diff = (out_uncompiled - out_compiled).abs().max().item()
    assert torch.allclose(out_uncompiled, out_compiled, atol=1e-3, rtol=1e-3), (
        f"Forward outputs differ for causal={causal}; maximum diff = {forward_diff}"
    )

    # Run backward pass using a random gradient tensor.
    grad_output = torch.randn_like(out_uncompiled) * 0.1
    grad_compiled = grad_output.detach().clone()

    out_uncompiled.backward(grad_output)
    out_compiled.backward(grad_compiled)

    # Compare the gradients.
    assert torch.allclose(q.grad, q_compiled.grad, atol=1e-3, rtol=1e-3), (
        f"Gradients for q differ; max diff = {(q.grad - q_compiled.grad).abs().max().item()}"
    )
    assert torch.allclose(k.grad, k_compiled.grad, atol=1e-3, rtol=1e-3), (
        f"Gradients for k differ; max diff = {(k.grad - k_compiled.grad).abs().max().item()}"
    )
    assert torch.allclose(v.grad, v_compiled.grad, atol=1e-3, rtol=1e-3), (
        f"Gradients for v differ; max diff = {(v.grad - v_compiled.grad).abs().max().item()}"
    )


# We parameterize over a few options: batch_size, seqlen, num_heads, feature dimension (d) and causal mode.
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seqlen", [16, 32, 64])
@pytest.mark.parametrize("num_heads", [2, 8, 16])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flash_attn_varlen_compiled_vs_uncompiled(batch_size, seqlen, num_heads, d, causal, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    device = "cuda"
    torch.random.manual_seed(0)

    inputs = torch.randn(batch_size, seqlen, 1, device=device, dtype=dtype)
    valid_lengths = torch.randint(1, seqlen + 1, (batch_size,))
    attn_mask = torch.zeros((batch_size, seqlen), device=device, dtype=torch.bool)
    for i, valid_len in enumerate(valid_lengths):
        attn_mask[i, :valid_len] = True
    inputs, indices, cu_seqlens, max_seqlen_in_batch, seqused = unpad_input(inputs, attention_mask=attn_mask)

    # Construct random query, key and value tensors.
    q = torch.randn(cu_seqlens[-1], num_heads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(cu_seqlens[-1], num_heads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(cu_seqlens[-1], num_heads, d, device=device, dtype=dtype, requires_grad=True)

    # raise ValueError(f"{inputs.shape=}, {cu_seqlens=}, {q.shape=}")

    q_compiled = q.detach().clone()
    k_compiled = k.detach().clone()
    v_compiled = v.detach().clone()
    q_compiled.requires_grad = True
    k_compiled.requires_grad = True
    v_compiled.requires_grad = True

    # Run the flash attention (uncompiled) function.
    out_uncompiled, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen_in_batch,
        max_seqlen_k=max_seqlen_in_batch,
        seqused_q=seqused,
        seqused_k=seqused,
        causal=causal,
    )

    # Compile the flash attention function.
    compiled_flash_attn_func = torch.compile(flash_attn_varlen_func)
    out_compiled, _ = compiled_flash_attn_func(
        q_compiled,
        k_compiled,
        v_compiled,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen_in_batch,
        max_seqlen_k=max_seqlen_in_batch,
        seqused_q=seqused,
        seqused_k=seqused,
        causal=causal,
    )

    # Compare the forward outputs.
    forward_diff = (out_uncompiled - out_compiled).abs().max().item()
    assert torch.allclose(out_uncompiled, out_compiled, atol=1e-3, rtol=1e-3), (
        f"Forward outputs differ for causal={causal}; maximum diff = {forward_diff}"
    )

    # Run backward pass using a random gradient tensor.
    grad_output = torch.randn_like(out_uncompiled) * 0.1
    grad_compiled = grad_output.detach().clone()

    out_uncompiled.backward(grad_output)
    out_compiled.backward(grad_compiled)

    # Compare the gradients.
    assert torch.allclose(q.grad, q_compiled.grad, atol=1e-3, rtol=1e-3), (
        f"Gradients for q differ; max diff = {(q.grad - q_compiled.grad).abs().max().item()}"
    )
    assert torch.allclose(k.grad, k_compiled.grad, atol=1e-3, rtol=1e-3), (
        f"Gradients for k differ; max diff = {(k.grad - k_compiled.grad).abs().max().item()}"
    )
    assert torch.allclose(v.grad, v_compiled.grad, atol=1e-3, rtol=1e-3), (
        f"Gradients for v differ; max diff = {(v.grad - v_compiled.grad).abs().max().item()}"
    )
