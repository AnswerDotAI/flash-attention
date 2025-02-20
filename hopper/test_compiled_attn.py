import torch
import pytest
from padding import unpad_input
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seqlen", [16, 32, 64])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("num_heads", [2, 8, 16])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flash_attn_compiled_vs_uncompiled(batch_size, seqlen, mha_type, num_heads, d, local, causal, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    device = "cuda"

    nheads_k = num_heads if mha_type == "mha" else (1 if mha_type == "mqa" else num_heads // 2)
    q = torch.randn(batch_size, seqlen, num_heads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads_k, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads_k, d, device=device, dtype=dtype, requires_grad=True)

    q_compiled = q.detach().clone()
    k_compiled = k.detach().clone()
    v_compiled = v.detach().clone()
    q_compiled.requires_grad = True
    k_compiled.requires_grad = True
    v_compiled.requires_grad = True

    window_size = (-1, -1) if not local else torch.randint(0, seqlen, (2,))

    out_uncompiled, _ = flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    compiled_flash_attn_func = torch.compile(flash_attn_func)
    out_compiled, _ = compiled_flash_attn_func(q_compiled, k_compiled, v_compiled, causal=causal, window_size=window_size)

    forward_diff = (out_uncompiled - out_compiled).abs().max().item()
    assert torch.allclose(out_uncompiled, out_compiled, atol=1e-3, rtol=1e-3), (
        f"Forward outputs differ for causal={causal}; maximum diff = {forward_diff}"
    )

    grad_output = torch.randn_like(out_uncompiled) * 0.1
    grad_compiled = grad_output.detach().clone()

    out_uncompiled.backward(grad_output)
    out_compiled.backward(grad_compiled)

    atol = 2e-3  # Base tolerance for bfloat16/float16
    rtol = 2e-3

    if local:  # Even higher for local attention
        atol *= 2
        rtol *= 2

    assert torch.allclose(q.grad, q_compiled.grad, atol=atol, rtol=rtol), (
        f"Gradients for q differ; max diff = {(q.grad - q_compiled.grad).abs().max().item()}"
    )
    assert torch.allclose(k.grad, k_compiled.grad, atol=atol, rtol=rtol), (
        f"Gradients for k differ; max diff = {(k.grad - k_compiled.grad).abs().max().item()}"
    )
    assert torch.allclose(v.grad, v_compiled.grad, atol=atol, rtol=rtol), (
        f"Gradients for v differ; max diff = {(v.grad - v_compiled.grad).abs().max().item()}"
    )


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seqlen", [16, 32, 64])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("num_heads", [2, 8, 16])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flash_attn_varlen_compiled_vs_uncompiled(batch_size, seqlen, mha_type, num_heads, d, local, causal, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    device = "cuda"

    inputs = torch.randn(batch_size, seqlen, 1, device=device, dtype=dtype)
    valid_lengths = torch.randint(1, seqlen + 1, (batch_size,))
    attn_mask = torch.zeros((batch_size, seqlen), device=device, dtype=torch.bool)
    for i, valid_len in enumerate(valid_lengths):
        attn_mask[i, :valid_len] = True
    inputs, indices, cu_seqlens, max_seqlen_in_batch, seqused = unpad_input(inputs, attention_mask=attn_mask)

    nheads_k = num_heads if mha_type == "mha" else (1 if mha_type == "mqa" else num_heads // 2)
    q = torch.randn(cu_seqlens[-1], num_heads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(cu_seqlens[-1], nheads_k, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(cu_seqlens[-1], nheads_k, d, device=device, dtype=dtype, requires_grad=True)

    q_compiled = q.detach().clone()
    k_compiled = k.detach().clone()
    v_compiled = v.detach().clone()
    q_compiled.requires_grad = True
    k_compiled.requires_grad = True
    v_compiled.requires_grad = True

    window_size = (-1, -1) if not local else torch.randint(0, seqlen, (2,))

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
        window_size=window_size,
    )

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
        window_size=window_size,
    )

    forward_diff = (out_uncompiled - out_compiled).abs().max().item()
    assert torch.allclose(out_uncompiled, out_compiled, atol=1e-3, rtol=1e-3), (
        f"Forward outputs differ for causal={causal}; maximum diff = {forward_diff}"
    )

    grad_output = torch.randn_like(out_uncompiled) * 0.1
    grad_compiled = grad_output.detach().clone()

    out_uncompiled.backward(grad_output)
    out_compiled.backward(grad_compiled)

    atol = 2e-3  # Base tolerance for bfloat16/float16
    rtol = 2e-3

    if local:  # Even higher for local attention
        atol *= 2
        rtol *= 2

    assert torch.allclose(q.grad, q_compiled.grad, atol=atol, rtol=rtol), (
        f"Gradients for q differ; max diff = {(q.grad - q_compiled.grad).abs().max().item()}"
    )
    assert torch.allclose(k.grad, k_compiled.grad, atol=atol, rtol=rtol), (
        f"Gradients for k differ; max diff = {(k.grad - k_compiled.grad).abs().max().item()}"
    )
    assert torch.allclose(v.grad, v_compiled.grad, atol=atol, rtol=rtol), (
        f"Gradients for v differ; max diff = {(v.grad - v_compiled.grad).abs().max().item()}"
    )
