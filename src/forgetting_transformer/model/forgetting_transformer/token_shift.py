import torch

import triton
import triton.language as tl
import pytest

def maybe_contiguous(x):
    # only when the inner most dimension is contiguous can LDGSTS be used
    # so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x

@triton.jit
def shift_fwd_kernel(
    X_PTR,
    PREV_WEIGHT_PTR,
    CURR_WEIGHT_PTR,
    OUT_PTR,

    stride_x_b, stride_x_t, stride_x_h, stride_x_d,
    stride_weight_b, stride_weight_t, stride_weight_h,
    T: tl.constexpr, D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
        everything is (B, T, D)
    """
    b_offset = tl.program_id(axis=0).to(tl.int64)
    t_offset = tl.program_id(axis=1).to(tl.int64) * BLOCK_T
    h_offset = tl.program_id(axis=2).to(tl.int64)


    x_ptr_offset = b_offset * stride_x_b + t_offset * stride_x_t + h_offset * stride_x_h
    X_PTR += x_ptr_offset
    OUT_PTR += x_ptr_offset

    weight_ptr_offset = b_offset * stride_weight_b + t_offset * stride_weight_t + h_offset * stride_weight_h
    CURR_WEIGHT_PTR += weight_ptr_offset
    PREV_WEIGHT_PTR += weight_ptr_offset

    x_ptr = X_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
    t_offset_block = t_offset + tl.arange(0, BLOCK_T)[:, None]
    x_mask = t_offset_block < T

    # Yeah this is correct
    x_prev_ptr = x_ptr - stride_x_t
    t_prev_offset_block = t_offset_block - 1
    x_prev_mask = ((t_prev_offset_block) < T) & (t_prev_offset_block >= 0)

    curr_weight_ptr = CURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
    prev_weight_ptr = PREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t


    x = tl.load(x_ptr, mask=x_mask, other=0.0)
    x_prev = tl.load(x_prev_ptr, mask=x_prev_mask, other=0.0)
    curr_weight = tl.load(curr_weight_ptr, mask=x_mask, other=0.0)
    prev_weight = tl.load(prev_weight_ptr, mask=x_mask, other=0.0)

    result = x * curr_weight.to(tl.float32) + x_prev * prev_weight.to(tl.float32)
    result = result.to(x.dtype)

    out_ptr = OUT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
    tl.store(out_ptr, result, mask=x_mask)


@triton.jit
def shift_bwd_kernel(
    X_PTR,
    PREV_WEIGHT_PTR,
    CURR_WEIGHT_PTR,

    DOUT_PTR,
    DX_PTR,
    DPREV_WEIGHT_PTR,
    DCURR_WEIGHT_PTR,

    stride_x_b, stride_x_t, stride_x_h, stride_x_d,
    stride_weight_b, stride_weight_t, stride_weight_h,
    T: tl.constexpr, D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
        everything is (B, T, D)
    """
    b_offset = tl.program_id(axis=0).to(tl.int64)
    t_offset = tl.program_id(axis=1).to(tl.int64) * BLOCK_T
    h_offset = tl.program_id(axis=2).to(tl.int64)


    x_ptr_offset = b_offset * stride_x_b + t_offset * stride_x_t + h_offset * stride_x_h
    X_PTR += x_ptr_offset
    DX_PTR += x_ptr_offset
    DOUT_PTR += x_ptr_offset

    weight_ptr_offset = b_offset * stride_weight_b + t_offset * stride_weight_t + h_offset * stride_weight_h
    CURR_WEIGHT_PTR += weight_ptr_offset
    PREV_WEIGHT_PTR += weight_ptr_offset
    DCURR_WEIGHT_PTR += weight_ptr_offset
    DPREV_WEIGHT_PTR += weight_ptr_offset

    x_ptr = X_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
    t_offset_block = t_offset + tl.arange(0, BLOCK_T)[:, None]
    x_mask = t_offset_block < T

    dout_ptr = DOUT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d

    # Yeah this is correct
    dout_next_ptr = dout_ptr + stride_x_t
    t_next_offset_block = t_offset_block + 1
    x_next_mask = (t_next_offset_block) < T


    # Yeah this is correct
    x_prev_ptr = x_ptr - stride_x_t
    t_prev_offset_block = t_offset_block - 1
    x_prev_mask = ((t_prev_offset_block) < T) & (t_prev_offset_block >= 0)

    curr_weight_ptr = CURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
    prev_weight_ptr = PREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
    next_prev_weight_ptr = prev_weight_ptr + stride_weight_t


    x = tl.load(x_ptr, mask=x_mask, other=0.0)
    x_prev = tl.load(x_prev_ptr, mask=x_prev_mask, other=0.0)
    dout = tl.load(dout_ptr, mask=x_mask, other=0.0)
    dout_next= tl.load(dout_next_ptr, mask=x_next_mask, other=0.0)

    curr_weight = tl.load(curr_weight_ptr, mask=x_mask, other=0.0)
    next_prev_weight = tl.load(next_prev_weight_ptr, mask=x_next_mask, other=0.0)

    dx =  dout * curr_weight.to(tl.float32) + dout_next * next_prev_weight.to(tl.float32)
    dx = dx.to(x.dtype)

    dcurr_weight = tl.sum(dout.to(tl.float32) * x, axis=1, keep_dims=True)
    dprev_weight = tl.sum(dout.to(tl.float32) * x_prev, axis=1, keep_dims=True)

    dx_ptr = DX_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
    tl.store(dx_ptr, dx, mask=x_mask)
    dcurr_weight_ptr = DCURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
    tl.store(dcurr_weight_ptr, dcurr_weight, mask=x_mask)
    dprev_weight_ptr = DPREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
    tl.store(dprev_weight_ptr, dprev_weight, mask=x_mask)



class TokenShift(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, prev_weight: torch.Tensor, curr_weight: torch.Tensor):

        B, T, H, D = x.size()
        assert D in {16, 32, 64, 128}
        assert prev_weight.size() == curr_weight.size() == (B, T, H)
        assert prev_weight.stride() == curr_weight.stride()
        x = maybe_contiguous(x)
        out = torch.empty_like(x)

        BLOCK_T = triton.next_power_of_2(min(64, T))

        grid = lambda meta: (B, triton.cdiv(T, meta["BLOCK_T"]), H)
        # NOTE:
        #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
        #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
        #  - Don't forget to pass meta-parameters as keywords arguments.
        shift_fwd_kernel[grid](
            x,
            prev_weight,
            curr_weight,
            out,
            *x.stride(),
            *curr_weight.stride(),
            T=T, D=D,
            BLOCK_T=BLOCK_T,
        )
        ctx.save_for_backward(x, prev_weight, curr_weight)
        # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
        # running asynchronously at this point.
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):

        x, prev_weight, curr_weight = ctx.saved_tensors
        B, T, H, D = x.size()
        assert D in {16, 32, 64, 128}
        assert prev_weight.size() == curr_weight.size() == (B, T, H)
        assert prev_weight.stride() == curr_weight.stride()
        x = maybe_contiguous(x)
        assert dout.stride() == x.stride()
        dx = torch.empty_like(x)
        dcurr_weight = torch.empty_like(curr_weight)
        dprev_weight = torch.empty_like(prev_weight)

        BLOCK_T = triton.next_power_of_2(min(64, T))

        grid = lambda meta: (B, triton.cdiv(T, meta["BLOCK_T"]), H)
        # NOTE:
        #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
        #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
        #  - Don't forget to pass meta-parameters as keywords arguments.
        shift_bwd_kernel[grid](
            x,
            prev_weight,
            curr_weight,
            dout,
            dx,
            dprev_weight,
            dcurr_weight,
            *x.stride(),
            *curr_weight.stride(),
            T=T,
            D=D,
            BLOCK_T=BLOCK_T,
        )
        # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
        # running asynchronously at this point.
        return dx, dprev_weight, dcurr_weight

def token_shift(x, prev_weight, curr_weight):
    return TokenShift.apply(x, prev_weight, curr_weight)



@pytest.mark.parametrize("B, T, H, D", [(4, 2048, 12, 128)])
def test_op(B, T, H, D, dtype=torch.float32):
    torch.manual_seed(24)
    B = 4
    T = 2088
    H = 12
    D = 128
    # x = torch.rand(size, device='cuda')
    x = torch.randn(B, T, H, D, device="cuda", dtype=dtype, requires_grad=True)
    dout = torch.randn(B, T, H, D, device="cuda", dtype=dtype)
    curr_weight = torch.rand(B, T, H, device="cuda", requires_grad=True)

    prev_weight = 1.0 - curr_weight
    x_prev = torch.roll(x, shifts=1, dims=1)
    x_prev[:, 0, :, :] = 0.0
    ref_out = (x_prev * prev_weight[..., None] + x * curr_weight[..., None]).to(dtype)

    ref_out.backward(dout)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dcurr_weight, curr_weight.grad = curr_weight.grad.clone(), None


    prev_weight = 1.0 - curr_weight
    # out_torch = x if x.sum() > 0.0 else y

    tri_out = token_shift(x, prev_weight, curr_weight)


    tri_out.backward(dout)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dcurr_weight, curr_weight.grad = curr_weight.grad.clone(), None

    # out_torch = x if x.sum() > 0.0 else y

    # import pdb; pdb.set_trace()

    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0), (ref_out - tri_out).abs().max()
    assert torch.allclose(ref_dx, tri_dx, atol=1e-2, rtol=0), (ref_dx - tri_dx).abs().max()
    assert torch.allclose(ref_dcurr_weight, tri_dcurr_weight, atol=1e-2, rtol=0), (ref_dcurr_weight - tri_dcurr_weight).abs().max()

if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    T = 2088
    H = 12
    D = 128
    # x = torch.rand(size, device='cuda')
    x = torch.randn(B, T, H, D, device="cuda")
    dout = torch.randn(B, T, H, D, device="cuda")
    curr_weight = torch.rand(B, T, H, device="cuda")
    prev_weight = 1.0 - curr_weight
    # out_torch = x if x.sum() > 0.0 else y
    result = shift_fwd(x, prev_weight, curr_weight)
    print(result[0, :, 0, 0])
    import ipdb; ipdb.set_trace()
    # # for mode in ["fwd", "bwd"]:
    # configs.append(
    #     triton.testing.Benchmark(
    #         x_names=["SIZE"],
    #         # x_vals=[2**i for i in range(10, 15)],
    #         x_vals=[98432],
    #         line_arg="provider",
    #         # line_vals=["triton-fp16", "flag"] + (["flash"] if HAS_FLASH else []),
    #         # line_names=["Triton [FP16]", "Flag"] + (["Flash-2"] if HAS_FLASH else []),
    #         line_vals=["debug"],
    #         line_names=["Debug"],
    #         styles=[("red", "-")],
    #         ylabel="ms",
    #         plot_name="hi",
    #         args={},
    #     )
    # )


    # @triton.testing.perf_report(configs)
    # def bench_flash_attention(SIZE, provider, device="cuda"):
    #     warmup = 25
    #     rep = 100
    #     torch.manual_seed(0)
    #     size = 98432
    #     # x = torch.rand(size, device='cuda')
    #     x = torch.ones(size, device="cuda")
    #     y = torch.rand(size, device="cuda")
    #     # out_torch = x if x.sum() > 0.0 else y
    #     fn = lambda: add(x, y)
    #     ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    #     return ms


    # if __name__ == "__main__":
    #     # only works on post-Ampere GPUs right now
    #     bench_flash_attention.run(save_path=".", print_data=True)
