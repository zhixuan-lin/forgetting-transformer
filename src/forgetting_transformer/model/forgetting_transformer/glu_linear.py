import torch
import torch.nn.functional as F


glu_fwd_codestring = """
template <typename T> T glu_fwd(T x, T y) {
    return float(y) / (1.0f + ::exp(-float(x)));
}
"""
glu_bwd_codestring = """
template <typename T> T glu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1.0f - x_sigmoid) * float(g) * float(y);
    dy = x_sigmoid * float(g);
}
"""

glu_bwd_with_output_codestring = """
template <typename T> T glu_bwd_with_output(T x, T y, T g, T& dx, T& dy, T& z) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1.0f - x_sigmoid) * float(g) * float(y);
    dy = x_sigmoid * float(g);
    z = x_sigmoid * float(y);
}
"""

glu_fwd = torch.cuda.jiterator._create_jit_fn(glu_fwd_codestring)
glu_bwd = torch.cuda.jiterator._create_multi_output_jit_fn(glu_bwd_codestring, num_outputs=2)
glu_bwd_with_output = torch.cuda.jiterator._create_multi_output_jit_fn(glu_bwd_with_output_codestring, num_outputs=3)


class GLULinearFunction(torch.autograd.Function):
    r"""
    Gated Linear Unit (GLU) function followed by a linear transformation.

    .. math::
        \text{GLULinear}(x, y, W, b) = (sh(x) * y) W + b

    This simple wrap discards the intermediate results of GLU(x, y) to save memory.
    """

    @staticmethod
    def forward(ctx, x, y, weight, bias):
        z = glu_fwd(x, y)
        out = F.linear(z.to(weight.dtype), weight, bias)
        # We don't store z, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(x, y, weight)
        ctx.linear_bias_is_none = bias is None
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        x, y, weight = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dz = F.linear(dout, weight.t()).view_as(x)
        dx, dy, z = glu_bwd_with_output(x, y, dz)
        dlinear_weight = torch.einsum("bo,bi->oi", dout, z.reshape(-1, z.shape[-1]))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias

glu_linear = GLULinearFunction.apply
