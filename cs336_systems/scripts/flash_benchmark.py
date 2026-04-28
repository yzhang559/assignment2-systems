import itertools
import torch
import triton

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attn_triton import MyFlashAttnTritonFunction

DEVICE = "cuda"

SEQ_LENS = [2**i for i in range(7, 17)]   # 128 to 65536
DIMS     = [2**i for i in range(4, 8)]    # 16 to 128
DTYPES   = [torch.bfloat16, torch.float32]


def pytorch_attention(q, k, v, is_causal=True):
    seq_len = q.shape[1]
    mask = None
    if is_causal:
        iota = torch.arange(seq_len, device=q.device)
        mask = iota[:, None] >= iota[None, :]  # (seq, seq)
        mask = mask[None, ...]                  # (1, seq, seq) — broadcast over batch
    return scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)


def bench_fwd(fn, q, k, v):
    return triton.testing.do_bench(lambda: fn(q, k, v, is_causal=True), quantiles=[0.5, 0.2, 0.8])


def bench_bwd(fn, q, k, v, dtype):
    do = torch.randn_like(q)

    def fwd_bwd():
        q.grad = k.grad = v.grad = None
        out = fn(q, k, v, is_causal=True)
        out.backward(do)

    return triton.testing.do_bench(fwd_bwd, quantiles=[0.5, 0.2, 0.8])


def bench_fwd_bwd(fn, q, k, v):
    do = torch.randn_like(q)

    def fwd_bwd():
        q.grad = k.grad = v.grad = None
        out = fn(q, k, v, is_causal=True)
        out.backward(do)

    return triton.testing.do_bench(fwd_bwd, quantiles=[0.5, 0.2, 0.8])


def make_inputs(seq_len, dim, dtype):
    q = torch.randn(1, seq_len, dim, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(1, seq_len, dim, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(1, seq_len, dim, device=DEVICE, dtype=dtype, requires_grad=True)
    return q, k, v


def run_benchmarks():
    header = f"{'impl':<10} {'seq_len':>8} {'dim':>5} {'dtype':<12} {'fwd_ms':>8} {'bwd_ms':>8} {'e2e_ms':>8}"
    print(header)
    print("-" * len(header))

    impls = {
        "triton": MyFlashAttnTritonFunction.apply,
        "torch":  pytorch_attention,
    }

    for seq_len, dim, dtype in itertools.product(SEQ_LENS, DIMS, DTYPES):
        dtype_name = str(dtype).split(".")[-1]

        for impl_name, fn in impls.items():
            if impl_name == "triton" and seq_len > 32768 and dim < 32:
                print(f"{'triton':<10} {seq_len:>8} {dim:>5} {dtype_name:<12} {'SKIP':>8} {'SKIP':>8} {'SKIP':>8}")
                continue

            q, k, v = make_inputs(seq_len, dim, dtype)

            try:
                fwd_ms   = bench_fwd(fn, q, k, v)[0]
            except Exception as e:
                print(f"  [{impl_name} fwd error] {e}")
                fwd_ms = float('nan')

            q, k, v = make_inputs(seq_len, dim, dtype)
            try:
                e2e_ms = bench_fwd_bwd(fn, q, k, v)[0]
                bwd_ms = e2e_ms - fwd_ms
            except Exception as e:
                print(f"  [{impl_name} e2e error] {e}")
                bwd_ms = e2e_ms = float('nan')

            print(f"{impl_name:<10} {seq_len:>8} {dim:>5} {dtype_name:<12} {fwd_ms:>8.3f} {bwd_ms:>8.3f} {e2e_ms:>8.3f}")


if __name__ == "__main__":
    run_benchmarks()