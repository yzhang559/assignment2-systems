import torch.autograd
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(Q_ptr, K_ptr, V_ptr,
                     O_ptr, L_ptr,
                     stride_qb, stride_qq, stride_qd,
                     stride_kb, stride_kk, stride_kd,
                     stride_vb, stride_vk, stride_vd,
                     stride_ob, stride_oq, stride_od,
                     stride_lb, stride_lq,
                     N_QUERIES, N_KEYS,
                     scale,
                     D: tl.constexpr,
                     Q_TILE_SIZE: tl.constexpr,
                     K_TILE_SIZE: tl.constexpr, ):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        s_i_j = tl.dot(q, k.T) * scale

        m_old = m
        m = tl.maximum(m, tl.max(s_i_j, axis=-1))
        p_i_j = tl.exp(s_i_j - m[:, None])

        l = tl.exp(m_old - m) * l + tl.sum(p_i_j, axis=-1)

        p_i_j = p_i_j.to(V_block_ptr.type.element_ty)
        o = tl.exp(m_old - m)[:, None] * o + tl.dot(p_i_j, v)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = o / l[:, None]
    o = o.to(O_block_ptr.type.element_ty)
    l = m + tl.log(l)

    tl.store(O_block_ptr, o, boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))


class MyFlashAttnTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        batch_size = q.shape[0]
        N_QUERIES, N_KEYS = q.shape[1], k.shape[1]
        D = q.shape[2]
        scale = D ** -0.5

        assert q.is_cuda and k.is_cuda and v.is_cuda, "Expected CUDA tensors"
        assert q.is_contiguous(), "Our pointer arithmetic will assume contiguous q"

        ctx.K_TILE_SIZE = 16
        ctx.Q_TILE_SIZE = 16

        O = torch.empty_like(q)
        L = torch.empty(q.shape[0], q.shape[1], device=q.device, dtype=torch.float32)

        flash_fwd_kernel[(triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), batch_size)](
            q, k, v, O, L,
            stride_qb=q.stride(0), stride_qq=q.stride(1), stride_qd=q.stride(2),
            stride_kb=k.stride(0), stride_kk=k.stride(1), stride_kd=k.stride(2),
            stride_vb=v.stride(0), stride_vk=v.stride(1), stride_vd=v.stride(2),
            stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
            stride_lb=L.stride(0), stride_lq=L.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS, scale=scale, D=D, Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE
        )

        ctx.save_for_backward(q, k, v, O, L)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError
