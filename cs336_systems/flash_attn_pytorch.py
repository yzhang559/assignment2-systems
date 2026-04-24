import math

import torch.autograd


def _flash_backward(q, k, v, O, dO, L, is_causal=False):
    d = q.shape[-1]

    D = torch.sum(dO * O, dim=-1)
    S = q @ k.transpose(-1, -2) * (d ** -0.5)
    if is_causal:
        nq, nk = q.shape[1], k.shape[1]
        iota = torch.arange(nq, device=q.device)
        mask = iota[:, None] >= torch.arange(nk, device=q.device)[None, :]
        S = S.masked_fill(~mask[None, ...], float('-inf'))
    P = torch.exp(S - L.unsqueeze(-1))

    dV = P.transpose(-1, -2) @ dO
    dP = dO @ v.transpose(-1, -2)
    dS = P * (dP - D.unsqueeze(-1))

    dQ = dS @ k * (d ** -0.5)
    dK = dS.transpose(-1, -2) @ q * (d ** -0.5)
    return dQ, dK, dV


_flash_backward_compiled = torch.compile(_flash_backward)


class MyFlashAttnAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # tile size
        Bq = Bk = 16
        batch_size = q.shape[0]
        Nq = q.shape[1]
        Nk = k.shape[1]

        d = q.shape[-1]

        num_of_q_tiles = math.ceil(Nq / Bq)
        num_of_k_tiles = math.ceil(Nk / Bk)

        O = torch.zeros_like(q)
        L = torch.zeros(batch_size, Nq, dtype=q.dtype, device=q.device)

        for i in range(num_of_q_tiles):
            Q_i = q[:, i * Bq: (i + 1) * Bq, :]
            m = torch.full((batch_size, Bq), float('-inf'), dtype=q.dtype, device=q.device)
            l = torch.zeros((batch_size, Bq), dtype=q.dtype, device=q.device)
            O_i = torch.zeros((batch_size, Bq, d), dtype=q.dtype, device=q.device)

            for j in range(num_of_k_tiles):
                K_j = k[:, j * Bk: (j + 1) * Bk, :]
                V_j = v[:, j * Bk: (j + 1) * Bk, :]
                # print("Q_i shape", Q_i.shape)
                # print("K_j shape", K_j.shape)
                S_i_j = Q_i @ K_j.transpose(-1, -2) * (d ** -0.5)
                m_old = m.clone()
                m = torch.max(m, S_i_j.max(dim=-1).values)
                P_i_j = torch.exp(S_i_j - m.unsqueeze(-1))
                l = torch.exp(m_old - m) * l + P_i_j.sum(dim=-1)

                correction = torch.exp(m_old - m).unsqueeze(-1)
                O_i = correction * O_i + P_i_j @ V_j

            O_i = O_i / l.unsqueeze(-1)
            L_i = m + torch.log(l)
            O[:, i * Bq:(i + 1) * Bq, :] = O_i
            L[:, i * Bq:(i + 1) * Bq] = L_i

        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, O, L)
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, O, L = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_compiled(q, k, v, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
