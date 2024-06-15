import json
from dataclasses import dataclass
from typing import TypeAlias, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn

Device: TypeAlias = str | torch.device | None


@dataclass
class Mamba2Config:
    d_model: int
    n_layer: int
    vocab_size: int
    pad_vocab_size_multiple: int = 16
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    chunk_size: int = 64
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class Mamba2LMHeadModel(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(args.vocab_size, args.d_model, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2(args, device=device),
                                norm=RMSNorm(args.d_model, device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )
        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device
        )
        self.lm_head.weight = self.backbone.embedding.weight

    @staticmethod
    def from_pretrained(huggingface_model_id: str, device: Device = None):
        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file

        config_path = cached_file(huggingface_model_id, CONFIG_NAME)
        assert config_path, "Failed to get huggingface config file"
        state_dict_path = cached_file(huggingface_model_id, WEIGHTS_NAME)
        assert state_dict_path, "Failed to get huggingface state dict file"

        config = json.load(open(config_path))
        args = Mamba2Config(
            d_model=config["d_model"],
            n_layer=config["n_layer"],
            vocab_size=config["vocab_size"],
            pad_vocab_size_multiple=config["pad_vocab_size_multiple"],
        )

        map_location = "cpu" if device is None else device
        state_dict = torch.load(
            state_dict_path, weights_only=True, map_location=map_location, mmap=True
        )
        model = Mamba2LMHeadModel(args, device=device)
        model.load_state_dict(state_dict)
        return model

    def forward(self, input_ids: LongTensor) -> LongTensor:
        """
        input_ids: (batch, seqlen)
            tokens from `EleutherAI/gpt-neox-20b` tokenizer
        Returns logits (batch, seqlen, vocab_size)
        """
        _batch, seqlen = input_ids.shape

        # Pad sequence to multiples of `chunk_size`
        chunk_excess = input_ids.shape[1] % self.args.chunk_size
        if chunk_excess != 0:
            input_ids = cast(
                LongTensor, F.pad(input_ids, (0, self.args.chunk_size - chunk_excess))
            )

        h = self.backbone.embedding(input_ids)
        for layer in self.backbone.layers:
            h = layer.mixer(layer.norm(h)) + h

        h = self.backbone.norm_f(h)
        logits = self.lm_head(h)
        return logits[:, :seqlen]


class Mamba2(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.ngroups * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=args.bias, device=device)

        conv_dim = args.d_inner + 2 * args.ngroups * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))
        self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))
        self.D = nn.Parameter(torch.empty(args.nheads, device=device))
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(
            args.d_inner, args.d_model, bias=args.bias, device=device
        )

    def forward(self, u):
        """
        u: (batch, seqlen, d_model)
        Returns (batch, seqlen, d_model)
        """
        _batch, seqlen, _d_model = u.shape

        A = -torch.exp(self.A_log)  # (nheads,) or (d_inner, d_state)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.ngroups * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)
        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen, :]
        )  # (batch, seqlen, d_inner + 2 * ngroups * d_state))
        x, B, C = torch.split(
            xBC,
            [
                self.args.d_inner,
                self.args.ngroups * self.args.d_state,
                self.args.ngroups * self.args.d_state,
            ],
            dim=-1,
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, _h = ssd(
            x,
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.args.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.args.ngroups),
            self.D,
            self.args.chunk_size,
            device=self.device,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        y = self.norm(y, z)
        return self.out_proj(y)


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """More stable segment sum calculation.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, dt, A, B, C, D, chunk_size, initial_states=None, device: Device = None):
    """
    x: (batch, seqlen, n_heads, d_head)
    dt: (batch, seqlen, n_heads)
    A: (batch, seqlen, n_heads)
    B: (batch, seqlen, n_heads, d_state)
    C: (batch, seqlen, n_heads, d_state)
    D: (nheads,)
    Returns (batch, seqlen, n_heads, d_head)

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    assert x.shape[1] % chunk_size == 0

    xD = x * D.unsqueeze(-1)

    A = A * dt
    x = x * dt.unsqueeze(-1)

    # Rearrange into chunks
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    Y = Y + xD

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * F.sigmoid(x)
