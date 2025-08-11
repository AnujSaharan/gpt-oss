from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass

import torch
try:
    import torch.distributed as dist  # type: ignore
except Exception:
    # Minimal shim when torch.distributed isn't available
    class _DummyDist:
        @staticmethod
        def is_initialized() -> bool:
            return False

        @staticmethod
        def get_world_size() -> int:
            return 1

        @staticmethod
        def get_rank() -> int:
            return 0

        class ReduceOp:
            SUM = None

        @staticmethod
        def all_reduce(*args, **kwargs):
            raise RuntimeError("torch.distributed is not available")

    dist = _DummyDist()  # type: ignore

from gpt_oss.torch_impl.weights import Checkpoint, MoECompressed
import torch.nn.functional as F


@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int, start: int = 0):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(start, start + num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens, start=offset)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa_flash(Q, K, V, sm_scale, sliding_window=0):
    """Use PyTorch SDPA/FlashAttention kernels. Ignores sinks for performance."""
    # Q: (T, Hkv, q_mult, Dh); K,V: (T, Hkv, Dh)
    T, Hkv, qmult, Dh = Q.shape
    H = Hkv * qmult
    # Expand K,V heads to match q_mult
    K = K[:, :, None, :].expand(T, Hkv, qmult, Dh)
    V = V[:, :, None, :].expand(T, Hkv, qmult, Dh)
    # Reshape to (B=1, H, T, Dh)
    Q = Q.reshape(T, H, Dh).transpose(0, 1).unsqueeze(0)
    K = K.reshape(T, H, Dh).transpose(0, 1).unsqueeze(0)
    V = V.reshape(T, H, Dh).transpose(0, 1).unsqueeze(0)

    attn_mask = None
    if sliding_window > 0:
        m = Q.new_full((T, T), float('-inf'))
        tri = torch.triu(m, diagonal=1)
        band = torch.tril(m, diagonal=-sliding_window)
        attn_mask = tri + band

    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, is_causal=True, scale=sm_scale)
    # back to (T, H*Dh)
    out = out.squeeze(0).transpose(0, 1).reshape(T, -1)
    return out


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor, cache: "Cache" | None = None) -> torch.Tensor:
        # x: (T, C) unbatched sequence
        T, _ = x.shape
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()

        # Reshape into heads
        q = q.view(
            T,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(T, self.num_key_value_heads, self.head_dim)
        v = v.view(T, self.num_key_value_heads, self.head_dim)

        # Apply RoPE
        if cache is not None:
            q, k = self.rope(q, k, offset=int(cache.offset.item()))
            # Extend cache and build full K/V up to current position
            K_full, V_full = cache.extend(k, v)
            # SDPA/FlashAttention path (sinks ignored for perf)
            t = sdpa_flash(q, K_full, V_full, self.sm_scale, self.sliding_window)
        else:
            q, k = self.rope(q, k, offset=0)
            t = sdpa_flash(q, k, v, self.sm_scale, self.sliding_window)

        t = self.out(t)
        t = x + t
        return t


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )
        assert config.intermediate_size % self.world_size == 0
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        # Set by from_checkpoint: per-layer compressed MoE weights on CPU
        self.compressed: MoECompressed | None = None
        # Simple device-side LRU cache for dequantized experts
        self._expert_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._expert_cache_order: list[int] = []
        self._expert_cache_capacity: int = 24
        # H2D prefetch stream (if CUDA)
        self.h2d_stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() and (device is None or device.type == 'cuda') else None

    def set_compressed(self, comp: MoECompressed):
        self.compressed = comp
        self._expert_cache.clear()
        self._expert_cache_order.clear()

    def _get_expert_weights(self, e: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if e in self._expert_cache:
            # update LRU order
            try:
                self._expert_cache_order.remove(e)
            except ValueError:
                pass
            self._expert_cache_order.append(e)
            return self._expert_cache[e]
        e_tensor = torch.tensor([e], device=device)
        W1 = self.compressed.dequantize_experts(e_tensor, "mlp1", device=device)[0]
        W2 = self.compressed.dequantize_experts(e_tensor, "mlp2", device=device)[0]
        # insert into cache
        self._expert_cache[e] = (W1, W2)
        self._expert_cache_order.append(e)
        if len(self._expert_cache_order) > self._expert_cache_capacity:
            old = self._expert_cache_order.pop(0)
            try:
                del self._expert_cache[old]
            except KeyError:
                pass
            torch.cuda.empty_cache()
        return W1, W2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.compressed is not None, "Compressed MoE weights not set."
        t = self.norm(x)  # (T, H)
        g = self.gate(t)  # (T, E)
        top = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        probs = torch.softmax(top.values, dim=-1)  # (T, k)
        idx = top.indices  # (T, k)

        T, H = t.shape
        k = self.experts_per_token
        E = self.num_experts
        I = self.compressed.inter_size

        # Group tokens by expert to avoid duplicating weights per token
        # Build mapping e -> list of token positions and their weight index
        uniq = torch.unique(idx)
        out = x.new_zeros((T, H))

        # Pre-fetch experts asynchronously to overlap H2D/dequant with compute
        if self.h2d_stream is not None:
            with torch.cuda.stream(self.h2d_stream):
                for e in uniq.tolist():
                    _ = self._get_expert_weights(e, t.device)
            torch.cuda.current_stream().wait_stream(self.h2d_stream)

        # For small T, simply loop over unique experts
        for e in uniq.tolist():
            # Find where this expert is used
            mask = (idx == e)
            if not mask.any():
                continue
            # Tokens selecting this expert and the corresponding mixture weights
            tok_pos, which = mask.nonzero(as_tuple=True)  # arrays of indices
            t_tok = t.index_select(0, tok_pos)  # (N, H)
            w_mix = probs[tok_pos, which]       # (N,)

            # Dequantize this expert's weights on device
            W1, W2 = self._get_expert_weights(e, t.device)
            b1 = self.mlp1_bias[e]  # (2I,)
            b2 = self.mlp2_bias[e]  # (H,)

            # First MLP: (N,H) @ (H,2I) -> (N,2I)
            x1 = torch.matmul(t_tok, W1.t()) + b1
            x1 = swiglu(x1, limit=self.swiglu_limit)  # (N, I)
            # Second MLP: (N,I) @ (I,H) -> (N,H)
            x2 = torch.matmul(x1, W2.t()) + b2

            # Mixture combine for these tokens
            out.index_add_(0, tok_pos, x2 * w_mix[:, None])

        if self.world_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)

        return x + out


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, device)

    def forward(self, x: torch.Tensor, cache: "Cache" | None = None) -> torch.Tensor:
        x = self.attn(x, cache=cache)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor, caches: list["Cache"] | None = None) -> torch.Tensor:
        # x: (T,) token ids (long or int)
        if x.dtype != torch.long:
            x = x.to(torch.long)
        x = self.embedding(x)
        caches = caches or [None] * len(self.block)
        for block, cache in zip(self.block, caches):
            x = block(x, cache=cache)
        x = self.norm(x)
        x = self.unembedding(x)
        return x

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "Transformer":
        if not isinstance(device, torch.device):
            device = torch.device(device)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        model = Transformer(
            config=config,
            device=device,
        )
        model.eval()

        # Load weights
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_rank_intermediate_size = config.intermediate_size // world_size

        checkpoint = Checkpoint(path, device)
        cpu_checkpoint = Checkpoint(path, torch.device("cpu"))

        for name, param in model.named_parameters():
            loaded_tensor = checkpoint.get(name)

            # Note: it would be more efficient to do sharding before upcasting from MXFP4,
            # but for simplicity we do it after.
            if "mlp1" in name:  # both weight and bias
                loaded_tensor = loaded_tensor[
                    :,
                    my_rank * 2
                    * per_rank_intermediate_size : (my_rank + 1) * 2
                    * per_rank_intermediate_size,
                    ...,
                ]
            elif "mlp2_weight" in name:  # only weight
                loaded_tensor = loaded_tensor[
                    ...,
                    my_rank
                    * per_rank_intermediate_size : (my_rank + 1)
                    * per_rank_intermediate_size,
                ]
            try:
                param.data.copy_(loaded_tensor)
            except:
                print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
                raise

        # Attach compressed MoE weights per layer (CPU resident)
        for layer_idx in range(config.num_hidden_layers):
            # Use PARAM_NAME_MAP in weights.Checkpoint
            from gpt_oss.torch_impl.weights import PARAM_NAME_MAP
            mlp1_blocks_key, mlp1_scales_key = PARAM_NAME_MAP[f"block.{layer_idx}.mlp.mlp1_weight"]
            mlp2_blocks_key, mlp2_scales_key = PARAM_NAME_MAP[f"block.{layer_idx}.mlp.mlp2_weight"]

            mlp1_blocks_cpu = cpu_checkpoint._get_tensor(mlp1_blocks_key).pin_memory()
            mlp1_scales_cpu = cpu_checkpoint._get_tensor(mlp1_scales_key).pin_memory()
            mlp2_blocks_cpu = cpu_checkpoint._get_tensor(mlp2_blocks_key).pin_memory()
            mlp2_scales_cpu = cpu_checkpoint._get_tensor(mlp2_scales_key).pin_memory()

            comp = MoECompressed(
                mlp1_blocks_cpu,
                mlp1_scales_cpu,
                mlp2_blocks_cpu,
                mlp2_scales_cpu,
                inter_size=config.intermediate_size // (dist.get_world_size() if dist.is_initialized() else 1),
                hidden_size=config.hidden_size,
            )
            model.block[layer_idx].mlp.set_compressed(comp)

        return model


class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)
        # Allocate per-layer KV caches for incremental decoding
        config = ModelConfig(**json.load(open(os.path.join(checkpoint, "config.json"))))
        self.caches = [
            Cache(
                n_ctx=config.initial_context_length,
                n_kv_heads=config.num_key_value_heads,
                d_head=config.head_dim,
                device=self.device,
            )
            for _ in range(config.num_hidden_layers)
        ]
        # CUDA graph setup for fixed 1-token step
        self._graph_ready = False
        if self.device.type == 'cuda':
            self._input_token = torch.zeros(1, dtype=torch.long, device=self.device)
            self._logits_buf = None
            self._graph = torch.cuda.CUDAGraph()

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int],
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        # Reset caches
        for c in self.caches:
            c.reset()

        # Prefill with all but last token to populate caches
        if len(prompt_tokens) > 1:
            prefill = torch.as_tensor(prompt_tokens[:-1], dtype=torch.long, device=self.device)
            _ = self.model(prefill, caches=self.caches)
            current = int(prompt_tokens[-1])
        else:
            current = int(prompt_tokens[0])

        num_generated_tokens = 0
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            if self.device.type == 'cuda':
                # Capture the 1-token step on first iteration
                self._input_token[0] = current
                if not self._graph_ready:
                    # Warmup allocation for logits buffer
                    logits_full = self.model(self._input_token, caches=self.caches)
                    self._logits_buf = torch.empty_like(logits_full)
                    torch.cuda.synchronize()
                    with torch.cuda.graph(self._graph):
                        self._logits_buf.copy_(self.model(self._input_token, caches=self.caches))
                    self._graph_ready = True
                    logits = self._logits_buf[-1]
                else:
                    self._graph.replay()
                    logits = self._logits_buf[-1]
            else:
                x = torch.as_tensor([current], dtype=torch.long, device=self.device)
                logits = self.model(x, caches=self.caches)[-1]
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            num_generated_tokens += 1
            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                yield next_token, logprobs[next_token].item()
            else:
                yield next_token

            if next_token in stop_tokens:
                break
            current = next_token

        return


class Cache:
    def __init__(self, n_ctx: int, n_kv_heads: int, d_head: int, device: torch.device | None = None):
        self.k = torch.zeros((n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.v = torch.zeros((n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.offset = torch.zeros((1,), dtype=torch.long, device=device)

    def reset(self):
        self.k.zero_()
        self.v.zero_()
        self.offset.zero_()

    def extend(self, k: torch.Tensor, v: torch.Tensor):
        # k,v: (T, n_kv_heads, d_head)
        T = k.shape[0]
        start = int(self.offset.item())
        end = start + T
        self.k[start:end, :, :].copy_(k)
        self.v[start:end, :, :].copy_(v)
        self.offset.add_(T)
        # Return full K/V up to current end
        return self.k[:end, :, :], self.v[:end, :, :]
