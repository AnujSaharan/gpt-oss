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


class GPUExpertCache:
    """Global GPU LRU cache for dequantized MoE expert weights.

    Keys by (layer_idx, expert_id) and stores a pair (W1, W2).
    """
    def __init__(self, capacity_bytes: int = 1_300_000_000) -> None:
        self.capacity_bytes = int(capacity_bytes)
        self._store: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._order: list[tuple[int, int]] = []  # LRU: front is oldest
        self._used_bytes: int = 0

    @staticmethod
    def _pair_nbytes(W1: torch.Tensor, W2: torch.Tensor) -> int:
        return (W1.numel() * W1.element_size()) + (W2.numel() * W2.element_size())

    def _touch(self, key: tuple[int, int]) -> None:
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.append(key)

    def _evict_if_needed(self) -> None:
        while self._used_bytes > self.capacity_bytes and self._order:
            old = self._order.pop(0)
            try:
                W1, W2 = self._store.pop(old)
            except KeyError:
                continue
            self._used_bytes -= self._pair_nbytes(W1, W2)
            # Help GC promptly; don't force empty_cache every time
            del W1, W2
        # Optional: allow caller to call torch.cuda.empty_cache() if needed

    def reserve(self, bytes_needed: int) -> None:
        """Evict entries until there is room for bytes_needed without exceeding cap."""
        target = self._used_bytes + max(0, int(bytes_needed))
        if target <= self.capacity_bytes:
            return
        # Evict oldest until we have space
        while self._order and self._used_bytes + bytes_needed > self.capacity_bytes:
            old = self._order.pop(0)
            try:
                W1, W2 = self._store.pop(old)
            except KeyError:
                continue
            self._used_bytes -= self._pair_nbytes(W1, W2)
            del W1, W2

    def ensure_present(self, layer_idx: int, expert_id: int, comp: MoECompressed, device: torch.device) -> None:
        key = (layer_idx, expert_id)
        if key in self._store:
            self._touch(key)
            return
        # Avoid illegal ops during CUDA graph capture
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MoE expert cache miss during CUDA graph capture")
        # Pre-evict to avoid OOM during dequant temporary allocations
        # W1: (2I, H), W2: (H, I), BF16 -> 2 bytes each element
        I = int(comp.inter_size)
        H = int(comp.hidden_size)
        bytes_needed = 2 * ((2 * I * H) + (H * I))
        self.reserve(bytes_needed)
        W1 = comp.dequantize_expert(expert_id, "mlp1", device=device)
        W2 = comp.dequantize_expert(expert_id, "mlp2", device=device)
        self._store[key] = (W1, W2)
        self._touch(key)
        self._used_bytes += self._pair_nbytes(W1, W2)
        self._evict_if_needed()

    def get_or_dequantize(self, layer_idx: int, expert_id: int, comp: MoECompressed, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer_idx, expert_id)
        if key in self._store:
            self._touch(key)
            return self._store[key]
        self.ensure_present(layer_idx, expert_id, comp, device)
        return self._store[key]

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


def sdpa_flash(Q, K, V, sm_scale):
    """Use PyTorch SDPA/FlashAttention kernels. Ignores sinks for performance.

    Supports Q length != K/V length (e.g., decoding with cache).
    """
    # Q: (Tq, Hkv, q_mult, Dh); K,V: (Tk, Hkv, Dh)
    Tq, Hkv, qmult, Dh = Q.shape
    Tk = K.shape[0]
    H = Hkv * qmult
    # Expand K,V heads to match q_mult on the head-multiplicity axis only
    K = K[:, :, None, :].expand(Tk, Hkv, qmult, Dh)
    V = V[:, :, None, :].expand(Tk, Hkv, qmult, Dh)
    # Reshape to (B=1, H, T{q|k}, Dh)
    Q = Q.reshape(Tq, H, Dh).transpose(0, 1).unsqueeze(0)
    K = K.reshape(Tk, H, Dh).transpose(0, 1).unsqueeze(0)
    V = V.reshape(Tk, H, Dh).transpose(0, 1).unsqueeze(0)

    # Keep FlashAttention by avoiding dense masks. If lengths differ, we've
    # already ensured K/V only include past tokens, so causal masking is redundant.
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=sm_scale)
    # back to (Tq, H*Dh)
    out = out.squeeze(0).transpose(0, 1).reshape(Tq, -1)
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
            # Apply RoPE w.r.t. running position
            o = int(cache.offset.item())
            q, k = self.rope(q, k, offset=o)
            # Extend cache and build full K/V up to current position
            K_full, V_full = cache.extend(k, v)
            # Sliding window without masks: slice KV to preserve FlashAttention
            if self.sliding_window > 0:
                lo = max(0, o - self.sliding_window + 1)
                hi = o + T
                K_win = K_full[lo:hi, ...]
                V_win = V_full[lo:hi, ...]
            else:
                K_win = K_full[: o + T, ...]
                V_win = V_full[: o + T, ...]
            # SDPA/FlashAttention path (sinks ignored for perf)
            t = sdpa_flash(q, K_win, V_win, self.sm_scale)
        else:
            q, k = self.rope(q, k, offset=0)
            # No cache: full attention over current tokens
            t = sdpa_flash(q, k, v, self.sm_scale)

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
        *,
        layer_idx: int = 0,
        gpu_cache: "GPUExpertCache | None" = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
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
        # Shared GPU cache of dequantized experts across layers
        self.gpu_cache = gpu_cache
        # H2D prefetch stream (if CUDA)
        self.h2d_stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() and (device is None or device.type == 'cuda') else None
        self._prefetch_evt = torch.cuda.Event() if self.h2d_stream is not None else None

    def set_compressed(self, comp: MoECompressed):
        self.compressed = comp

    def _get_expert_weights(self, e: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.compressed is not None, "Compressed MoE weights not set."
        if self.gpu_cache is None:
            # Fallback to per-call dequant without caching (unlikely path)
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("MoE expert fetch during CUDA graph capture without cache")
            W1 = self.compressed.dequantize_expert(e, "mlp1", device=device)
            W2 = self.compressed.dequantize_expert(e, "mlp2", device=device)
            return W1, W2
        return self.gpu_cache.get_or_dequantize(self.layer_idx, e, self.compressed, device)

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
        if self.h2d_stream is not None and self.gpu_cache is not None:
            with torch.cuda.stream(self.h2d_stream):
                for e in uniq.tolist():
                    # Enqueue H2D + dequant on H2D stream via cache
                    self.gpu_cache.ensure_present(self.layer_idx, e, self.compressed, t.device)
                # Record a single event after enqueuing H2D/dequant for this batch of experts
                self._prefetch_evt.record()
            # Wait only on the recorded event instead of whole-stream serialization
            torch.cuda.current_stream().wait_event(self._prefetch_evt)

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

    def forward_probe(self, x: torch.Tensor, record: set[int]) -> torch.Tensor:
        """Record needed experts without performing MoE matmuls.

        Updates `record` with expert ids selected by gate.topk. Returns x unchanged.
        """
        assert self.compressed is not None, "Compressed MoE weights not set."
        t = self.norm(x)
        g = self.gate(t)
        top = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        uniq = torch.unique(top.indices)
        for e in uniq.tolist():
            record.add(int(e))
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
        *,
        gpu_cache: "GPUExpertCache | None" = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, device, layer_idx=layer_idx, gpu_cache=gpu_cache)

    def forward(self, x: torch.Tensor, cache: "Cache" | None = None, probe: list[set[int]] | None = None) -> torch.Tensor:
        x = self.attn(x, cache=cache)
        if probe is not None:
            x = self.mlp.forward_probe(x, probe[self.layer_idx])
        else:
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
        self.unembedding: torch.nn.Parameter | None = None
        # Global GPU expert cache shared across layers
        cache_capacity = int(os.environ.get("GPT_OSS_GPU_EXPERT_CACHE_BYTES", str(1_300_000_000)))
        self.moe_gpu_cache = GPUExpertCache(capacity_bytes=cache_capacity)
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device, gpu_cache=self.moe_gpu_cache)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        # Tie unembedding to embedding to save VRAM
        # We do not allocate a separate unembedding matrix; compute logits via matmul.

    def forward(self, x: torch.Tensor, caches: list["Cache"] | None = None, probe: list[set[int]] | None = None) -> torch.Tensor:
        # x: (T,) token ids (long or int)
        if x.dtype != torch.long:
            x = x.to(torch.long)
        x = self.embedding(x)
        caches = caches or [None] * len(self.block)
        for block, cache in zip(self.block, caches):
            x = block(x, cache=cache, probe=probe)
        x = self.norm(x)
        # Matmul with tied or untied unembedding: (T, H) @ (H, V)
        W_out = self.unembedding if self.unembedding is not None else self.embedding.weight
        x = x @ W_out.t()
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

        # Optional untied unembedding weight
        for possible in ("unembedding.weight", "lm_head.weight"):
            if possible in checkpoint.tensor_name_to_file:
                w = checkpoint.get(possible)
                # Allocate parameter to hold untied unembedding
                model.unembedding = torch.nn.Parameter(w.to(dtype=torch.bfloat16, device=device))
                break

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
        # Sanity toggles for perf on consumer GPUs
        try:
            if device.type == 'cuda':
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass
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
        # CUDA graph setup for fixed 1-token step (disabled by default for MoE safety)
        self._graphs_enabled = bool(int(os.environ.get("GPT_OSS_ENABLE_CUDAGRAPH", "0")))
        self._graph_ready = False
        if self.device.type == 'cuda' and self._graphs_enabled:
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
            # Micro-batch prefill to limit concurrent expert usage and memory
            chunk = int(os.environ.get("GPT_OSS_PREFILL_CHUNK", "128"))
            if chunk <= 0:
                chunk = len(prefill)
            for i in range(0, len(prefill), chunk):
                _ = self.model(prefill[i : i + chunk], caches=self.caches)
            current = int(prompt_tokens[-1])
        else:
            current = int(prompt_tokens[0])

        num_generated_tokens = 0
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            if self.device.type == 'cuda' and self._graphs_enabled:
                # Probe needed experts for this step without heavy MoE compute
                self._input_token[0] = current
                needed: list[set[int]] = [set() for _ in range(len(self.caches))]
                _ = self.model(self._input_token, caches=self.caches, probe=needed)
                # Roll back cache offset by 1 (we wrote K/V for the probe step)
                for c in self.caches:
                    c.offset.sub_(1)
                # Prefetch needed experts on a dedicated stream
                if not hasattr(self, "_prefetch_stream"):
                    self._prefetch_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
                if self.device.type == 'cuda' and self._prefetch_stream is not None:
                    evt = torch.cuda.Event()
                    with torch.cuda.stream(self._prefetch_stream):
                        for layer_idx, experts in enumerate(needed):
                            if not experts:
                                continue
                            comp = self.model.block[layer_idx].mlp.compressed
                            if comp is None:
                                continue
                            for e in experts:
                                self.model.moe_gpu_cache.ensure_present(layer_idx, int(e), comp, self.device)
                        evt.record()
                    torch.cuda.current_stream().wait_event(evt)

                # Capture or replay the 1-token step now that experts are ensured
                if not self._graph_ready:
                    logits_full = self.model(self._input_token, caches=self.caches)
                    self._logits_buf = torch.empty_like(logits_full)
                    torch.cuda.synchronize()
                    with torch.cuda.graph(self._graph):
                        self._logits_buf.copy_(self.model(self._input_token, caches=self.caches))
                    self._graph_ready = True
                    logits = self._logits_buf[-1]
                else:
                    try:
                        self._graph.replay()
                        logits = self._logits_buf[-1]
                    except RuntimeError:
                        logits = self.model(self._input_token, caches=self.caches)[-1]
            else:
                x = torch.as_tensor([current], dtype=torch.long, device=self.device)
                logits = self.model(x, caches=self.caches)[-1]
            # Use float32 logits for numerically stable softmax/log_softmax
            logits = logits.to(dtype=torch.float32)
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
