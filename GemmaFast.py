import json
import time
from CacheKV import KVCache
from GemmaTokener import Gemma3Tokenizer
import numpy as np

from transcob16f import load_safetensors_bf16_to_fp32


class Gemma3NumpyFast:

  def __init__(self, model_dir):
    with open(f"{model_dir}/config.json", "r") as f:
      self.cfg = json.load(f)

    #self.sp = spm.SentencePieceProcessor()
    #self.sp.load(f"{model_dir}/tokenizer.model")
    self.tokenizer = Gemma3Tokenizer(f"{model_dir}/tokenizer.json")

    print("Chargement des poids...")
    self.weights = load_safetensors_bf16_to_fp32(
        f"{model_dir}/model.safetensors"
    )

    self.dim = self.cfg["hidden_size"]
    self.n_layers = self.cfg["num_hidden_layers"]
    self.n_heads = self.cfg["num_attention_heads"]
    self.n_kv_heads = self.cfg["num_key_value_heads"]
    self.head_dim = self.cfg["head_dim"]
    self.eps = self.cfg["rms_norm_eps"]
    self.scale = 1.0 / np.sqrt(self.cfg.get("query_pre_attn_scalar", 256))

  def create_cache(self, max_seq_len=2048):
    """Crée une instance de KVCache dimensionnée pour ce modèle."""
    return KVCache(self.n_layers, max_seq_len, self.n_kv_heads, self.head_dim)

  def rms_norm(self, x, weight):
    norm = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
    return (x / norm) * (1.0 + weight)

  def gelu(self, x):
    return (
        0.5
        * x
        * (
            1.0
            + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3)))
        )
    )

  def softmax(self, x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

  def apply_rope(self, x, start_pos, theta_base):
    seq_len = x.shape[0]
    half_dim = self.head_dim // 2
    freqs = 1.0 / (
        theta_base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim)
    )
    pos = np.arange(start_pos, start_pos + seq_len, dtype=np.float32)[:, None]
    angles = pos * freqs[None, :]

    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

  def forward(self, token_ids, cache: KVCache):
    seq_len = len(token_ids)
    start_pos = cache.seq_len  # Position actuelle lue directement depuis le cache

    h = self.weights["model.embed_tokens.weight"][token_ids] * np.sqrt(self.dim)
    mask = (
        np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        if seq_len > 1
        else None
    )

    for i in range(self.n_layers):
      prefix = f"model.layers.{i}"

      # 1. Projections Q, K, V
      h_norm = self.rms_norm(
          h, self.weights[f"{prefix}.input_layernorm.weight"]
      )
      q = (h_norm @ self.weights[f"{prefix}.self_attn.q_proj.weight"].T).reshape(
          seq_len, self.n_heads, self.head_dim
      )
      k = (h_norm @ self.weights[f"{prefix}.self_attn.k_proj.weight"].T).reshape(
          seq_len, self.n_kv_heads, self.head_dim
      )
      v = (h_norm @ self.weights[f"{prefix}.self_attn.v_proj.weight"].T).reshape(
          seq_len, self.n_kv_heads, self.head_dim
      )

      # 2. QK-Norm & RoPE
      q = self.rms_norm(q, self.weights[f"{prefix}.self_attn.q_norm.weight"])
      k = self.rms_norm(k, self.weights[f"{prefix}.self_attn.k_norm.weight"])

      theta = 1000000.0 if (i + 1) % 6 == 0 else 10000.0
      q = self.apply_rope(q, start_pos, theta)
      k = self.apply_rope(k, start_pos, theta)

      # 3. Mise à jour propre via la classe KVCache
      k, v = cache.update(i, k, v)

      # 4. Multi-Query Attention
      k_rep = np.repeat(k, self.n_heads // self.n_kv_heads, axis=1)
      v_rep = np.repeat(v, self.n_heads // self.n_kv_heads, axis=1)

      scores = (q.transpose(1, 0, 2) @ k_rep.transpose(1, 2, 0)) * self.scale
      if mask is not None:
        scores += mask[None, :, :]

      attn_out = (
          (self.softmax(scores) @ v_rep.transpose(1, 0, 2))
          .transpose(1, 0, 2)
          .reshape(seq_len, -1)
      )
      attn_proj = attn_out @ self.weights[f"{prefix}.self_attn.o_proj.weight"].T
      h = h + self.rms_norm(
          attn_proj, self.weights[f"{prefix}.post_attention_layernorm.weight"]
      )

      # 5. MLP (GeGLU)
      mlp_in = self.rms_norm(
          h, self.weights[f"{prefix}.pre_feedforward_layernorm.weight"]
      )
      gate = self.gelu(mlp_in @ self.weights[f"{prefix}.mlp.gate_proj.weight"].T)
      up = mlp_in @ self.weights[f"{prefix}.mlp.up_proj.weight"].T
      mlp_out = (gate * up) @ self.weights[f"{prefix}.mlp.down_proj.weight"].T
      h = h + self.rms_norm(
          mlp_out, self.weights[f"{prefix}.post_feedforward_layernorm.weight"]
      )

    # On fait avancer le pointeur du cache une fois toutes les couches terminées
    cache.advance(seq_len)

    h = self.rms_norm(h, self.weights["model.norm.weight"])
    return h[-1] @ self.weights["model.embed_tokens.weight"].T

  def generate(self, prompt, max_new_tokens=60):
    formatted = (
        f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    )
    tokens = self.tokenizer.encode(formatted, add_bos=False)

    # Instanciation du cache
    cache = self.create_cache(max_seq_len=len(tokens) + max_new_tokens)

    # Prefill
    t0 = time.time()
    logits = self.forward(tokens, cache)
    print(f"[Prefill : {cache.seq_len} tokens en {time.time() - t0:.2f}s]")

    # Decode
    print("Gemma : ", end="", flush=True)
    for _ in range(max_new_tokens):
      next_token = int(np.argmax(logits))
      if next_token == self.tokenizer.eos_token_id or next_token == 106:
        break

      print(self.tokenizer.decode([next_token]), end="", flush=True)
      logits = self.forward([next_token], cache)
    print()