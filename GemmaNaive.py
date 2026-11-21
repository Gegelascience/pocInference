import json
import time
from CacheKV import KVCache
import numpy as np
from safetensors import safe_open
import sentencepiece as spm
from transcob16f import load_safetensors_bf16_to_fp32


class Gemma3Numpy:

  def __init__(self, model_dir):
    # 1. Chargement de la configuration et du tokenizer SentencePiece
    with open(f"{model_dir}/config.json", "r") as f:
      self.cfg = json.load(f)

    self.sp = spm.SentencePieceProcessor()
    self.sp.load(f"{model_dir}/tokenizer.model")

    # 2. Chargement des poids en float32 via safetensors (sans PyTorch)
    """self.weights = {}
    with safe_open(
        f"{model_dir}/model.safetensors", framework="np", device="cpu"
    ) as f:
      for k in f.keys():
        self.weights[k] = f.get_tensor(k).astype(np.float32)"""

    self.weights = load_safetensors_bf16_to_fp32(
        f"{model_dir}/model.safetensors"
    )

    self.dim = self.cfg["hidden_size"]  # 1152 pour Gemma 3 1B
    self.n_layers = self.cfg["num_hidden_layers"]  # 26 couches
    self.n_heads = self.cfg["num_attention_heads"]  # 4 têtes Q
    self.n_kv_heads = self.cfg["num_key_value_heads"]  # 1 tête KV (MQA)
    self.head_dim = self.cfg["head_dim"]  # 256
    self.eps = self.cfg["rms_norm_eps"]  # 1e-6

  # --- Briques mathématiques de base ---
  def rms_norm(self, x, weight):
    norm = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
    # Spécificité Gemma : le poids appris est centré autour de 0, d'où le (1.0 + weight)
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

  def apply_rope(self, x, seq_len, theta_base):
    # x shape: (seq_len, n_heads, head_dim)
    half_dim = self.head_dim // 2
    freqs = 1.0 / (
        theta_base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim)
    )
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    angles = pos * freqs[None, :]  # (seq_len, half_dim)

    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]

    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

  # --- Passe avant (Forward Pass) ---
  def forward(self, token_ids):
    seq_len = len(token_ids)
    # 1. Embeddings + mise à l'échelle Gemma
    h = self.weights["model.embed_tokens.weight"][token_ids] * np.sqrt(self.dim)

    # Masque causal triangulaire
    mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)

    for i in range(self.n_layers):
      prefix = f"model.layers.{i}"

      # A. Pre-Attention RMSNorm
      h_norm = self.rms_norm(
          h, self.weights[f"{prefix}.input_layernorm.weight"]
      )

      # B. Projections Q, K, V
      q = (h_norm @ self.weights[f"{prefix}.self_attn.q_proj.weight"].T).reshape(
          seq_len, self.n_heads, self.head_dim
      )
      k = (h_norm @ self.weights[f"{prefix}.self_attn.k_proj.weight"].T).reshape(
          seq_len, self.n_kv_heads, self.head_dim
      )
      v = (h_norm @ self.weights[f"{prefix}.self_attn.v_proj.weight"].T).reshape(
          seq_len, self.n_kv_heads, self.head_dim
      )

      # C. QK-Norm (Spécificité Gemma 3)
      q = self.rms_norm(q, self.weights[f"{prefix}.self_attn.q_norm.weight"])
      k = self.rms_norm(k, self.weights[f"{prefix}.self_attn.k_norm.weight"])

      # D. RoPE (Attention globale 1 couche sur 6 avec theta=1M, sinon locale theta=10k)
      is_global = (i + 1) % 6 == 0
      theta = 1000000.0 if is_global else 10000.0
      q = self.apply_rope(q, seq_len, theta)
      k = self.apply_rope(k, seq_len, theta)

      # E. Multi-Query Attention (répétition de la tête K/V pour les 4 têtes Q)
      k = np.repeat(k, self.n_heads // self.n_kv_heads, axis=1)
      v = np.repeat(v, self.n_heads // self.n_kv_heads, axis=1)

      # Calcul des scores d'attention : (n_heads, seq_len, seq_len)
      q_t = q.transpose(1, 0, 2)
      k_t = k.transpose(1, 2, 0)
      v_t = v.transpose(1, 0, 2)

      scale = 1.0 / np.sqrt(self.cfg.get("query_pre_attn_scalar", 256))
      scores = (q_t @ k_t) * scale + mask[None, :, :]
      attn_out = (self.softmax(scores) @ v_t).transpose(1, 0, 2).reshape(seq_len, -1)

      # Post-Attention Norm & Résiduel (Gemma 3 normalise avant d'ajouter le résiduel)
      attn_proj = attn_out @ self.weights[f"{prefix}.self_attn.o_proj.weight"].T
      h = h + self.rms_norm(
          attn_proj, self.weights[f"{prefix}.post_attention_layernorm.weight"]
      )

      # F. MLP (GeGLU)
      mlp_in = self.rms_norm(
          h, self.weights[f"{prefix}.pre_feedforward_layernorm.weight"]
      )
      gate = self.gelu(mlp_in @ self.weights[f"{prefix}.mlp.gate_proj.weight"].T)
      up = mlp_in @ self.weights[f"{prefix}.mlp.up_proj.weight"].T
      mlp_out = (gate * up) @ self.weights[f"{prefix}.mlp.down_proj.weight"].T

      h = h + self.rms_norm(
          mlp_out, self.weights[f"{prefix}.post_feedforward_layernorm.weight"]
      )

    # 3. Norm finale et projection vers le vocabulaire (poids partagés avec embed_tokens)
    h = self.rms_norm(h, self.weights["model.norm.weight"])
    logits = h[-1] @ self.weights["model.embed_tokens.weight"].T
    return logits

  def generate(self, prompt, max_new_tokens=30):
    # Formatage manuel du template de chat Gemma 3
    formatted = (
        f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    )
    tokens = self.sp.encode(formatted, add_bos=False)

    print("Génération : ", end="", flush=True)
    for _ in range(max_new_tokens):
      logits = self.forward(tokens)
      next_token = int(np.argmax(logits))  # Décodage glouton (greedy)

      if next_token == self.sp.eos_id() or next_token == 106:  # <end_of_turn>
        break

      tokens.append(next_token)
      print(self.sp.decode([next_token]), end="", flush=True)
    print()

