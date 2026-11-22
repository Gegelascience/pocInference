import time
from CacheKV import KVCache
from GemmaTokener import Gemma3Tokenizer
import numpy as np

from utils import StreamDecoder, load_safetensors_bf16_to_fp32, sample_token, ModelConfig


class Gemma3NumpyFast:

  def __init__(self, model_dir:str):

    self.modelConfig =ModelConfig(f"{model_dir}/config.json")

    self.tokenizer = Gemma3Tokenizer(f"{model_dir}/tokenizer.json")

    print("Chargement des poids...")
    self.weights = load_safetensors_bf16_to_fp32(
        f"{model_dir}/model.safetensors"
    )


  def create_cache(self, max_seq_len:int=2048):
    """Crée une instance de KVCache dimensionnée pour ce modèle."""
    return KVCache(self.modelConfig.n_layers, max_seq_len, self.modelConfig.n_kv_heads, self.modelConfig.head_dim)

  def rms_norm(self, x, weight):
    norm = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.modelConfig.eps)
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
    half_dim = self.modelConfig.head_dim // 2
    freqs = 1.0 / (
        theta_base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim)
    )
    pos = np.arange(start_pos, start_pos + seq_len, dtype=np.float32)[:, None]
    angles = pos * freqs[None, :]

    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

  def forward(self, token_ids:list[int], cache: KVCache):
    seq_len = len(token_ids)
    start_pos = cache.seq_len  # Position actuelle lue directement depuis le cache

    h = self.weights["model.embed_tokens.weight"][token_ids] * np.sqrt(self.modelConfig.dim)

    mask = None
    if seq_len > 1:
      # Partie gauche (tokens déjà en cache) : 0.0 (tout est visible)
      past_mask = np.zeros((seq_len, start_pos), dtype=np.float32)
      # Partie droite (nouveaux tokens) : masque triangulaire supérieur à -1e9
      curr_mask = np.triu(
          np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1
      )
      # Concaténation -> shape (seq_len, start_pos + seq_len)
      mask = np.concatenate([past_mask, curr_mask], axis=1)

    for i in range(self.modelConfig.n_layers):
      prefix = f"model.layers.{i}"

      # 1. Projections Q, K, V
      h_norm = self.rms_norm(
          h, self.weights[f"{prefix}.input_layernorm.weight"]
      )
      q = (h_norm @ self.weights[f"{prefix}.self_attn.q_proj.weight"].T).reshape(
          seq_len, self.modelConfig.n_heads, self.modelConfig.head_dim
      )
      k = (h_norm @ self.weights[f"{prefix}.self_attn.k_proj.weight"].T).reshape(
          seq_len, self.modelConfig.n_kv_heads, self.modelConfig.head_dim
      )
      v = (h_norm @ self.weights[f"{prefix}.self_attn.v_proj.weight"].T).reshape(
          seq_len, self.modelConfig.n_kv_heads, self.modelConfig.head_dim
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
      k_rep = np.repeat(k, self.modelConfig.n_heads // self.modelConfig.n_kv_heads, axis=1)
      v_rep = np.repeat(v, self.modelConfig.n_heads // self.modelConfig.n_kv_heads, axis=1)

      scores = (q.transpose(1, 0, 2) @ k_rep.transpose(1, 2, 0)) * self.modelConfig.scale
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

  def generate(self, 
        prompt:str,
        max_new_tokens:int=60,
        temperature:float=0.7,
        top_k:int=50,
        top_p:float=0.9,
        repetition_penalty:float=1.1):
    formatted = (
        f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    )
    tokens = self.tokenizer.encode(formatted, add_bos=False)
    all_tokens = list(tokens)  # Historique pour la pénalité de répétition

    # Instanciation du cache
    cache = self.create_cache(max_seq_len=len(tokens) + max_new_tokens)

    # Prefill
    t0 = time.perf_counter()
    logits = self.forward(tokens, cache)
    print(f"[Prefill : {cache.seq_len} tokens en {time.perf_counter() - t0:.2f}s]")

    # Decode

    # Initialisation du décodeur avec tampon UTF-8
    stream = StreamDecoder(self.tokenizer)

    print("Gemma : ", end="", flush=True)
    for _ in range(max_new_tokens):
      next_token = sample_token(
          logits,
          generated_ids=all_tokens,
          temperature=temperature,
          top_k=top_k,
          top_p=top_p,
          repetition_penalty=repetition_penalty,
      )

      if next_token == self.tokenizer.eos_token_id or next_token == 106:
        break

      all_tokens.append(next_token)
      # Affichage garanti sans caractère cassé
      chunk = stream.push(next_token)
      if chunk:
        print(chunk, end="", flush=True)

      logits = self.forward([next_token], cache)
    print(stream.flush())

  def chat(
        self,
        max_context_len:int=2048,
        max_new_tokens:int=256,
        temperature:float=0.7,
        top_k:int=50,
        top_p:float=0.9,
        repetition_penalty:float=1.1,
    ):
      """Boucle de conversation interactive réutilisant le même KVCache d'un tour à l'autre."""
      # On alloue un seul cache pour toute la durée de la conversation
      cache = self.create_cache(max_seq_len=max_context_len)
      all_tokens = []
      is_first_turn = True

      print(
          f"Chat prêt (capacité : {max_context_len} tokens). Tapez 'quitter'"
          " pour arrêter, 'reset' pour effacer la mémoire.\n"
          + "-" * 60
      )

      while True:
        user_input = input("\nVous : ").strip()
        if not user_input:
          continue
        if user_input.lower() in ["quitter", "exit", "quit"]:
          break
        if user_input.lower() == "reset":
          cache.reset()
          all_tokens.clear()
          is_first_turn = True
          print("[Mémoire de conversation effacée]")
          continue

        # 1. Formatage uniquement du NOUVEAU tour (sans ré-encoder l'historique)
        if is_first_turn:
          turn_text = (
              f"<bos><start_of_turn>user\n{user_input}<end_of_turn>\n"
              "<start_of_turn>model\n"
          )
          is_first_turn = False
        else:
          # On ferme le tour précédent du modèle (<end_of_turn>\n) et on ouvre celui de l'utilisateur
          turn_text = (
              f"<end_of_turn>\n<start_of_turn>user\n{user_input}<end_of_turn>\n"
              "<start_of_turn>model\n"
          )

        new_tokens = self.tokenizer.encode(turn_text, add_bos=False)

        # Vérification de la place restante dans le cache
        if cache.seq_len + len(new_tokens) + max_new_tokens > max_context_len:
          print(
              "\n[Attention : limite du cache atteinte, tapez 'reset' pour"
              " recommencer une conversation.]"
          )
          continue

        all_tokens.extend(new_tokens)

        # 2. Prefill incrémental : on ne calcule QUE les tokens de la nouvelle question
        t0 = time.perf_counter()
        logits = self.forward(new_tokens, cache)
        prefill_time = time.perf_counter() - t0

        # 3. Génération de la réponse en streaming
        stream = StreamDecoder(self.tokenizer)
        print("Gemma : ", end="", flush=True)

        generated_count = 0
        for _ in range(max_new_tokens):
          next_token = sample_token(
              logits,
              generated_ids=all_tokens,
              temperature=temperature,
              top_k=top_k,
              top_p=top_p,
              repetition_penalty=repetition_penalty,
          )

          if next_token == self.tokenizer.eos_token_id or next_token == 106:
            break

          all_tokens.append(next_token)
          generated_count += 1

          chunk = stream.push(next_token)
          if chunk:
            print(chunk, end="", flush=True)

          logits = self.forward([next_token], cache)

        print(stream.flush())
        print(
            f"   [Cache : {cache.seq_len}/{max_context_len} tokens | Prefill"
            f" incrémental : {prefill_time:.2f}s]"
        )