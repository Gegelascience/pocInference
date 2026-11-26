import numpy as np
import cupy as cp
import numpy.typing as npt

class KVCache:
  """Gère le stockage et la mise à jour des clés (K) et valeurs (V)

  pour toutes les couches du Transformer.
  """

  def __init__(self, n_layers:int, max_seq_len:int, n_kv_heads:int, head_dim:int):
    self.max_seq_len = max_seq_len
    self.seq_len = 0  # Nombre de tokens actuellement stockés dans le cache

    # Pré-allocation de la mémoire : (n_layers, max_seq_len, n_kv_heads, head_dim)
    shape = (n_layers, max_seq_len, n_kv_heads, head_dim)
    self.k_cache = np.zeros(shape, dtype=np.float32)
    self.v_cache = np.zeros(shape, dtype=np.float32)

  def update(self, layer_idx:int, k:npt.NDArray[np.float32], v:npt.NDArray[np.float32]):
    """Enregistre les nouveaux vecteurs k et v (shape: [num_new_tokens, n_kv_heads, head_dim])

    et renvoie l'intégralité de l'historique (de 0 à seq_len + num_new_tokens).
    """
    num_new:int = int(k.shape[0])
    end_pos:int = self.seq_len + num_new

    if end_pos > self.max_seq_len:
      raise ValueError(
          f"Capacité maximale du KV Cache atteinte ({self.max_seq_len} tokens)."
      )

    # Écriture directe dans la tranche pré-allouée (sans copie de tableau)
    self.k_cache[layer_idx, self.seq_len : end_pos] = k
    self.v_cache[layer_idx, self.seq_len : end_pos] = v

    # Renvoie une vue sur tous les tokens valides jusqu'à end_pos
    return (
        self.k_cache[layer_idx, :end_pos],
        self.v_cache[layer_idx, :end_pos],
    )

  def advance(self, num_tokens:int):
    """Avance le curseur de position une fois que toutes les couches ont été calculées."""
    self.seq_len += num_tokens

  def reset(self):
    """Réinitialise le cache pour une nouvelle conversation sans réallouer la RAM."""
    self.seq_len = 0