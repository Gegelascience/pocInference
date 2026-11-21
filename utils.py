import json
import struct
import numpy as np


def load_safetensors_bf16_to_fp32(filepath):
  """Lit un fichier .safetensors en pur Python/NumPy et convertit

  les tenseurs bfloat16 en float32.
  """
  weights = {}

  with open(filepath, "rb") as f:
    # 1. Les 8 premiers octets donnent la taille de l'en-tête JSON (entier 64 bits little-endian)
    header_size = struct.unpack("<Q", f.read(8))[0]

    # 2. Lecture de l'en-tête JSON contenant les noms, formes et positions des tenseurs
    header = json.loads(f.read(header_size).decode("utf-8"))

    # Le début des données brutes se situe juste après les 8 octets + l'en-tête
    data_start = 8 + header_size

    for name, info in header.items():
      if name == "__metadata__":
        continue

      dtype = info["dtype"]
      shape = info["shape"]
      start_offset, end_offset = info["data_offsets"]
      num_bytes = end_offset - start_offset

      # 3. Positionnement et lecture des octets bruts du tenseur
      f.seek(data_start + start_offset)
      raw_bytes = f.read(num_bytes)

      # 4. Conversion selon le type stocké
      if dtype == "BF16":
        # Lecture en uint16 -> extension en uint32 -> décalage de 16 bits -> vue float32
        arr_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
        arr_fp32 = (arr_u16.astype(np.uint32) << 16).view(np.float32)
        weights[name] = arr_fp32.reshape(shape)

      elif dtype == "F32":
        weights[name] = np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)

      elif dtype == "F16":
        weights[name] = (
            np.frombuffer(raw_bytes, dtype=np.float16)
            .astype(np.float32)
            .reshape(shape)
        )

  return weights


def sample_token(
    logits,
    generated_ids=None,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
):
  """Sélectionne le prochain token à partir des logits bruts en appliquant :

  repetition_penalty -> temperature -> top_k -> top_p -> tirage aléatoire.
  """
  # On travaille sur une copie en float32 pour ne pas modifier le tableau d'origine
  logits = logits.astype(np.float32).copy()

  # 1. Pénalité de répétition sur les tokens déjà vus
  if generated_ids and repetition_penalty != 1.0:
    unique_ids = list(set(generated_ids))
    scores = logits[unique_ids]
    # Si le score est positif, on le divise ; s'il est négatif, on le multiplie
    logits[unique_ids] = np.where(
        scores > 0, scores / repetition_penalty, scores * repetition_penalty
    )

  # Si température <= 0, on retombe sur le choix glouton (déterministe)
  if temperature <= 0.0:
    return int(np.argmax(logits))

  # 2. Application de la température
  logits = logits / temperature

  # 3. Filtrage Top-K (on ne garde que les K plus grands logits)
  if top_k > 0 and top_k < len(logits):
    # np.argpartition trouve les K plus grands éléments en temps linéaire O(N)
    kth_val = np.partition(logits, -top_k)[-top_k]
    logits[logits < kth_val] = -float("inf")

  # 4. Conversion en probabilités (Softmax numériquement stable)
  logits_shifted = logits - np.max(logits)
  exp_logits = np.exp(logits_shifted)
  probs = exp_logits / np.sum(exp_logits)

  # 5. Filtrage Top-P (Nucleus Sampling)
  if 0.0 < top_p < 1.0:
    # Tri des probabilités par ordre décroissant
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)

    # On garde tous les tokens jusqu'à dépasser top_p (en gardant toujours au moins le 1er)
    cutoff_mask = cumulative_probs > top_p
    cutoff_mask[1:] = cutoff_mask[:-1].copy()
    cutoff_mask = False

    # Mise à zéro des tokens éliminés et renormalisation
    indices_to_remove = sorted_indices[cutoff_mask]
    probs[indices_to_remove] = 0.0
    probs /= np.sum(probs)

  # 6. Tirage pondéré aléatoire selon la distribution finale
  return int(np.random.choice(len(probs), p=probs))


class StreamDecoder:
  """Accumule les tokens en streaming jusqu'à former des caractères UTF-8 valides."""

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer
    self.pending_tokens = []

  def push(self, token_id):
    """Ajoute un nouveau token et renvoie le texte affichable (ou '' si caractère incomplet)."""
    self.pending_tokens.append(token_id)
    text = self.tokenizer.decode(self.pending_tokens)

    # \ufffd est le caractère de remplacement produit par errors="replace" sur un octet incomplet
    if "\ufffd" in text:
      return ""

    # La séquence UTF-8 est valide et complète : on vide le tampon et on renvoie le texte
    self.pending_tokens.clear()
    return text

  def flush(self):
    """Vide le reste du tampon en fin de génération (au cas où)."""
    if not self.pending_tokens:
      return ""
    text = self.tokenizer.decode(self.pending_tokens)
    self.pending_tokens.clear()
    return text