import json
import re


class Gemma3Tokenizer:
  """Tokenizer BPE pur Python lisant directement le fichier tokenizer.json de Gemma 3."""

  def __init__(self, tokenizer_json_path):
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
      data = json.load(f)

    # 1. Vocabulaire : texte -> ID et ID -> texte
    self.vocab = data["model"]["vocab"]
    self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}

    # 2. Ajout des tokens spéciaux (added_tokens)
    self.special_tokens = {}
    for item in data.get("added_tokens", []):
      content = item["content"]
      idx = item["id"]
      self.vocab[content] = idx
      self.id_to_token[idx] = content
      self.special_tokens[content] = idx

    # Expression régulière pour isoler les tokens spéciaux (triés du plus long au plus court)
    sorted_specials = sorted(self.special_tokens.keys(), key=len, reverse=True)
    self.special_pattern = re.compile(
        "(" + "|".join(re.escape(s) for s in sorted_specials) + ")"
    )

    # 3. Règles de fusion BPE : (token_a, token_b) -> rang de priorité
    self.bpe_ranks = {}
    for rank, merge in enumerate(data["model"]["merges"]):
      # Selon la version de tokenizers, merge est soit une chaîne "a b", soit une liste ["a", "b"]
      pair = tuple(merge.split(" ")) if isinstance(merge, str) else tuple(merge)
      self.bpe_ranks[pair] = rank

    self.cache = {}  # Cache pour accélérer l'encodage des mots déjà vus
    self.eos_token_id = self.vocab.get("<eos>", 1)

  def _bpe(self, word):
    """Applique l'algorithme Byte-Pair Encoding sur un morceau de texte."""
    if word in self.cache:
      return self.cache[word]

    symbols = list(word)
    while len(symbols) > 1:
      # Trouver la paire adjacente avec le plus petit rang dans self.bpe_ranks
      pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
      best_pair = min(
          pairs, key=lambda p: self.bpe_ranks.get(p, float("inf"))
      )

      if best_pair not in self.bpe_ranks:
        break  # Plus aucune fusion possible

      # Fusionner toutes les occurrences de best_pair
      first, second = best_pair
      new_symbols = []
      i = 0
      while i < len(symbols):
        if (
            i < len(symbols) - 1
            and symbols[i] == first
            and symbols[i + 1] == second
        ):
          new_symbols.append(first + second)
          i += 2
        else:
          new_symbols.append(symbols[i])
          i += 1
      symbols = new_symbols

    self.cache[word] = symbols
    return symbols

  def encode(self, text, add_bos=False):
    """Convertit une chaîne de caractères en liste d'IDs."""
    token_ids = [self.vocab["<bos>"]] if add_bos else []

    # Séparation entre tokens spéciaux et texte normal
    chunks = self.special_pattern.split(text)

    for chunk in chunks:
      if not chunk:
        continue
      if chunk in self.special_tokens:
        token_ids.append(self.special_tokens[chunk])
      else:
        # Gemma remplace les espaces par '▁' (U+2581)
        normalized = chunk.replace(" ", "▁")
        for sub_token in self._bpe(normalized):
          if sub_token in self.vocab:
            token_ids.append(self.vocab[sub_token])
          else:
            # Repli sur les octets bruts UTF-8 (<0xXX>) pour les caractères inconnus
            for byte in sub_token.encode("utf-8"):
              byte_tok = f"<0x{byte:02X}>"
              if byte_tok in self.vocab:
                token_ids.append(self.vocab[byte_tok])

    return token_ids

  def decode(self, token_ids):
    """Reconstruit le texte à partir d'une liste d'IDs."""
    pieces = []
    byte_buffer = bytearray()

    for idx in token_ids:
      tok = self.id_to_token.get(idx, "")
      # Ignorer les balises de contrôle à l'affichage
      if tok in ("<bos>", "<eos>", "<pad>", "<start_of_turn>", "<end_of_turn>"):
        continue

      # Gestion des tokens d'octets bruts (<0xXX>) pour reconstruire les accents/emojis
      if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
        byte_buffer.append(int(tok[3:5], 16))
      else:
        if byte_buffer:
          pieces.append(byte_buffer.decode("utf-8", errors="replace"))
          byte_buffer.clear()
        pieces.append(tok)

    if byte_buffer:
      pieces.append(byte_buffer.decode("utf-8", errors="replace"))

    # Remplacement inverse de '▁' par un espace standard
    return "".join(pieces).replace("▁", " ")