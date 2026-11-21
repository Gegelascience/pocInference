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