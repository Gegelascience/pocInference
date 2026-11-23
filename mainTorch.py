import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#model_id = "google/gemma-3-1b-it"
model_id = "../../modelsIA/gemma3_1b"

print("Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto"
).eval()

# Historique de la conversation (permet au modèle de se souvenir du contexte)
messages:list[dict[str,str]] = []

print("\nModèle prêt ! Tapez 'quitter' pour arrêter.\n" + "-" * 45)

while True:
  user_input = input("\nVous : ")
  if user_input.strip().lower() in ["quitter", "exit", "quit"]:
    break

  # 1. Ajout du message de l'utilisateur à l'historique
  messages.append({"role": "user", "content": user_input})

  # 2. Encodage de tout l'historique au format Gemma 3
  inputs = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=True,
      return_dict=True,
      return_tensors="pt",
  ).to(model.device)

  # 3. Génération de la réponse
  with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

  # 4. Extraction et décodage de la nouvelle réponse uniquement
  input_len = inputs["input_ids"].shape[-1]
  reponse = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

  print(f"\nGemma : {reponse}")

  # 5. Sauvegarde de la réponse du modèle dans l'historique
  messages.append({"role": "assistant", "content": reponse})
