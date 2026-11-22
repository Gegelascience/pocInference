from GemmaNaive import Gemma3Numpy
from GemmaFast import Gemma3NumpyFast


if __name__ == "__main__":

    # --- Exécution ---
    # Remplacez par le chemin local dans votre cache Hugging Face (ex: ~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/...)
    modelPath ="../../modelsIA/gemma3_1b"
    model = Gemma3Numpy(modelPath)
    #model.generate("Quelle est la capitale de la France ?")

    model = Gemma3NumpyFast(modelPath)
    #model.generate("Quelle est la capitale de la France ?")

    model.chat()