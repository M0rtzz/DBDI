"""
Configuration file .
"""

# Data paths
path_harmful = "data/harmful.csv"
path_harmless = "data/harmless.csv"
path_harmful_test = "data/harmful_test.csv"
path_harmless_test = "data/harmless_test.csv"
path_harmful_calibration = "data/harmful_calibration.csv"
path_harmless_calibration = "data/harmless_calibration.csv"

# Model paths
model_paths = {
    "mistral": "./models2/Mistral-7B-Instruct-v0.2",
    "llama-2": "./models/Llama-2-7b-chat-hf",
    "vicuna-7b": "./models/vicuna-7b-v1.5",
    "gemma-2-9b-it": "./models2/gemma-2-9b-it",
    "llama-3": "./models/Llama-3.1-8B-Instruct",
    "llama-3.2-3b": "./models/Llama-3.2-3B-Instruct", 
    "Qwen7B": "./models/Qwen2.5-7B-Instruct", 
    # "mistral-sorry-bench": "./models/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",
    "deepseek":"./models2/deepseek-llm-7b-chat",
    "Llamaguard":"./models2/Llama-Guard-3-8B"
}
