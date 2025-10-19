# llama_8b.py (updated for attack model with progress)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, logging

# Enable info logging from Transformers
logging.set_verbosity_info()

def get_quant_config():
    """Returns the BitsAndBytesConfig for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def load_attack_model():
    print("--- Starting to load Attack Model (Qwen2.5-7B) ---")
    quant_config = get_quant_config()
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Tokenizer loaded.")

    print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",  # Automatically use available GPUs
        low_cpu_mem_usage=True,  # Reduce memory spikes
    )
    print("--- Attack Model Loaded Successfully ---")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_attack_model()
