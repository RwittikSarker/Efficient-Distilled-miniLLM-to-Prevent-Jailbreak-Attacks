import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llama_8b():
    """
    Loads the Llama 3.1 8B model and tokenizer in FP16.
    """
    print("Loading Meta-Llama-3.1-8B-Instruct in FP16 (Teacher)...")
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Load tokenizer with left-padding (standard for generation/distillation)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load directly in Float16 (No Quantization)
    # device_map="auto" allows splitting across GPUs if needed, or defaults to CUDA
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",          
        torch_dtype=torch.bfloat16        
    )

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_llama_8b()
    print("✅ Teacher Model (8B) loaded in FP16 successfully")