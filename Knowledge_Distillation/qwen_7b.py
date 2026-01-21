import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen_2_5_7b():
    """
    Loads the Qwen2.5-7B-Instruct model and tokenizer in BF16/FP16.
    """
    print("Loading Qwen/Qwen2.5-7B-Instruct in FP16/BF16 (Teacher)...")
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    # Load tokenizer with left-padding (standard for generation/distillation)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Qwen tokenizers usually define pad_token, but keep this for safety
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in BF16 (preferred on Ampere+ GPUs; use torch.float16 if needed)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_qwen_2_5_7b()
    print("✅ Teacher Model (Qwen2.5-7B) loaded successfully")
