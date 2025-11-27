import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_llama_8b():
    """
    Loads the Llama 3.1 8B model and tokenizer.

    - If CUDA is available: load in 4-bit on GPU.
    - If CUDA is NOT available: load on CPU (no 4-bit).
    """

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        print("✅ CUDA available – loading LLaMA 8B in 4-bit on GPU")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",          # let accelerate decide
            quantization_config=bnb_config,
            dtype=torch.float16,        # replaces torch_dtype (deprecated)
        )

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_llama_8b()
    print("Model and tokenizer loaded successfully")
