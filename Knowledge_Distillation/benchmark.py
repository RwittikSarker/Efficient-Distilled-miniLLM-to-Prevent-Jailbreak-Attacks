# benchmark.py
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

def load_model_for_evaluation(model_path: str):
    """
    Loads a model and tokenizer from a specified path for evaluation.
    This can be a local path or a Hugging Face model ID.
    """
    print(f"\n--- Loading model from: {model_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto", # Automatically use GPU if available
        torch_dtype=torch.bfloat16 # Use bfloat16 for efficiency
    )
    print("--- Model loaded successfully ---")
    return model, tokenizer

def calculate_perplexity(model, tokenizer, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="test"):
    """
    Calculates the perplexity of a model on a given dataset.
    Lower perplexity is better.
    """
    print(f"\n--- Calculating perplexity on '{dataset_name}' dataset ---")
    
    # Load the dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    
    # Tokenize the dataset
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", max_length=4096, truncation=True)
    
    # Define evaluation parameters
    max_length = min(getattr(model.config, "max_position_embeddings", 4096), 4096)
    stride = 256  # Reduce stride to lower memory usage
    seq_len = encodings.input_ids.size(1)

    nlls = [] # Negative log likelihoods
    prev_end_loc = 0
    
    # Use tqdm for a progress bar
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # Mask tokens that are not part of the new sequence

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    # Calculate perplexity
    perplexity = torch.exp(torch.stack(nlls).mean())
    print("--- Perplexity calculation complete ---")
    return perplexity.item()

def generate_qualitative_samples(model, tokenizer, prompts: list):
    """
    Generates responses to a list of prompts for qualitative comparison.
    """
    print("\n--- Generating qualitative samples ---")
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        messages = [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"Response: {response.strip()}")
    print("--- Sample generation complete ---")

def main():
    # --- Define models to benchmark ---
    # The original, pre-trained model from Hugging Face
    original_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    
    # The path where you saved your distilled model
    distilled_model_path = "distilled_llama_1b_final_local"  # Remove './' for local path
    
    models_to_benchmark = {
        "Original 1B Model": original_model_id,
        "Distilled 1B Model": distilled_model_path
    }
    
    # --- Define prompts for qualitative analysis ---
    qualitative_prompts = [
        "Explain the concept of knowledge distillation in one sentence.",
        "Write a python code snippet to reverse a string.",
        "What are the three most important things to consider when starting a new project?"
    ]
    
    # --- Run the benchmark for each model ---
    results = {}
    for name, path in models_to_benchmark.items():
        print(f"\n=========================================")
        print(f"  Benchmarking: {name}")
        print(f"=========================================")
        try:
            model, tokenizer = load_model_for_evaluation(path)
            
            # Quantitative Analysis (Perplexity)
            perplexity = calculate_perplexity(model, tokenizer)
            results[name] = {"perplexity": perplexity}
            
            # Qualitative Analysis (Sample Generations)
            generate_qualitative_samples(model, tokenizer, qualitative_prompts)
            
            # Clean up memory
            del model
            del tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Could not benchmark {name}. Error: {e}")
            results[name] = {"perplexity": "Error", "error_message": str(e)}

    # --- Print Final Results ---
    print("\n\n=========================================")
    print("        Benchmark Results Summary")
    print("=========================================")
    print(f"{'Model':<25} | {'Perplexity (lower is better)':<30}")
    print("-" * 60)
    for name, res in results.items():
        ppl = res.get('perplexity', 'N/A')
        if isinstance(ppl, float):
            ppl = f"{ppl:.4f}"
        print(f"{name:<25} | {ppl:<30}")
    print("=========================================\n")


if __name__ == "__main__":
    main()