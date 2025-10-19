import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# 1. Import your model loading functions from the other files
from llama_8b import load_llama_8b
from llama_1b import load_llama_1b


class DistillationTrainer(Trainer):
    """
    Custom Trainer for knowledge distillation.
    """
    def __init__(self, teacher_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.to(self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # --- Student Loss ---
        student_outputs = model(**inputs)
        student_loss = student_outputs.loss
        student_logits = student_outputs.logits

        # --- Teacher (Soft Targets) ---
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        # --- Distillation Loss ---
        temperature = 2.0
        distillation_loss = F.kl_div(
            input=F.log_softmax(student_logits / temperature, dim=-1),
            target=F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean"
        ) * (temperature ** 2)

        # --- Combine ---
        alpha = 0.5
        loss = alpha * distillation_loss + (1 - alpha) * student_loss

        return (loss, student_outputs) if return_outputs else loss


def main():
    # --- Step 1: Load Models and Tokenizer ---
    student_model, tokenizer = load_llama_1b()
    teacher_model, _ = load_llama_8b()
    # Add LoRA adapter to student model (not teacher)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student_model = get_peft_model(student_model, lora_config)

    # --- Step 2: Load a Real Hugging Face Dataset ---
    print("\n--- Loading Wikitext Dataset ---")
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    # --- Step 3: Preprocess (Tokenize) ---
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=["text"])
    train_dataset = tokenized_datasets["train"]

    print("--- Dataset Ready ---\n")

    # --- Step 4: Set up Trainer ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./distilled_llama_1b_results",
        overwrite_output_dir=True,
        num_train_epochs=5,  # change if you want more
        per_device_train_batch_size=5,
        gradient_accumulation_steps=1,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
    )

    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # --- Step 5: Start Training ---
    print("--- Starting Knowledge Distillation ---")
    trainer.train()
    print("--- Distillation Complete ---")

    # --- Step 6: Save Final Model ---
    final_model_path = "./distilled_llama_1b_final"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Distilled student model saved to {final_model_path}")


if __name__ == "__main__":
    main()
