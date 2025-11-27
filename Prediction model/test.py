from datasets import load_dataset
import pandas as pd
import torch
from transformers import RobertaTokenizer
from harm_model import HarmScoringModel

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load dataset
ds = load_dataset("JailbreakV-28K/JailBreakV-28k", "JailBreakV_28K")

# Convert the 'train' split to a DataFrame
df = ds["JailBreakV_28K"].to_pandas()

# 2. Extract prompt text from dictionary
df["prompt_text"] = df["redteam_query"]

# 3. Sample max 500 prompts
df_sample = df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)

# 4. Tokenize
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
enc = tokenizer(
    df_sample["prompt_text"].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# 5. Load your trained harm scoring model
model = HarmScoringModel().to(DEVICE)
# Safe loading of model weights
state_dict = torch.load("harm_model.pt", map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# 6. Run inference
with torch.no_grad():
    outputs = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
    outputs = torch.sigmoid(outputs) # Apply sigmoid to the raw model output
preds = outputs.squeeze().cpu().numpy()

# 7. Attach predictions
df_sample["predicted_harmscore"] = preds

# 8. Inspect top harmful prompts
df_sample.to_csv("predicted_harm_scores2.csv", index=False)
