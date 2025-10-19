import torch
from transformers import RobertaTokenizer
from harm_model import HarmScoringModel

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'harm_model.pt'
MODEL_NAME = 'roberta-base'

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = HarmScoringModel(model_name=MODEL_NAME)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

def predict(prompt):
    encoded = tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        score = model(input_ids, attention_mask)
    return score.item()

# Example usage
if __name__ == "__main__":
    test_prompt = input("Enter a prompt to test: ")
    score = predict(test_prompt)
    print(f"Harm score: {score:.4f}")
