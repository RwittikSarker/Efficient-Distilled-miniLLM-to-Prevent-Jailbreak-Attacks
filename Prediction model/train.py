import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

from harm_model import HarmScoringModel
from utils import PromptDataset

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 7
BATCH_SIZE = 12
LR = 1e-5

#Load data and stratified split based on harm_score categories
df = pd.read_csv('data.csv', encoding='utf-8-sig')  # columns: prompt, harm_score
bins = [-0.1, 0.3, 0.6, 1.1]
labels = ['low', 'mid', 'high']
df['harm_category'] = pd.cut(df['harmscore'], bins=bins, labels=labels)


train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df['harm_category']
)
train_df = train_df.drop(columns=['harm_category'])
val_df = val_df.drop(columns=['harm_category'])


# Tokenizer and datasets
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_dataset = PromptDataset(train_df, tokenizer)
val_dataset = PromptDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model, optimizer, loss
model = HarmScoringModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Calculate total training steps
total_steps = len(train_loader) * EPOCHS

# Create the scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0, # Default, no warmup
    num_training_steps=total_steps
)

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    
    # Store all true labels and predictions
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            # Get model outputs (these are raw logits)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # --- New: Store labels and predictions for the report ---
            # Apply sigmoid to convert logits to probabilities for evaluation
            preds_probs = torch.sigmoid(outputs)
            
            # Move tensors to CPU and append to our lists
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds_probs.cpu().numpy())

    # --- New: Generate and print the classification report ---
    print(f"\n--- Classification Report for Epoch {epoch+1} ---")
    
    # Define the harm score bins and labels, matching your dataset creation
    bins = [-0.1, 0.3, 0.6, 1.1]
    class_names = ['low', 'mid', 'high']
    
    # Convert continuous scores (true labels) into categorical classes
    true_classes = pd.cut(all_labels, bins=bins, labels=class_names, ordered=False)
    
    # Convert predicted probabilities into categorical classes
    pred_classes = pd.cut(all_preds, bins=bins, labels=class_names, ordered=False)

    # Handle cases where a prediction might fall outside the bins, though unlikely
    true_classes = true_classes.fillna('low')
    pred_classes = pred_classes.fillna('low')
    
    # Create and print the report
    # zero_division=0 prevents warnings if a class has no predicted samples
    report = classification_report(true_classes, pred_classes, target_names=class_names, zero_division=0)
    print(report)
    print("--------------------------------------------------\n")
    
    # The function still returns the average loss as before
    print(f"Epoch {epoch+1} | Validation loss: {total_loss}")
    return total_loss / len(val_loader)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch+1} | Total train loss: {total_train_loss}")
    print(f"Epoch {epoch+1} | Train Loss: {total_train_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), 'harm_model.pt')
