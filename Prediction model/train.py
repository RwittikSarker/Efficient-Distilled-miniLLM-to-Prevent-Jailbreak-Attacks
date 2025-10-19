import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

from harm_model import HarmScoringModel
from utils import PromptDataset

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5

# Load data and stratified split based on harm_score categories
df = pd.read_csv('data/prompts.csv', encoding='utf-8-sig')  # columns: prompt, harm_score
bins = [-0.1, 0.3, 0.6, 1.1]
labels = ['low', 'mid', 'high']
df['harm_category'] = pd.cut(df['harm_score'], bins=bins, labels=labels)

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
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.L1Loss()  # for regression

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
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

        total_train_loss += loss.item()

    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {total_train_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), 'harm_model.pt')
