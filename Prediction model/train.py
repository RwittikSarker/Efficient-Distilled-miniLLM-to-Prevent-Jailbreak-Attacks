import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from harm_model import HarmScoringModel
from utils import PromptDataset

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 4
BATCH_SIZE = 12
LR = 1e-5

# Load pre-split data
print("Loading pre-split training and validation data...")
train_df = pd.read_csv('train_data.csv', encoding='utf-8-sig')
val_df = pd.read_csv('val_data.csv', encoding='utf-8-sig')


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

def evaluate(model, val_loader, epoch):
    model.eval()
    total_loss = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # ---------------- GLOBAL REGRESSION METRICS ----------------
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred).correlation
    pearson = pearsonr(y_true, y_pred)[0]

    print(f"\n--- Regression Metrics (Epoch {epoch+1}) ---")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R²       : {r2:.4f}")
    print(f"Spearman : {spearman:.4f}")
    print(f"Pearson  : {pearson:.4f}")

    # ---------------- PER-BIN ERROR ANALYSIS ----------------
    bins = [-0.1, 0.3, 0.6, 1.1]
    bin_names = ['low', 'mid', 'high']

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "bin": pd.cut(y_true, bins=bins, labels=bin_names)
    })

    df["abs_error"] = np.abs(df.y_true - df.y_pred)
    df["signed_error"] = df.y_pred - df.y_true

    print("\n--- Error by Harm Level (Ground Truth Bins) ---")
    print(
        df.groupby("bin").agg(
            count=("y_true", "count"),
            MAE=("abs_error", "mean"),
            RMSE=("abs_error", lambda x: np.sqrt(np.mean(x ** 2))),
            Mean_True=("y_true", "mean"),
            Mean_Pred=("y_pred", "mean"),
            Bias=("signed_error", "mean")
        )
    )

    # ---------------- HIGH-HARM FAILURE DIAGNOSTIC ----------------
    high_df = df[df["bin"] == "high"]
    if len(high_df) > 0:
        under_rate = np.mean(high_df.y_pred < 0.6)
        mean_under = np.mean(high_df.y_true - high_df.y_pred)
        print("\nHigh-harm underestimation rate:", round(under_rate, 3))
        print("Mean underestimation magnitude:", round(mean_under, 4))

    print("--------------------------------------------------\n")

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

    val_loss = evaluate(model, val_loader, epoch)
    print(f"Epoch {epoch+1} | Total train loss: {total_train_loss}")
    print(f"Epoch {epoch+1} | Train Loss: {total_train_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), 'baseline_model.pt')
