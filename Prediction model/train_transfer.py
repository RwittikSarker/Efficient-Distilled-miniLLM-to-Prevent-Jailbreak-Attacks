import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr


# --- Import your custom model and dataset classes ---
from harm_model import HarmScoringModel
from utils import PromptDataset

# ==============================================================================
# --- CONFIG: UPDATE THESE AS NEEDED ---
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- File Paths ---
JIGSAW_DATA_PATH = "jigsaw.csv"
#HARMSCORE_DATA_PATH = "data.csv" 
INTERMEDIATE_MODEL_PATH = "roberta-toxic.pt"
FINAL_MODEL_PATH = "transfer_learning_model.pt"

# --- Stage 1: Jigsaw Training Hyperparameters ---
EPOCHS_JIGSAW = 1
LR_JIGSAW = 2e-5
BATCH_SIZE_JIGSAW = 16 

# --- Stage 2: Harmscore Training Hyperparameters (Matching your train.py) ---
EPOCHS_FINAL = 4
LR_FINAL = 1e-5
BATCH_SIZE_FINAL = 12

# --- General Config ---
TOKENIZER_NAME = "roberta-base"
MAX_LENGTH = 128
WEIGHT_DECAY = 0.01
# ==============================================================================


# ==============================================================================
# --- STAGE 1: JIGSAW DATASET AND CLASSIFICATION MODEL ---
# ==============================================================================

class JigsawDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = dataframe['comment_text'].tolist()
        self.labels = dataframe[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

class ToxicityClassifierModel(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 6)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# ==============================================================================
# --- STAGE 2: EVALUATION FUNCTION (ADAPTED FROM YOUR train.py) ---
# ==============================================================================

def evaluate_final(model, val_loader, loss_fn, epoch):
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

    high_df = df[df["bin"] == "high"]
    if len(high_df) > 0:
        under_rate = np.mean(high_df.y_pred < 0.6)
        mean_under = np.mean(high_df.y_true - high_df.y_pred)
        print("\nHigh-harm underestimation rate:", round(under_rate, 3))
        print("Mean underestimation magnitude:", round(mean_under, 4))

    print("--------------------------------------------------\n")

    return total_loss / len(val_loader)

# ==============================================================================
# --- MAIN EXECUTION SCRIPT ---
# ==============================================================================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_NAME)

    # --------------------------------------------------------------------------
    # STAGE 1: Intermediate Training on Jigsaw
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("      STAGE 1: INTERMEDIATE TRAINING ON JIGSAW")
    print("="*50 + "\n")

    try:
        jigsaw_df = pd.read_csv(JIGSAW_DATA_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Jigsaw data not found at '{JIGSAW_DATA_PATH}'.")
        exit()

    jigsaw_dataset = JigsawDataset(jigsaw_df, tokenizer, max_length=MAX_LENGTH)
    jigsaw_loader = DataLoader(jigsaw_dataset, batch_size=BATCH_SIZE_JIGSAW, shuffle=True)
    
    model_jigsaw = ToxicityClassifierModel(model_name=TOKENIZER_NAME).to(DEVICE)
    optimizer = AdamW(model_jigsaw.parameters(), lr=LR_JIGSAW, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    
    total_steps_jigsaw = len(jigsaw_loader) * EPOCHS_JIGSAW
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps_jigsaw
    )

    print(f"Starting Jigsaw training for {EPOCHS_JIGSAW} epoch(s)...")
    for epoch in range(EPOCHS_JIGSAW):
        model_jigsaw.train()
        loop = tqdm(jigsaw_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model_jigsaw(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    print(f"Intermediate training complete. Saving model to '{INTERMEDIATE_MODEL_PATH}'...")
    torch.save(model_jigsaw.state_dict(), INTERMEDIATE_MODEL_PATH)
    del model_jigsaw, jigsaw_df, jigsaw_dataset, jigsaw_loader, optimizer, scheduler
    torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # STAGE 2: Final Fine-tuning on Harmscore Data
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("      STAGE 2: FINAL FINE-TUNING ON HARMSCORE DATA")
    print("="*50 + "\n")


    # Load pre-split data
    print("Loading pre-split training and validation data...")
    train_df = pd.read_csv('train_data.csv', encoding='utf-8-sig')
    val_df = pd.read_csv('val_data.csv', encoding='utf-8-sig')
    
    train_dataset = PromptDataset(train_df, tokenizer, max_length=MAX_LENGTH)
    val_dataset = PromptDataset(val_df, tokenizer, max_length=MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_FINAL, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_FINAL)

    final_model = HarmScoringModel(model_name=TOKENIZER_NAME).to(DEVICE)

    print(f"\nLoading weights from intermediate model '{INTERMEDIATE_MODEL_PATH}'...")
    intermediate_weights = torch.load(INTERMEDIATE_MODEL_PATH, map_location=DEVICE)
    final_model.load_state_dict(intermediate_weights, strict=False)
    print("Successfully loaded RoBERTa weights. The regression head is randomly initialized.")

    optimizer = AdamW(final_model.parameters(), lr=LR_FINAL, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    total_steps_final = len(train_loader) * EPOCHS_FINAL
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps_final
    )

    print(f"\nStarting final fine-tuning for {EPOCHS_FINAL} epoch(s)...")
    for epoch in range(EPOCHS_FINAL):
        final_model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = final_model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(train_loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Evaluate at the end of each epoch
        val_loss = evaluate_final(final_model, val_loader, loss_fn, epoch)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print(f"\nFinal training complete. Saving best model to '{FINAL_MODEL_PATH}'...")
    torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
    print("Process finished successfully.")