import torch
import pandas as pd
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# --- Import your custom model and dataset classes ---
# Make sure harm_model.py and utils.py are in the same directory
from harm_model import HarmScoringModel
from utils import PromptDataset

# ==============================================================================
# --- CONFIG: UPDATE THESE PATHS ---
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to the trained model weights for your baseline model
MODEL_A_PATH = "baseline_model.pt" 

# Path to the trained model weights for your new transfer learning model
MODEL_B_PATH = "transfer_learning_model.pt"

# Path to your held-out, final test set
TEST_SET_PATH = "final_test_set.csv"

# Model and tokenizer configuration
TOKENIZER_NAME = "roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
# ==============================================================================


def get_predictions_and_report(model, dataloader, device):
    """
    Runs inference on a given model and dataloader, returning predictions
    and a formatted classification report.
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            # Get raw logits from the model
            outputs = model(input_ids, attention_mask)
            
            # Apply sigmoid to convert logits to probabilities
            preds_probs = torch.sigmoid(outputs)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds_probs.cpu().numpy())

    # --- Generate Classification Report ---
    bins = [-0.1, 0.3, 0.6, 1.1]
    class_names = ['low', 'mid', 'high']
    
    # Convert continuous scores into categorical classes
    true_classes = pd.cut(all_labels, bins=bins, labels=class_names, ordered=False)
    pred_classes = pd.cut(all_preds, bins=bins, labels=class_names, ordered=False)

    # Fill any potential NaNs (if a prediction falls outside bins)
    true_classes = true_classes.fillna('low')
    pred_classes = pred_classes.fillna('low')

    report_text = classification_report(
        true_classes, 
        pred_classes, 
        target_names=class_names, 
        zero_division=0
    )
    
    return pred_classes.tolist(), report_text


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Tokenizer and Test Data
    print(f"\nLoading tokenizer '{TOKENIZER_NAME}'...")
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_NAME)
    
    print(f"Loading test data from '{TEST_SET_PATH}'...")
    try:
        test_df = pd.read_csv(TEST_SET_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Test set not found at '{TEST_SET_PATH}'. Please create it first.")
        exit()

    test_dataset = PromptDataset(test_df, tokenizer, max_length=MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 2. Evaluate Model A (Baseline)
    print("\n--- Evaluating Model A (Baseline) ---")
    try:
        model_a = HarmScoringModel().to(DEVICE)
        model_a.load_state_dict(torch.load(MODEL_A_PATH, map_location=DEVICE))
        preds_a, report_a = get_predictions_and_report(model_a, test_loader, DEVICE)
    except FileNotFoundError:
        print(f"Could not find model weights at '{MODEL_A_PATH}'. Skipping Model A evaluation.")
        report_a, preds_a = "Model A not found.", None
    
    # 3. Evaluate Model B (Transfer Learning)
    print("\n--- Evaluating Model B (Transfer Learning) ---")
    try:
        model_b = HarmScoringModel().to(DEVICE)
        model_b.load_state_dict(torch.load(MODEL_B_PATH, map_location=DEVICE))
        preds_b, report_b = get_predictions_and_report(model_b, test_loader, DEVICE)
    except FileNotFoundError:
        print(f"Could not find model weights at '{MODEL_B_PATH}'. Skipping Model B evaluation.")
        report_b, preds_b = "Model B not found.", None

    # 4. Display Comparative Results
    print("\n\n==================================================")
    print("      FINAL MODEL COMPARISON RESULTS")
    print("==================================================")
    
    print("\n--- REPORT FOR MODEL A (Baseline) ---")
    print(report_a)
    
    print("\n--- REPORT FOR MODEL B (Transfer Learning) ---")
    print(report_b)

    # 5. Perform and Display Error Analysis
    if preds_a is not None and preds_b is not None:
        print("\n\n==================================================")
        print("              ERROR ANALYSIS")
        print("==================================================")
        
        bins = [-0.1, 0.3, 0.6, 1.1]
        class_names = ['low', 'mid', 'high']
        test_df['true_class'] = pd.cut(test_df['harmscore'], bins=bins, labels=class_names, ordered=False).fillna('low')
        test_df['pred_class_A'] = preds_a
        test_df['pred_class_B'] = preds_b

        fixed_errors = test_df[
            (test_df['true_class'] != test_df['pred_class_A']) & 
            (test_df['true_class'] == test_df['pred_class_B'])
        ]

        if not fixed_errors.empty:
            print("\n--- Prompts where Model B FIXED an error from Model A ---")
            for index, row in fixed_errors.iterrows():
                print(f"\nPrompt: {row['prompt']}")
                print(f"  > True Label: {row['true_class']}")
                print(f"  > Baseline A Prediction (WRONG): {row['pred_class_A']}")
                print(f"  > Transfer B Prediction (CORRECT): {row['pred_class_B']}")
        else:
            print("\nNo cases found where Model B fixed an error made by Model A on this test set.")
            
        new_errors = test_df[
            (test_df['true_class'] == test_df['pred_class_A']) & 
            (test_df['true_class'] != test_df['pred_class_B'])
        ]
        
        if not new_errors.empty:
            print("\n--- Prompts where Model B CREATED a NEW error (Regression) ---")
            for index, row in new_errors.iterrows():
                print(f"\nPrompt: {row['prompt']}")
                print(f"  > True Label: {row['true_class']}")
                print(f"  > Baseline A Prediction (CORRECT): {row['pred_class_A']}")
                print(f"  > Transfer B Prediction (WRONG): {row['pred_class_B']}")
        else:
            print("\nNo new errors were created by Model B on this test set.")

    print("\n\nEvaluation complete.")