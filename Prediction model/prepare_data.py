# prepare_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

print("--- Starting Data Preparation ---")

# --- CONFIG ---
# This should be your full, balanced dataset
INPUT_DATA_FILE = "data.csv" 
TEST_SET_SIZE = 0.15 # 15% for the final test set
VALIDATION_SET_SIZE = 0.1 # 10% of the remaining data for validation

# 1. Load the full dataset
try:
    df = pd.read_csv(INPUT_DATA_FILE, encoding='utf-8-sig')
    print(f"Loaded {len(df)} rows from '{INPUT_DATA_FILE}'.")
except FileNotFoundError:
    print(f"FATAL ERROR: Input data file not found at '{INPUT_DATA_FILE}'.")
    exit()

# 2. Create the harm_category for stratification
bins = [-0.1, 0.3, 0.6, 1.1]
labels = ['low', 'mid', 'high']
df['harm_category'] = pd.cut(df['harmscore'], bins=bins, labels=labels)

# 3. Split off the final, held-out test set
train_val_df, test_df = train_test_split(
    df, 
    test_size=TEST_SET_SIZE, 
    random_state=42, 
    stratify=df['harm_category']
)
print(f"Split off {len(test_df)} rows for the final test set.")

# 4. Split the remaining data into training and validation sets
train_df, val_df = train_test_split(
    train_val_df,
    test_size=VALIDATION_SET_SIZE,
    random_state=42,
    stratify=train_val_df['harm_category']
)
print(f"Remaining data split into {len(train_df)} training rows and {len(val_df)} validation rows.")

# 5. Save the three distinct datasets
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
test_df.to_csv('final_test_set.csv', index=False)

print("\n--- Data Preparation Complete ---")
print("Successfully created:")
print(f"  - train_data.csv ({len(train_df)} rows)")
print(f"  - val_data.csv ({len(val_df)} rows)")
print(f"  - final_test_set.csv ({len(test_df)} rows)")