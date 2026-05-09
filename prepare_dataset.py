import pandas as pd
import re
from utils.text_cleaner import clean_text

# Load datasets
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")
wel_df = pd.read_csv("dataset/WELFake_Dataset.csv")

# Add labels
fake_df["label"] = 0
true_df["label"] = 1

# Keep required columns
fake_df = fake_df[["text", "label"]]
true_df = true_df[["text", "label"]]

# WELFake dataset
wel_df = wel_df[["text", "label"]]

# Merge datasets
final_df = pd.concat([
    fake_df,
    true_df,
    wel_df
], ignore_index=True)

# Remove null values
final_df.dropna(inplace=True)

# Remove duplicates
final_df.drop_duplicates(subset="text", inplace=True)

final_df["text"] = final_df["text"].apply(clean_text)

# Save final dataset
final_df.to_csv("dataset/final_dataset.csv", index=False)

print("Dataset Prepared Successfully")
print(final_df.shape)