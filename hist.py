from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import os

label_names = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

train_dataset = load_dataset("Dataset/split", split="train")
valid_dataset = load_dataset("Dataset/split", split="validation")
test_dataset  = load_dataset("Dataset/split", split="test")

df_train = pd.DataFrame(train_dataset)
df_valid = pd.DataFrame(valid_dataset)
df_test = pd.DataFrame(test_dataset)

os.makedirs("plots", exist_ok=True)

def save_label_histogram(df, name):
    label_counts = df['label'].value_counts().sort_index()
    labels = [label_names[i] for i in label_counts.index]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, label_counts.values, color='skyblue')
    plt.title(f"Distribuția etichetelor în setul {name}")
    plt.xlabel("Emoție")
    plt.ylabel("Număr de exemple")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f"plots/label_distribution_{name}.png", dpi=300)
    plt.close()

save_label_histogram(df_train, "train")
save_label_histogram(df_valid, "validation")
save_label_histogram(df_test, "test")