from datasets import load_dataset
import pandas as pd

train_dataset = load_dataset("Dataset/split", split="train")
valid_dataset = load_dataset("Dataset/split", split="validation")
test_dataset  = load_dataset("Dataset/split", split="test")

df_train = pd.DataFrame(train_dataset)
df_valid = pd.DataFrame(valid_dataset)
df_test = pd.DataFrame(test_dataset)

df_train = df_train.drop_duplicates(keep='first')
df_valid = df_valid.drop_duplicates(keep='first')
df_test = df_test.drop_duplicates(keep='first')

# df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

df_train = df_train.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_valid = df_valid.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_test = df_test.applymap(lambda x: x.lower() if isinstance(x, str) else x)

print(df_train)
print(df_valid)
print(df_test)