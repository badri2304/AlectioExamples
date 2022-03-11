import os
import numpy as np
import pandas as pd

data_path = './data/data.csv'

train_size = 0.8
df = pd.read_csv(data_path).fillna("none")
train_dataset = df.sample(frac=train_size, random_state=42)
test_dataset = df.drop(train_dataset.index)

train_dataset.to_csv('./data/train.csv', index=False, header=True)
test_dataset.to_csv('./data/test.csv', index=False, header=True)