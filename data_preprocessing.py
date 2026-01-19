import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


train_df = pd.read_csv("train_cleaned.csv")
test_df = pd.read_csv("test_cleaned.csv")

for idx in range(0, train_df.shape[0]):
    index, filename, label = train_df.iloc[idx][0], train_df.iloc[idx][1], train_df.iloc[idx][2]
    print(filename)