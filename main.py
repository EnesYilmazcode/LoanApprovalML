import sys
import torch
from torch import nn
import pandas as pd

df = pd.read_csv("data/loan_data.csv")
df = df[["loan_status", "person_income", "loan_intent", "loan_percent_income", "credit_score"]]
df = pd.get_dummies(df, columns=["loan_intent"])


y = torch.tensor(df["loan_status"].values, dtype=torch.float32).reshape((-1, 1))
x = torch.tensor(df.drop("loan_status", axis=1).values, dtype=torch.float32)

x_mean = x.mean(dim=0)
x_std = x.std(dim=0)

# normalization
x = (x - x_mean) / x_std


model = nn.Sequential(
    nn.Linear(9, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
