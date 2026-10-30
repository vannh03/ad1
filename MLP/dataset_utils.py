# MLP/dataset_utils.py
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
LABEL_TO_ID = {
    "crossing": 0,
    "normal": 1,
    "wrong-way": 2,
    "wrong_way": 2,
    "wrong way": 2
}

ID_TO_LABEL = {
    0: "crossing",
    1: "normal",
    2: "wrong-way"
}

NUMERIC_FEATURES = ["x", "dx", "y", "dy", "angle_sin", "angle_cos", "distance", "speed"]
CATEGORICAL_FEATURES = ["roi_side"]

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, hidden_dim_2=32, hidden_dim_3=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_3, 3)
        )

    def forward(self, x):
        return self.net(x)

def prepare_features(df):
    df = df.copy()
    angles_rad = np.radians(df["angle"].astype(float))
    df["angle_sin"] = np.sin(angles_rad)
    df["angle_cos"] = np.cos(angles_rad)
    df["roi_side"] = df["roi_side"].fillna("None").astype(str)
    return df

def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor
