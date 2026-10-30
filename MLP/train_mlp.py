# MLP/train_mlp.py
import os
import sys
import time
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from configs.settings import MODEL_MLP_PATH, SCALER_PATH
from MLP.dataset_utils import (
    LABEL_TO_ID,
    ID_TO_LABEL,
    MLPModel,
    prepare_features,
    build_preprocessor
)
def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình MLP phân loại hành vi")
    parser.add_argument("--csv-path", type=str, default="", help="Đường dẫn file dataset CSV")
    parser.add_argument("--epochs", type=int, default=60, help="Số epochs huấn luyện")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Tốc độ học (Learning Rate)")
    parser.add_argument("--save", action="store_true", default=True, help="Lưu đè model và scaler vào models/decision/")
    return parser.parse_args()

def train_model():
    args = parse_args()
    csv_file = None
    if args.csv_path and Path(args.csv_path).exists():
        csv_file = Path(args.csv_path)
    elif (ROOT / "MLP" / "MLP_dataset.csv").exists():
        csv_file = ROOT / "MLP" / "MLP_dataset.csv"
    elif Path(r"D:\SYLLABUS\AIP491\data\MLP\MLP_dataset.csv").exists():
        csv_file = Path(r"D:\SYLLABUS\AIP491\data\MLP\MLP_dataset.csv")
    if csv_file is None or not csv_file.exists():
        print(f"Không tìm thấy file dataset CSV! Hãy chạy MLP/collect_dataset.py trước.")
        return

    print("=" * 65)
    print("BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH MLP (AIP491)")
    print(f"Dataset:   {csv_file}")
    print(f"Epochs:    {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 65)
    df = pd.read_csv(csv_file)
    print(f"Tổng số mẫu ban đầu: {len(df)}")
    df = df.dropna(subset=["x", "y", "status"])
    df["status_clean"] = df["status"].astype(str).str.lower().str.strip()
    df["target"] = df["status_clean"].map(LABEL_TO_ID)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    print(f"Phân bố các lớp sau làm sạch:")
    for label_str, count in df["status_clean"].value_counts().items():
        print(f"   • {label_str:<12}: {count} mẫu")
    if len(df) < 50:
        print("Dữ liệu quá ít (< 50 mẫu). Vui lòng thu thập thêm video trước khi train!")
        return
    df_prepared = prepare_features(df)
    preprocessor = build_preprocessor()
    X_raw = df_prepared[["x", "dx", "y", "dy", "angle_sin", "angle_cos", "distance", "speed", "roi_side"]]
    y_raw = df_prepared["target"].values
    X_transformed = preprocessor.fit_transform(X_raw)
    input_dim = X_transformed.shape[1]
    print(f"\n Số chiều vector đầu vào sau khi tiền xử lý (Input Dim): {input_dim}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    classes, counts = np.unique(y_train, return_counts=True)
    weights = len(y_train) / (len(classes) * counts.astype(float))
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Huấn luyện trên thiết bị: {device}")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = MLPModel(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print("\n Bắt đầu quá trình huấn luyện...")
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)
        epoch_loss = train_loss / total
        epoch_acc = correct / total
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                val_loss += loss.item() * len(y_b)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_state = model.state_dict().copy()
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"   [Epoch {epoch:02d}/{args.epochs}] Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc*100:5.1f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:5.1f}%")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            outputs = model(X_b)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.numpy())
    target_names = ["crossing (0)", "normal (1)", "wrong-way (2)"]
    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    print("CONFUSION MATRIX:")
    print(confusion_matrix(all_labels, all_preds))
    print("=" * 65)
    if args.save:
        MODEL_MLP_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(MODEL_MLP_PATH))
        joblib.dump(preprocessor, str(SCALER_PATH))
        print("\n ĐÃ LƯU THÀNH CÔNG VÀO HỆ THỐNG:")
        print(f"   • Model Weights: {MODEL_MLP_PATH}")
        print(f"   • Scaler:        {SCALER_PATH}")
if __name__ == "__main__":
    train_model()
