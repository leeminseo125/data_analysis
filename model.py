import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp_classifier(df, epochs=500, batch_size=32, lr=1e-3, plot=True):
    # 1. 입력 및 타겟 분리
    X = np.stack(df["keypoints"].values)  # shape: (N, 33)
    y = df["category"].values

    # 2. Tensor 변환
    X_tensor = torch.tensor(X, dtype=torch.float32)
    le = LabelEncoder()
    y_tensor = torch.tensor(le.fit_transform(y), dtype=torch.long)

    # 3. 데이터셋 / 로더
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. 모델, 손실함수, 옵티마이저
    model = MLP(input_dim=X.shape[1], num_classes=len(le.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 5. 학습 루프
    loss_list = []
    acc_list = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
        avg_loss = running_loss / total
        acc = correct / total
        loss_list.append(avg_loss)
        acc_list.append(acc)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    # 6. 학습 곡선 시각화
    if plot:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(loss_list, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss')
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(acc_list, label='Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train Accuracy')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 7. 학습된 모델과 라벨 인코더 반환
    return model, le
