import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error


# 1. Set random seeds
np.random.seed(42)
torch.manual_seed(42)


# 2. Load dataset
df = pd.read_csv("regression_insurance.csv")

X = df.drop(columns=["charges"])
y = df["charges"]


# 3. Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Preprocess features
numerical_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 5. Standardize target using training data only
y_mean = y_train.mean()
y_std = y_train.std()

y_train_scaled = (y_train - y_mean) / y_std


# 6. Convert to PyTorch tensors
X_train_tensor = torch.tensor(
    X_train_processed, dtype=torch.float32
)
X_test_tensor = torch.tensor(
    X_test_processed, dtype=torch.float32
)
y_train_tensor = torch.tensor(
    y_train_scaled.to_numpy().reshape(-1, 1),
    dtype=torch.float32,
)


# 7. Define neural network
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


model = RegressionNN(X_train_tensor.shape[1])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 8. Train model
epochs = 500
loss_history = []

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")


# 9. Evaluate model
model.eval()

with torch.no_grad():
    train_pred_scaled = model(X_train_tensor).numpy().flatten()
    test_pred_scaled = model(X_test_tensor).numpy().flatten()

train_pred = train_pred_scaled * y_std + y_mean
test_pred = test_pred_scaled * y_std + y_mean

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print(f"\nNeural Network Train RMSE: {train_rmse:.3f}")
print(f"Neural Network Test RMSE: {test_rmse:.3f}")


# 10. Plot training loss
plt.figure(figsize=(8, 6))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Neural Network Training Loss")
plt.tight_layout()
plt.savefig("figures/neural_network_loss.png", dpi=300)
plt.show()


# 11. Plot actual vs predicted
plt.figure(figsize=(8, 6))

plt.scatter(y_test, test_pred, alpha=0.5)

min_value = min(y_test.min(), test_pred.min())
max_value = max(y_test.max(), test_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--",
    label="Perfect Prediction",
)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Neural Network: Actual vs Predicted")
plt.legend()
plt.tight_layout()

plt.savefig("figures/neural_network_scatter.png", dpi=300)
plt.show()