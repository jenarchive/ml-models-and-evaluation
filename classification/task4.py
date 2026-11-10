import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from torchvision.datasets import CIFAR10

# Configuration
RANDOM_STATE = 42
N_COMPONENTS = 200

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# 1. Load CIFAR-10
train_dataset = CIFAR10(root=DATA_DIR, train=True, download=False)
test_dataset = CIFAR10(root=DATA_DIR, train=False, download=False)

X_train = train_dataset.data.reshape(-1, 3072).astype(np.float32)
X_test = test_dataset.data.reshape(-1, 3072).astype(np.float32)

y_train = np.array(train_dataset.targets)
y_test = np.array(test_dataset.targets)

print(f"Original train shape: {X_train.shape}")
print(f"Original test shape: {X_test.shape}")

# 2. Normalize pixel values
X_train /= 255.0
X_test /= 255.0

# 3. Reduce features from 3072 to 200 using PCA
pca = PCA(
    n_components=N_COMPONENTS,
    svd_solver="randomized",
    random_state=RANDOM_STATE,
)

X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# 4. Print results
print(f"Reduced train shape: {X_train_reduced.shape}")
print(f"Reduced test shape: {X_test_reduced.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# 5. Save processed data
np.savez_compressed(
    DATA_DIR / "cifar10_pca200.npz",
    X_train=X_train_reduced,
    X_test=X_test_reduced,
    y_train=y_train,
    y_test=y_test,
)

print("Processed dataset saved successfully.")