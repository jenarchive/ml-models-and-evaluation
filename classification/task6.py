import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)


# 1. Configuration
RANDOM_STATE = 42

DATA_PATH = Path("data/cifar10_pca200.npz")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# 2. Load PCA-reduced CIFAR-10 dataset
data = np.load(DATA_PATH)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")


# 3. Define Random Forest and cross-validation
model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=1,
)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE,
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
}


# 4. Hyperparameter tuning
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

print("\nStarting hyperparameter tuning...")

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")


# 5. Evaluate best model
best_model = grid_search.best_estimator_

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=class_names,
    )
)


# 6. Plot confusion matrix
fig, ax = plt.subplots(figsize=(11, 9))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_test_pred,
    display_labels=class_names,
    cmap="Blues",
    xticks_rotation=45,
    ax=ax,
)

ax.set_title("Random Forest: Confusion Matrix")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "random_forest_confusion_matrix.png",
    dpi=300,
)

plt.close()


# 7. Plot hyperparameter tuning results
results = grid_search.cv_results_

plt.figure(figsize=(9, 6))

for n_trees in param_grid["n_estimators"]:
    depths = []
    scores = []

    for depth in param_grid["max_depth"]:
        for i, params in enumerate(results["params"]):
            if (
                params["n_estimators"] == n_trees
                and params["max_depth"] == depth
            ):
                depths.append(
                    "None" if depth is None else str(depth)
                )
                scores.append(
                    results["mean_test_score"][i]
                )

    plt.plot(
        depths,
        scores,
        marker="o",
        label=f"n_estimators={n_trees}",
    )

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Mean CV Accuracy")
plt.title("Random Forest: Hyperparameter Tuning")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "random_forest_hyperparameters.png",
    dpi=300,
)

plt.close()

print("\nFigures saved successfully.")