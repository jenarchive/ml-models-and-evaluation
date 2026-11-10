import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)


# 1. Configuration
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cifar10_pca200.npz"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


# 2. Load PCA-reduced CIFAR-10 dataset
with np.load(DATA_PATH) as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")


# 3. Define cross-validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# 4. Define SVM models and hyperparameter grids
models = {
    "Linear SVM": {
        "pipeline": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    LinearSVC(
                        dual=False,
                        max_iter=10000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "param_grid": {
            "svm__C": [0.01, 0.1, 1, 10],
        },
    },
    "RBF SVM": {
        "pipeline": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        cache_size=1000,
                    ),
                ),
            ]
        ),
        "param_grid": {
            "svm__C": [1, 10],
            "svm__gamma": ["scale", 0.01],
        },
    },
}


# 5. Hyperparameter tuning
best_models = {}
cv_results = {}
best_cv_scores = {}

for model_name, config in models.items():
    print(f"\n{'=' * 50}")
    print(f"Training {model_name}")
    print(f"{'=' * 50}")

    grid_search = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["param_grid"],
        scoring="accuracy",
        cv=cv,
        n_jobs=2,
        verbose=2,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    best_models[model_name] = grid_search.best_estimator_
    cv_results[model_name] = grid_search.cv_results_
    best_cv_scores[model_name] = grid_search.best_score_

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")


# 6. Select the best SVM model
best_model_name = max(
    best_cv_scores,
    key=best_cv_scores.get,
)

best_model = best_models[best_model_name]

print(f"\n{'=' * 50}")
print("Best SVM Model")
print(f"{'=' * 50}")

print(f"Model: {best_model_name}")
print(f"Best CV Accuracy: {best_cv_scores[best_model_name]:.4f}")


# 7. Evaluate best model
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


# 8. Plot confusion matrix
fig, ax = plt.subplots(figsize=(11, 9))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_test_pred,
    display_labels=class_names,
    cmap="Blues",
    xticks_rotation=45,
    ax=ax,
)

ax.set_title(f"{best_model_name}: Confusion Matrix")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "svm_confusion_matrix.png",
    dpi=300,
)

plt.close()


# 9. Plot hyperparameter tuning results
labels = []
scores = []

for model_name, results in cv_results.items():
    for params, score in zip(
        results["params"],
        results["mean_test_score"],
    ):
        c_value = params["svm__C"]

        if model_name == "RBF SVM":
            gamma = params["svm__gamma"]
            label = f"RBF\nC={c_value}\ngamma={gamma}"
        else:
            label = f"Linear\nC={c_value}"

        labels.append(label)
        scores.append(score)

plt.figure(figsize=(12, 6))

plt.bar(
    range(len(scores)),
    scores,
)

plt.xticks(
    range(len(labels)),
    labels,
    rotation=30,
    ha="right",
)

plt.xlabel("Hyperparameter Configuration")
plt.ylabel("Mean CV Accuracy")
plt.title("SVM: Hyperparameter Tuning")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "svm_hyperparameters.png",
    dpi=300,
)

plt.close()


# 10. Compare classification models
print(f"\n{'=' * 50}")
print("Classification Model Comparison")
print(f"{'=' * 50}")

print(f"Decision Tree Test Accuracy: 0.3110")
print(f"Random Forest Test Accuracy: 0.4655")
print(f"SVM Test Accuracy: {test_accuracy:.4f}")

print("\nFigures saved successfully.")