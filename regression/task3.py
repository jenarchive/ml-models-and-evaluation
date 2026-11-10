import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error


# 1. Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# 2. Load dataset
df = pd.read_csv("regression_insurance.csv")

X = df.drop(columns=["charges"])
y = df["charges"]


# 3. Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)


# 4. Preprocess features
numerical_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

X_train_processed = np.asarray(
    preprocessor.fit_transform(X_train),
    dtype=np.float64,
)

X_test_processed = np.asarray(
    preprocessor.transform(X_test),
    dtype=np.float64,
)

feature_names = preprocessor.get_feature_names_out()
n_features = X_train_processed.shape[1]


# 5. Standardize target using training data only
y_mean = y_train.mean()
y_std = y_train.std()

y_train_scaled = (
    (y_train.to_numpy() - y_mean) / y_std
)


# 6. Define Bayesian Linear Regression model
with pm.Model() as bayesian_model:

    # Priors
    intercept = pm.Normal(
        "intercept",
        mu=0,
        sigma=10,
    )

    coefficients = pm.Normal(
        "coefficients",
        mu=0,
        sigma=10,
        shape=n_features,
    )

    sigma = pm.HalfNormal(
        "sigma",
        sigma=1,
    )

    # Expected value
    mu = intercept + pm.math.dot(
        X_train_processed,
        coefficients,
    )

    # Likelihood
    observations = pm.Normal(
        "observations",
        mu=mu,
        sigma=sigma,
        observed=y_train_scaled,
    )

    # Posterior sampling
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=2,
        random_seed=RANDOM_STATE,
        target_accept=0.9,
        return_inferencedata=True,
    )


# 7. Summarize posterior distributions
print("\nBayesian Regression Posterior Summary:")

summary = az.summary(
    trace,
    var_names=["intercept", "coefficients", "sigma"],
    round_to=3,
)

print(summary)


# 8. Extract posterior mean coefficients
intercept_mean = (
    trace.posterior["intercept"]
    .mean(dim=("chain", "draw"))
    .values
)

coefficients_mean = (
    trace.posterior["coefficients"]
    .mean(dim=("chain", "draw"))
    .values
)

print(f"\nPosterior Mean Intercept: {float(intercept_mean):.3f}")

print("\nPosterior Mean Coefficients:")

for feature, coefficient in zip(
    feature_names,
    coefficients_mean,
):
    print(f"{feature}: {coefficient:.3f}")


# 9. Make predictions using posterior mean
train_pred_scaled = (
    intercept_mean
    + X_train_processed @ coefficients_mean
)

test_pred_scaled = (
    intercept_mean
    + X_test_processed @ coefficients_mean
)

train_pred = train_pred_scaled * y_std + y_mean
test_pred = test_pred_scaled * y_std + y_mean


# 10. Calculate RMSE
train_rmse = np.sqrt(
    mean_squared_error(y_train, train_pred)
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_pred)
)

print(f"\nBayesian Regression Train RMSE: {train_rmse:.3f}")
print(f"Bayesian Regression Test RMSE: {test_rmse:.3f}")


# 11. Plot posterior distributions
posterior = trace.posterior

parameter_samples = {
    "intercept": posterior["intercept"].values.flatten(),
    "sigma": posterior["sigma"].values.flatten(),
}

for i, feature in enumerate(feature_names):
    parameter_samples[feature] = (
        posterior["coefficients"]
        .values[:, :, i]
        .flatten()
    )

n_params = len(parameter_samples)
n_cols = 2
n_rows = int(np.ceil(n_params / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(14, 3 * n_rows),
)

axes = np.asarray(axes).flatten()

for ax, (name, samples) in zip(
    axes,
    parameter_samples.items(),
):
    ax.hist(
        samples,
        bins=40,
        density=True,
        alpha=0.7,
    )

    ax.axvline(
        samples.mean(),
        color="red",
        linestyle="--",
        label="Posterior Mean",
    )

    ax.set_title(name)
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Density")
    ax.legend()

for ax in axes[n_params:]:
    ax.set_visible(False)

plt.suptitle("Bayesian Regression: Posterior Distributions")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "bayesian_posterior.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# 12. Plot actual vs predicted values
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    test_pred,
    alpha=0.5,
)

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
plt.title("Bayesian Regression: Actual vs Predicted")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "bayesian_regression_scatter.png",
    dpi=300,
)

plt.close()

print("\nFigures saved successfully.")