"""Coursework Task 8: supervised and unsupervised categorical HMMs.

Run from the sequential directory:
    python hmms.py

Dependencies: numpy, pandas, matplotlib, pymc, hmmlearn.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from hmmlearn.hmm import CategoricalHMM

RANDOM_STATE = 42
N_STATES = 3
N_OBSERVATIONS = 3
N_GENERATED = 10_000
LABELS = ["Low", "Medium", "High"]


def load_data() -> pd.DataFrame:
    """Read the coursework dataset, using PyMC's example-data fallback."""
    local_paths = [
        Path(__file__).resolve().parent / "data" / "deaths_and_temps_england_wales.csv",
        Path(__file__).resolve().parent.parent / "data" / "deaths_and_temps_england_wales.csv",
    ]
    for path in local_paths:
        if path.is_file():
            df = pd.read_csv(path)
            break
    else:
        df = pd.read_csv(pm.get_data("deaths_and_temps_england_wales.csv"))

    required = {"date", "temp", "deaths"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}; found {set(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["temp", "deaths"])
    return df.reset_index(drop=True)


def discretize(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-based low/medium/high categories (0, 1, 2)."""
    thresholds = np.quantile(values.to_numpy(dtype=float), [1 / 3, 2 / 3])
    if thresholds[0] >= thresholds[1]:
        raise ValueError("Quantile boundaries coincide; cannot create three categories")
    categories = np.digitize(values.to_numpy(dtype=float), thresholds)
    return categories.astype(int), thresholds


def supervised_hmm(states: np.ndarray, observations: np.ndarray) -> CategoricalHMM:
    """Estimate initial, transition and emission probabilities from known states.

    Add-one smoothing prevents zero probabilities for rare transitions.
    """
    start_counts = np.ones(N_STATES, dtype=float)
    start_counts[states[0]] += 1

    transition_counts = np.ones((N_STATES, N_STATES), dtype=float)
    np.add.at(transition_counts, (states[:-1], states[1:]), 1)

    emission_counts = np.ones((N_STATES, N_OBSERVATIONS), dtype=float)
    np.add.at(emission_counts, (states, observations), 1)

    model = CategoricalHMM(n_components=N_STATES, n_features=N_OBSERVATIONS,
                           init_params="", params="", random_state=RANDOM_STATE)
    model.startprob_ = start_counts / start_counts.sum()
    model.transmat_ = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    model.emissionprob_ = emission_counts / emission_counts.sum(axis=1, keepdims=True)
    return model


def unsupervised_hmm(observations: np.ndarray) -> CategoricalHMM:
    """Fit an HMM to death categories alone using Baum-Welch (EM)."""
    model = CategoricalHMM(n_components=N_STATES, n_features=N_OBSERVATIONS,
                           n_iter=500, tol=1e-5, random_state=RANDOM_STATE,
                           implementation="log")
    model.fit(observations.reshape(-1, 1))
    return model


def sample_observations(model: CategoricalHMM, n_samples: int) -> np.ndarray:
    samples, _ = model.sample(n_samples=n_samples, random_state=RANDOM_STATE + 1)
    return samples.ravel()


def proportions(categories: np.ndarray) -> np.ndarray:
    return np.bincount(categories, minlength=N_OBSERVATIONS) / len(categories)


def print_model(name: str, model: CategoricalHMM) -> None:
    print(f"\n{name}")
    print("Initial probabilities:", np.round(model.startprob_, 4))
    print("Transition probabilities:\n", np.round(model.transmat_, 4))
    print("Emission probabilities:\n", np.round(model.emissionprob_, 4))


def main() -> None:
    df = load_data()
    temp_states, temp_thresholds = discretize(df["temp"])
    death_obs, death_thresholds = discretize(df["deaths"])

    supervised = supervised_hmm(temp_states, death_obs)
    unsupervised = unsupervised_hmm(death_obs)

    print(f"Monthly observations: {len(df)}")
    print("Temperature thresholds:", np.round(temp_thresholds, 3))
    print("Death thresholds:", np.round(death_thresholds, 3))
    print_model("HMM 1 — supervised (temperature states)", supervised)
    print_model("HMM 2 — unsupervised (latent states)", unsupervised)
    print("HMM 2 converged:", unsupervised.monitor_.converged)

    generated_supervised = sample_observations(supervised, N_GENERATED)
    generated_unsupervised = sample_observations(unsupervised, N_GENERATED)

    distribution = pd.DataFrame({
        "Actual": proportions(death_obs),
        "Supervised HMM": proportions(generated_supervised),
        "Unsupervised HMM": proportions(generated_unsupervised),
    }, index=LABELS)
    print("\nDeath-category distributions (proportions):")
    print(distribution.round(4).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(N_OBSERVATIONS)
    width = 0.25
    for i, column in enumerate(distribution.columns):
        ax.bar(x + (i - 1) * width, distribution[column], width, label=column)
    ax.set_xticks(x, LABELS)
    ax.set_xlabel("Monthly deaths category")
    ax.set_ylabel("Proportion")
    ax.set_title("Actual vs HMM-generated death distributions")
    ax.legend()
    ax.set_ylim(0, max(0.45, distribution.to_numpy().max() * 1.15))
    fig.tight_layout()

    output = Path(__file__).resolve().parent / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "hmm_death_distributions.png", dpi=180)
    distribution.to_csv(output / "hmm_death_distributions.csv", index_label="Category")
    print(f"\nSaved figure and CSV in: {output}")
    plt.show()


if __name__ == "__main__":
    main()
