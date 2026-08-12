"""Estimate bounded distance-decay relations for five metro systems.

The model is fitted to 0.5 km distance bins. For each bin, f_r is observed
metro flow, p_r is the number of represented directed station pairs, and
pi_r = f_r / p_r is flow per represented pair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROFILE_FILE = ROOT / "data" / "metro_distance_profiles.csv"
SAMPLE_SIZE_FILE = ROOT / "data" / "effective_sample_sizes.csv"
OUTPUT_FILE = ROOT / "results" / "metro_law_estimates.csv"

CITY_ORDER = ["Tokyo", "London", "Tianjin", "Taipei", "NYC"]
DISPLAY_ORDER = ["London", "NYC", "Tokyo", "Taipei", "Tianjin"]
BIN_WIDTH_KM = 0.5
MIN_BINS = 12
BOOTSTRAP_REPLICATES = 800
RANDOM_SEED = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load model inputs and verify the released data contract."""
    profiles = pd.read_csv(PROFILE_FILE)
    required = {"city", "r_mid_km", "f_r", "p_r", "pi_r"}
    missing = required.difference(profiles.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if profiles[["city", "r_mid_km"]].duplicated().any():
        raise ValueError("City-distance records must be unique.")
    if (profiles["r_mid_km"] <= 0).any() or (profiles["p_r"] <= 0).any():
        raise ValueError("Distances and pair counts must be positive.")
    if (profiles["f_r"] < 0).any():
        raise ValueError("Flows must be non-negative.")
    if not np.allclose(
        profiles["pi_r"], profiles["f_r"] / profiles["p_r"],
        rtol=1e-10, atol=1e-10,
    ):
        raise ValueError("pi_r is inconsistent with f_r / p_r.")

    sample_sizes = pd.read_csv(SAMPLE_SIZE_FILE).set_index("city")["neff_kish"]
    if set(CITY_ORDER).difference(sample_sizes.index):
        raise ValueError("Effective sample sizes are incomplete.")
    return profiles, sample_sizes


def estimate_alpha(flow: np.ndarray, pairs: np.ndarray, distance: np.ndarray) -> float:
    """Solve the discrete maximum-likelihood score equation for alpha."""
    log_r = np.log(distance)
    log_p = np.log(pairs)
    empirical_mean = float(np.sum(flow * log_r) / np.sum(flow))

    def score_and_variance(alpha: float) -> tuple[float, float]:
        log_weights = log_p - alpha * log_r
        log_weights -= np.max(log_weights)
        probabilities = np.exp(log_weights)
        probabilities /= probabilities.sum()
        model_mean = float(np.sum(probabilities * log_r))
        variance = float(np.sum(probabilities * (log_r - model_mean) ** 2))
        return model_mean - empirical_mean, variance

    lower, upper = 1e-6, 6.0
    score_lower, _ = score_and_variance(lower)
    score_upper, _ = score_and_variance(upper)
    while score_lower * score_upper > 0 and upper < 6144:
        upper *= 2
        score_upper, _ = score_and_variance(upper)
    if score_lower * score_upper > 0:
        return float(lower if abs(score_lower) < abs(score_upper) else upper)

    alpha = 1.0
    for _ in range(40):
        score, variance = score_and_variance(alpha)
        if abs(score) < 1e-10:
            break
        if score > 0:
            lower = alpha
        else:
            upper = alpha
        proposal = alpha + score / variance if variance > 0 else np.nan
        alpha = proposal if np.isfinite(proposal) and lower < proposal < upper else (lower + upper) / 2
    return float(alpha)


def model_probabilities(
    alpha: float, pairs: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    """Return normalized flow probabilities under the fitted model."""
    log_weights = np.log(pairs) - alpha * np.log(distance)
    log_weights -= np.max(log_weights)
    probabilities = np.exp(log_weights)
    return probabilities / probabilities.sum()


def ks_distance(flow: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate the distance between empirical and fitted cumulative flows."""
    empirical = np.cumsum(flow) / flow.sum()
    fitted = np.cumsum(probabilities)
    return float(np.max(np.abs(empirical - fitted)))


def select_window(city_data: pd.DataFrame) -> dict[str, float | int | str]:
    """Search all contiguous intervals and retain the minimum-KS fit."""
    data = city_data.sort_values("r_mid_km").reset_index(drop=True)
    r = data["r_mid_km"].to_numpy(float)
    f = data["f_r"].to_numpy(float)
    p = data["p_r"].to_numpy(float)
    best = {"ks": np.inf, "alpha": np.nan, "start": -1, "stop": -1}

    for start in range(len(data) - MIN_BINS + 1):
        for stop in range(start + MIN_BINS, len(data) + 1):
            flow = f[start:stop]
            if flow.sum() <= 0:
                continue
            pairs = p[start:stop]
            distance = r[start:stop]
            alpha = estimate_alpha(flow, pairs, distance)
            ks = ks_distance(flow, model_probabilities(alpha, pairs, distance))
            if ks < best["ks"]:
                best = {"ks": ks, "alpha": alpha, "start": start, "stop": stop}

    start, stop = int(best["start"]), int(best["stop"])
    if start < 0:
        raise RuntimeError(f"No valid window found for {data['city'].iloc[0]}.")
    a_km = float(r[start] - BIN_WIDTH_KM / 2)
    b_km = float(r[stop - 1] + BIN_WIDTH_KM / 2)
    return {
        "city": str(data["city"].iloc[0]),
        "a_km": a_km,
        "b_km": b_km,
        "window_width_km": b_km - a_km,
        "alpha_hat": float(best["alpha"]),
        "ks_distance": float(best["ks"]),
        "bins": stop - start,
        "start": start,
        "stop": stop,
        "trips_in_window": float(f[start:stop].sum()),
        "pairs_in_window": float(p[start:stop].sum()),
    }


def bootstrap_fit(
    city_data: pd.DataFrame,
    fit: dict[str, float | int | str],
    effective_sample_size: float,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Estimate alpha uncertainty and a parametric KS goodness-of-fit p-value."""
    data = city_data.sort_values("r_mid_km").reset_index(drop=True)
    start, stop = int(fit["start"]), int(fit["stop"])
    window = data.iloc[start:stop]
    r = window["r_mid_km"].to_numpy(float)
    f = window["f_r"].to_numpy(float)
    p = window["p_r"].to_numpy(float)

    fitted = model_probabilities(float(fit["alpha_hat"]), p, r)
    observed_ks = ks_distance(f, fitted)
    sample_size = max(int(round(effective_sample_size)), 1)
    alpha_samples = np.empty(BOOTSTRAP_REPLICATES)
    ks_samples = np.empty(BOOTSTRAP_REPLICATES)

    for index in range(BOOTSTRAP_REPLICATES):
        simulated_flow = rng.multinomial(sample_size, fitted).astype(float)
        alpha = estimate_alpha(simulated_flow, p, r)
        alpha_samples[index] = alpha
        ks_samples[index] = ks_distance(
            simulated_flow, model_probabilities(alpha, p, r)
        )

    ci_lower, ci_upper = np.quantile(alpha_samples, [0.025, 0.975])
    return {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "effective_sample_size": effective_sample_size,
        "bootstrap_p_value": float(np.mean(ks_samples >= observed_ks)),
        "alpha_ci_lower": float(ci_lower),
        "alpha_ci_upper": float(ci_upper),
    }


def run_analysis() -> pd.DataFrame:
    """Run the complete five-city analysis and write the result table."""
    profiles, sample_sizes = load_data()
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for city in CITY_ORDER:
        city_data = profiles.loc[profiles["city"] == city]
        fit = select_window(city_data)
        uncertainty = bootstrap_fit(city_data, fit, sample_sizes.loc[city], rng)
        fit.pop("start")
        fit.pop("stop")
        rows.append(fit | uncertainty)

    results = pd.DataFrame(rows).set_index("city").loc[DISPLAY_ORDER].reset_index()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    return results


if __name__ == "__main__":
    estimates = run_analysis()
    columns = [
        "city", "a_km", "b_km", "alpha_hat", "ks_distance",
        "bootstrap_p_value", "alpha_ci_lower", "alpha_ci_upper",
    ]
    print(estimates[columns].to_string(index=False))
