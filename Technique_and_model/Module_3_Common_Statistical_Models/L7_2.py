# ───────────── 0. Dependencies ───────────────────────
import numpy as np
import statsmodels.api as sm               # only for fetching the dataset
import pymc as pm                          # PyMC ≥ 5
import arviz as az

# ───────────── 1. Load + preprocess data ─────────────
dataset = sm.datasets.get_rdataset("Leinhardt", package="carData")
df = (
    dataset.data
        .dropna()                          # drop 4 rows with NAs (Nepal, Laos, Haiti, Iran)
        .assign(
            logInfant = lambda d: np.log(d["infant"]),
            logIncome = lambda d: np.log(d["income"]),
        )
)

y          = df["logInfant"].values
log_income = df["logIncome"].values
n          = len(df)

# ───────────── 2. Build Bayesian model ───────────────
rng = np.random.default_rng(2025)          # reproducible seed

with pm.Model() as mod1:
    # Regression coefficients: β_j ~ N(0, 1e6)
    beta = pm.Normal("beta", mu=0.0, sigma=np.sqrt(1e6), shape=2)

    # Precision prior: τ ~ Gamma(α=2.5, β=25)   ↔ Inv-Gamma on σ²
    tau = pm.Gamma("tau", alpha=2.5, beta=25.0)
    sigma2 = pm.Deterministic("sigma2", 1.0 / tau)
    sigma  = pm.Deterministic("sigma", pm.math.sqrt(sigma2))

    # Linear predictor   μ_i = β₁ + β₂·log_income_i
    mu = beta[0] + beta[1] * log_income

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, tau=tau, observed=y)

# ───────────── 3. Sampling (3 chains) ────────────────
def run_sampling(model):
    with model:
        trace = pm.sample(
            draws         = 5000,   # kept iterations per chain
            tune          = 1000,   # burn-in (adaptation)
            chains        = 3,
            cores         = 3,      # ← Change to 1 to avoid Windows multi-process issues
            random_seed   = 2025,
            target_accept = 0.90,
        )
    return trace

# ───────────── 4. Main guard (needed on Windows) ─────
if __name__ == "__main__":
    trace = run_sampling(mod1)
    print(az.summary(trace, var_names=["beta", "sigma"], round_to=4))
