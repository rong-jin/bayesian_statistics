 
# # Lesson 6: Convergence diagnostics
# *(Translated from the original R notebook to Python; narrative text unchanged)*
# 
# In the previous two lessons, we have demonstrated ways you can simulate a Markov chain whose stationary
# distribution is the target distribution (usually the posterior). Before using the simulated chain to obtain Monte Carlo
# estimates, we should first ask ourselves: **Has our simulated Markov chain converged to its stationary distribution
# yet?** Unfortunately, this is a difficult question to answer, but we can do several things to investigate.
# 
# ## Trace plots
# 
# A trace plot shows the history of a parameter value across iterations of the chain and lets us see precisely where
# the chain has been exploring.

 
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# ----------  helper Metropolis–Hastings sampler ----------
def mh(n, ybar, n_iter, mu_init=0.0, cand_sd=0.9):
    """Metropolis–Hastings sampler for the posterior of μ with
    independent Normal likelihood 𝒩(ȳ, 1/n) and a flat prior.

    Parameters
    ----------
    n : int
        Sample size of the data.
    ybar : float
        Sample mean of the data.
    n_iter : int
        Number of M-H iterations.
    mu_init : float, optional
        Initial value of μ.
    cand_sd : float, optional
        Standard deviation of the Normal proposal distribution.

    Returns
    -------
    dict
        ``{'mu': np.ndarray, 'accpt': float}``
    """
    mu = np.empty(n_iter, dtype=float)
    mu[0] = mu_init
    accepted = 0

    for t in range(1, n_iter):
        mu_prop = np.random.normal(mu[t - 1], cand_sd)

        # log-posterior ∝ −½ n (μ − ȳ)², so the log-ratio simplifies:
        log_alpha = -0.5 * n * ((mu_prop - ybar) ** 2 - (mu[t - 1] - ybar) ** 2)

        if np.log(np.random.rand()) < log_alpha:
            mu[t] = mu_prop
            accepted += 1
        else:
            mu[t] = mu[t - 1]

    return {"mu": mu, "accpt": accepted / (n_iter - 1)}

 
# Re-create the first chain from the R lesson
np.random.seed(61)
n, ybar = 30, 1.0  # <-- replace with your data summary values
post0 = mh(n=n, ybar=ybar, n_iter=10_000, mu_init=0.0, cand_sd=0.9)
print(f"acceptance rate = {post0['accpt']:.3f}")

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(post0["mu"][500:], lw=0.6)
ax.set_xlabel("Iteration")
ax.set_ylabel("$\\mu$")
ax.set_title("Trace plot (post0)")
plt.show()

 
# If the chain is stationary, it should not be showing any long-term trends; 
# the average value for the chain should be roughly flat. It should not be 
# wandering as in the following example:

np.random.seed(61)
post1 = mh(n=n, ybar=ybar, n_iter=1_000, mu_init=0.0, cand_sd=0.04)

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(post1["mu"][500:], lw=0.6)
ax.set_xlabel("Iteration")
ax.set_ylabel("$\\mu$")
ax.set_title("Trace plot (post1)")
plt.show()

 
# If this is the case, you need to run the chain many more iterations, as seen here:

np.random.seed(61)
post2 = mh(n=n, ybar=ybar, n_iter=100_000, mu_init=0.0, cand_sd=0.04)

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(post2["mu"], lw=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("$\\mu$")
ax.set_title("Trace plot (post2)")
plt.show()

 
# ## Monte Carlo effective sample size
# 
# One major difference between the two chains we’ve looked at is the **level of autocorrelation** in each.  
# Autocorrelation is a number between −1 and 1 that measures how linearly dependent the current value of the
# chain is on past values (called *lags*).

 
az.plot_autocorr(post0["mu"], max_lag=50)

 
# Autocorrelation is important because it tells us how much information is available in our Markov chain.
# Sampling 1000 iterations from a highly correlated chain yields *less* information about the stationary
# distribution than we would obtain from 1000 *independent* samples.

ess0 = az.ess(np.asarray(post0["mu"]))
ess2 = az.ess(np.asarray(post2["mu"]))
print(f"Effective sample size for post0 ≈ {ess0:.0f}")
print(f"Effective sample size for post2 ≈ {ess2:.0f}")

  
# We can also *thin* the highly correlated chain until autocorrelation is essentially 0:

import pandas as pd
from statsmodels.tsa.stattools import acf

# Find first lag where ACF≈0
lags = 500
acf_vals = acf(post2["mu"], nlags=lags, fft=False)
thin_lag = np.where(acf_vals < 0.02)[0][0]   # arbitrary small threshold

thin_idx = np.arange(thin_lag, len(post2["mu"]), thin_lag)
post2_thin = post2["mu"][thin_idx]
print(f"Thinning interval = {thin_lag}")
print(f"Length after thinning = {len(post2_thin)}")

az.plot_autocorr(post2_thin, max_lag=10)

 
# ## Burn-in
# 
# We have also seen how the initial value of the chain can affect how quickly the chain converges.  
# If our initial value is far from the bulk of the posterior distribution, then it may take a while for the chain to travel there.

 
np.random.seed(62)
post3 = mh(n=n, ybar=ybar, n_iter=500, mu_init=10.0, cand_sd=0.3)

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(post3["mu"], lw=0.6)
ax.axvline(100, color="red", ls="--", label="suggested burn-in")
ax.set_xlabel("Iteration")
ax.set_ylabel("$\\mu$")
ax.legend()
ax.set_title("Trace plot with burn-in (post3)")
plt.show()

 
# Clearly, the first ≈100 iterations do not reflect draws from the stationary distribution, so they should be discarded
# before we use this chain for Monte Carlo estimates. This is called the **burn-in** period.
# 
# ## Multiple chains and the Gelman–Rubin diagnostic

 
np.random.seed(61)
nsim = 500
starts = [15.0, -5.0, 7.0, 23.0, -17.0]
chains = []

for mu0 in starts:
    chains.append(mh(n=n, ybar=ybar, n_iter=nsim, mu_init=mu0, cand_sd=0.4)["mu"])

idata = az.convert_to_inference_data(np.array(chains))
az.plot_trace(idata, compact=True, figsize=(8, 4))

 
# The Gelman–Rubin statistic (also called R-hat) compares the variability within chains 
# to the variability between chains. Values close to 1 indicate convergence.

rhat = az.rhat(idata)
print(f"Gelman–Rubin R-hat = {float(rhat.sel(chain_draw='mu')):.2f}")

 
# From the plot, we can see that after about iteration ≈300, the shrink factor is essentially 1, indicating that by then
# we have probably reached convergence. Of course, we shouldn’t stop sampling as soon as we reach convergence;
# instead, this is where we should *begin* saving our samples for Monte Carlo estimation.
# 
# ## Monte Carlo estimation
# 
# If we are reasonably confident that our Markov chain has converged, then we can treat it as a Monte Carlo sample
# from the posterior distribution and calculate posterior quantities directly:

 
nburn = 1_000  # discard early iterations
mu_keep = post0["mu"][nburn:]

az.summary(mu_keep, hdi_prob=0.95)


# We can also ask, for example, for the posterior probability that μ > 1.0:

prob_mu_gt_1 = np.mean(mu_keep > 1.0)
print(f"Pr(μ > 1) ≈ {prob_mu_gt_1:.3f}")

 
# ---
# 
# > **Tip** If you need reliable *95 % posterior intervals*, check that the *effective sample size* is at least a few
# > thousand. A quick rule of thumb is that you need roughly  
# > $$\\text{ESS} \\gtrsim \\frac{100}{1-q},$$  
# > where *q* is the posterior tail probability you care about (e.g. $q=0.025$ for a two-sided 95 % interval).
