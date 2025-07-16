import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma

# 数据：10 个公司的人员变动百分比
# y = np.array([1.2, 0.8, 1.0, 0.9, 1.1, 0.95, 1.05, 0.85, 1.15, 1.0])
y = np.array([-0.2, -1.5, -5.3, 0.3, -0.8, -2.2])
ybar = np.mean(y)
n = len(y)

# Prior for mu ~ N(mu_0, sig2_0)
mu_0 = 0
sig2_0 = 1

# Prior for sigma^2 ~ Inv-Gamma(nu_0/2, beta_0/2)
n_0 = 2             # prior effective sample size
s2_0 = 1            # prior guess of sigma^2
nu_0 = n_0 / 2
beta_0 = n_0 * s2_0 / 2

prior = {
    'mu_0': mu_0,
    'sig2_0': sig2_0,
    'nu_0': nu_0,
    'beta_0': beta_0
}

def update_mu(n, ybar, sig2, mu_0, sig2_0):
    post_var = 1 / (n / sig2 + 1 / sig2_0)
    post_mean = post_var * (n * ybar / sig2 + mu_0 / sig2_0)
    return np.random.normal(post_mean, np.sqrt(post_var))

def update_sig2(n, y, mu, nu_0, beta_0):
    rss = np.sum((y - mu)**2)
    post_nu = nu_0 + n / 2
    post_beta = beta_0 + 0.5 * rss
    return invgamma.rvs(a=post_nu, scale=post_beta)

def gibbs(y, n_iter, init, prior):
    n = len(y)
    ybar = np.mean(y)
    
    mu_out = np.zeros(n_iter)
    sig2_out = np.zeros(n_iter)
    
    mu_now = init['mu']
    
    for i in range(n_iter):
        sig2_now = update_sig2(n, y, mu_now, prior['nu_0'], prior['beta_0'])
        mu_now = update_mu(n, ybar, sig2_now, prior['mu_0'], prior['sig2_0'])
        
        sig2_out[i] = sig2_now
        mu_out[i] = mu_now

    return np.column_stack((mu_out, sig2_out))

np.random.seed(53)
init = {'mu': 0}
n_iter = 5000

post = gibbs(y=y, n_iter=n_iter, init=init, prior=prior)
mu_samples = post[:, 0]
sig2_samples = post[:, 1]

fig, axs = plt.subplots(2, 2, figsize=(12, 6))

# Trace plots
axs[0, 0].plot(mu_samples)
axs[0, 0].set_title('Trace of mu')
axs[1, 0].plot(sig2_samples)
axs[1, 0].set_title('Trace of sigma^2')

# Posterior densities
axs[0, 1].hist(mu_samples, bins=40, density=True, alpha=0.7)
axs[0, 1].set_title('Posterior of mu')
axs[1, 1].hist(sig2_samples, bins=40, density=True, alpha=0.7)
axs[1, 1].set_title('Posterior of sigma^2')

plt.tight_layout()
plt.show()

print("Posterior mean of mu:", np.mean(mu_samples))
print("Posterior sd of mu:", np.std(mu_samples, ddof=1))

print("Posterior mean of sigma^2:", np.mean(sig2_samples))
print("Posterior sd of sigma^2:", np.std(sig2_samples, ddof=1))
