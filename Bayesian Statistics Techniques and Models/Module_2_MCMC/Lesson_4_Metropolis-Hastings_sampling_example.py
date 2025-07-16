import numpy as np
import matplotlib.pyplot as plt

# 1.定义 log posterior 函数
def log_g(mu, ybar, n):
    mu2 = mu**2
    return n * (ybar * mu - mu2 / 2) - np.log(1.0 + mu2)

# 2.实现随机游走 Metropolis-Hastings 采样器
def metropolis_hastings(ybar, n, n_iter, mu_init, proposal_sd):
    mu_out = np.zeros(n_iter)
    accpt = 0

    mu_now = mu_init
    lg_now = log_g(mu_now, ybar, n)

    for i in range(n_iter):
        # Step a: propose candidate
        mu_cand = np.random.normal(loc=mu_now, scale=proposal_sd)

        # Step b: compute acceptance ratio on log scale
        lg_cand = log_g(mu_cand, ybar, n)
        log_alpha = lg_cand - lg_now
        alpha = np.exp(log_alpha)

        # Step c: accept or reject
        u = np.random.uniform()
        if u < alpha:
            mu_now = mu_cand
            lg_now = lg_cand
            accpt += 1

        # Save sample
        mu_out[i] = mu_now

    return {
        "mu": mu_out,
        "accept_rate": accpt / n_iter
    }

# 3.运行示例
# 假设数据来自 10 个公司的人员增长率
# 模拟一些 y 值
np.random.seed(123)
y = np.random.normal(loc=3, scale=1, size=10)
ybar = np.mean(y)
n = len(y)

# Metropolis-Hastings 参数
n_iter = 5000
mu_init = 0.0
proposal_sd = 0.5

# 运行采样器
result = metropolis_hastings(ybar, n, n_iter, mu_init, proposal_sd)

mu_samples = result["mu"]
accept_rate = result["accept_rate"]

print(f"接受率: {accept_rate:.3f}")

# 4.可视化结果
# 绘制 mu 的采样轨迹
plt.figure(figsize=(12, 4))
plt.plot(mu_samples, lw=0.5)
plt.title("Metropolis-Hastings Sampling Trace of mu")
plt.xlabel("Iteration")
plt.ylabel("mu")
plt.grid(True)
plt.show()

# 绘制 mu 的后验近似分布
plt.hist(mu_samples, bins=50, density=True, alpha=0.6, color='skyblue')
plt.title("Posterior Distribution Approximation of mu")
plt.xlabel("mu")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# 设置随机种子以确保可复现
np.random.seed(123)

# 模拟 10 个公司的人员变动百分比
y = np.array([1.2, 0.8, 1.0, 0.9, 1.1, 0.95, 1.05, 0.85, 1.15, 1.0])
ybar = np.mean(y)
n = len(y)

print(f"Sample mean: {ybar}, Sample size: {n}")

# 绘制数据直方图
plt.hist(y, bins=6, density=True, alpha=0.6, edgecolor='black', range=(-1, 3))
plt.scatter(y, np.zeros_like(y), color='black')  # 添加数据点
plt.scatter([ybar], [0], color='red', marker='o', label='Sample Mean')

# 添加先验分布（t 分布 df=1，即Cauchy）
x_vals = np.linspace(-1, 3, 300)
plt.plot(x_vals, t.pdf(x_vals, df=1), 'r--', label='Prior: t(df=1)')
plt.xlabel("y")
plt.title("Histogram of y with Prior and Sample Mean")
plt.legend()
plt.grid(True)
plt.show()

# 定义采样参数
n_iter = 1000
mu_init = 0
proposal_sd = 3  # 初始尝试，较大，可能接受率低

# 执行采样
post = metropolis_hastings(ybar=ybar, n=n, n_iter=n_iter,
                           mu_init=mu_init, proposal_sd=proposal_sd)

mu_samples = post["mu"]
accept_rate = post["accept_rate"]

print(f"Acceptance Rate: {accept_rate:.3f}")

# Trace plot
plt.plot(mu_samples, lw=0.7)
plt.title("Trace plot of mu")
plt.xlabel("Iteration")
plt.ylabel("mu")
plt.grid(True)
plt.show()

for sigma in [0.05, 0.9, 3]:
    post = metropolis_hastings(ybar=ybar, n=n, n_iter=1000,
                                mu_init=0, proposal_sd=sigma)
    print(f"Proposal SD = {sigma:.2f}, Acceptance Rate = {post['accept_rate']:.2f}")
