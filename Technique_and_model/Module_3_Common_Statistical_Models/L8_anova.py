import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
import statsmodels.formula.api as smf, statsmodels.api as sm
import pymc as pm, arviz as az
from io import StringIO

CSV = """weight,group
4.17,ctrl
5.58,ctrl
5.18,ctrl
6.11,ctrl
4.50,ctrl
4.61,ctrl
5.17,ctrl
4.53,ctrl
5.33,ctrl
5.14,ctrl
4.81,trt1
4.17,trt1
4.41,trt1
3.59,trt1
5.87,trt1
3.83,trt1
6.03,trt1
4.89,trt1
4.32,trt1
4.69,trt1
6.31,trt2
5.12,trt2
5.54,trt2
5.50,trt2
5.37,trt2
5.29,trt2
4.92,trt2
6.15,trt2
5.80,trt2
5.26,trt2
"""
plants = pd.read_csv(StringIO(CSV))
plants["group"] = plants["group"].astype("category")

def main():
    # 1. 描述性图
    sns.boxplot(x="group", y="weight", data=plants, hue="group", palette="Set2", legend=False)
    plt.show()

    # 2. 经典 ANOVA
    lm = smf.ols("weight ~ group", data=plants).fit()
    print(lm.summary())
    print(sm.stats.anova_lm(lm, typ=2))

    # 3. Bayesian cell-means
    g_idx = plants["group"].cat.codes.values        # 直接 NumPy，跨版本稳
    coords = {"group": plants["group"].cat.categories}

    with pm.Model(coords=coords) as model:
        mu    = pm.Normal("mu", 0, 10, dims="group")
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("y", mu=mu[g_idx], sigma=sigma, observed=plants["weight"].values)

        # Windows: cores=1 或放在 main 里
        idata = pm.sample(5000, tune=1000, chains=4, cores=1, target_accept=0.9)

    print(az.summary(idata, var_names=["mu", "sigma"], round_to=3))

    # 后验比较
    mu_ctrl = idata.posterior["mu"].sel(group="ctrl")
    mu_trt2 = idata.posterior["mu"].sel(group="trt2")
    print("P(mu_trt2 > mu_ctrl) =", float((mu_trt2 > mu_ctrl).mean()))
    print("P(mu_trt2 ≥ 1.1·mu_ctrl) =", float((mu_trt2 > 1.1*mu_ctrl).mean()))

if __name__ == "__main__":
    main()


