"""
Author : Rong Jin, University of Kentucky
Date   : 07-17-2025
"""
# ───────────── 0. Dependencies ────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm          # For loading data & linear regression
import statsmodels.formula.api as smf

# ───────────── 1. Load the Leinhardt data set ────────────────
# statsmodels can fetch the dataset directly from R’s carData package
dataset = sm.datasets.get_rdataset("Leinhardt", package="carData")
df = dataset.data.copy()

print("First 6 rows:")
print(df.head())
print("\nData types overview:")
print(df.dtypes)

# ───────────── 2. Initial visual exploration ─────────────────
# (a) Scatter-matrix for numeric variables (income & infant)
numeric_cols = ["income", "infant"]
g = sns.pairplot(df[numeric_cols], height=3)
g.fig.suptitle("Scatter-matrix for raw numeric variables", fontsize=16)

# Leave 8 % of the canvas free at the top for the title
g.fig.subplots_adjust(top=0.92)
plt.show()

# (b) Histograms for each variable (both are strongly right-skewed)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col in zip(axes, numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Histogram of {col}")
plt.tight_layout()
plt.show()

# ───────────── 3. Missing-value handling + variable transforms ─────────
df_clean = (
    df.dropna()            # Remove rows with missing values
      .copy()              # Explicit deep copy to avoid SettingWithCopyWarning
      .assign(             # Add / transform multiple columns at once
          logInfant = lambda d: np.log(d["infant"]),
          logIncome = lambda d: np.log(d["income"]),
          region    = lambda d: d["region"].astype("category"),   # Optional: convert to R-like factor
          oil       = lambda d: d["oil"].astype("category"),
      )
)

# Quick sanity check: scatter plot after log transform
sns.scatterplot(data=df_clean, x="logIncome", y="logInfant")
plt.title("log(infant) vs. log(income)")
plt.show()

# ───────────── 5. Linear regression (logInfant ~ logIncome) ───────────
# Using statsmodels OLS (equivalent to R’s lm())
model = smf.ols("logInfant ~ logIncome", data=df_clean).fit()
print(model.summary())

# ───────────── 6. Residuals & diagnostic plot (optional) ──────────────
# Residuals vs. fitted values
fig, ax = plt.subplots(figsize=(7, 5))          # Slightly larger canvas
sns.residplot(x=model.fittedvalues,
              y=model.resid,
              lowess=True,
              ax=ax)

ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Fitted")

fig.tight_layout()                              # One-liner to avoid clipping
# Alternatively: fig.subplots_adjust(bottom=0.15)

plt.show()
