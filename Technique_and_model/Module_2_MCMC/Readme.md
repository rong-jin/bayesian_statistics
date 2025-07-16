# Bayesian MCMC Notes (Lessons 4 – 6)

> These notebooks are study notes for the **[Bayesian Statistics MCMC](https://www.coursera.org/learn/mcmc-bayesian-statistics/home/module/2)** on Coursera.  
> The original course examples are written in **R**; here they are fully **re-implemented in Python** for easier integration with the scientific-Python stack.

This repository contains three Jupyter notebooks that walk through fundamental Markov-chain Monte-Carlo (MCMC) algorithms in Python:

| Notebook | Topic | File |
|----------|-------|------|
| Lesson 4 | Random-Walk Metropolis–Hastings | `lesson4_metropolis_hastings_intro.ipynb` |
| Lesson 5 | Gibbs Sampling for Normal–Inverse-Gamma | `lesson5_gibbs_sampling_intro.ipynb` |
| Lesson 6 | Convergence Diagnostics (trace plots, ESS, R-hat …) | `lesson6_convergence_diagnostics_simplified_intro.ipynb` |

---

## 1 Prerequisites

|                       | Minimum version | Notes                                   |
|-----------------------|-----------------|-----------------------------------------|
| **Python**            | 3.9 +           | Tested on 3.10 & 3.11                  |
| **Jupyter**           | any             | e.g. `jupyterlab` or `notebook`         |
| `numpy`               | 1.22 +          | core array maths                        |
| `matplotlib`          | 3.5 +           | plotting                                |
| `arviz`               | 0.14 +          | MCMC diagnostics (ESS, R-hat, trace)    |
| `scipy`               | 1.8 +           | Student-t / Inverse-Gamma distributions |
| `statsmodels`         | 0.14 +          | autocorrelation (`acf`)                 |

> **Tip:** the notebooks will also run happily inside **VS Code** using its Jupyter extension.

---

## 2 Quick-start (recommended)

### 2.1 Conda environment

```bash
# clone / download project first
cd bayesian-mcmc-tutorials

# create & activate environment
conda env create -f environment.yml
conda activate mcmc-lessons

# launch JupyterLab
jupyter lab
````

<details>
<summary><code>environment.yml</code></summary>

```yaml
name: mcmc-lessons
channels:
  - conda-forge
dependencies:
  - python=3.10
  - notebook
  - numpy>=1.22
  - matplotlib>=3.5
  - scipy>=1.8
  - arviz>=0.14
  - statsmodels>=0.14
```

</details>

---

### 2.2 Virtualenv / pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

jupyter lab
```

<details>
<summary><code>requirements.txt</code></summary>

```
numpy>=1.22
matplotlib>=3.5
scipy>=1.8
arviz>=0.14
statsmodels>=0.14
notebook
```

</details>

---

## 3 Running the notebooks

1. Launch JupyterLab or VS Code.
2. Open any of the `*_intro.ipynb` files.
3. Step through the cells **in order**; random seeds are set for reproducibility.

Feel free to tweak hyper-parameters (`cand_sd`, prior settings, iteration counts) to see how mixing and diagnostics change!

---




