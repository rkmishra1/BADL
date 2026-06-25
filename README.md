
# Bayesian Adaptive Lasso (BaLasso) in R

An elegant, from-scratch R implementation of the **Bayesian Adaptive Lasso (BaLasso)** algorithm, reproducing the methodology, simulation studies, and real data analyses described in the paper:

> **Bayesian adaptive Lasso**  
> Chenlei Leng, Minh-Ngoc Tran & David Nott (2014)  
> *Annals of the Institute of Statistical Mathematics*, 66:221–244.

---

## Mathematical Formulation

The standard Lasso penalty applies the same amount of shrinkage $\lambda$ to all coefficients, which can lead to biased estimates for large active coefficients. The **Bayesian Adaptive Lasso (BaLasso)** addresses this by adopting different shrinkage parameters $\lambda_j$ for different coefficients $\beta_j$.

### Hierarchical Model
For centered response $y$ and design matrix $X$:
$$y | X, \beta, \sigma^2 \sim N_n(X\beta, \sigma^2 I_n)$$
$$\beta | \sigma^2, \tau_1^2, \dots, \tau_p^2 \sim N_p(0_p, \sigma^2 D_\tau), \quad D_\tau = \text{diag}(\tau_1^2, \dots, \tau_p^2)$$

With the adaptive priors:
$$\sigma^2 \sim \pi(\sigma^2) \propto 1/\sigma^2$$
$$\tau_j^2 | \lambda_j^2 \sim \text{Exponential}(\lambda_j^2 / 2), \quad j = 1, \dots, p$$
$$\lambda_j^2 \sim \text{Gamma}(r, \delta), \quad j = 1, \dots, p$$

Integrating out $\tau_j^2$, the conditional prior on $\beta_j | \sigma^2$ is Laplace with coefficient-specific penalty parameter $\lambda_j$:
$$\pi(\beta_j | \sigma^2) = \frac{\lambda_j}{2\sqrt{\sigma^2}} e^{-\lambda_j |\beta_j| / \sqrt{\sigma^2}}$$

### Gibbs Sampler Updates
The full conditionals used in our MCMC Gibbs sampler are:
1. **Beta**: $\beta | y, \sigma^2, \tau^2 \sim N(A^{-1} X^T y, \sigma^2 A^{-1})$ where $A = X^T X + D_\tau^{-1}$.
2. **Sigma-squared**: $\sigma^2 | y, \beta, \tau^2 \sim \text{Inverse-Gamma}\left(\frac{n-1+p}{2}, \frac{\|y - X\beta\|_2^2 + \beta^T D_\tau^{-1} \beta}{2}\right)$.
3. **Tau-squared**: $1/\tau_j^2 | \beta_j, \sigma, \lambda_j \sim \text{Inverse-Gaussian}\left(\mu' = \frac{\lambda_j \sigma}{|\beta_j|}, \lambda' = \lambda_j^2\right)$.
4. **Lambda-squared (Hierarchical Bayes)**: $\lambda_j^2 | \tau_j^2 \sim \text{Gamma}\left(1 + r, \frac{\tau_j^2}{2} + \delta\right)$.
5. **Lambda (Empirical Bayes / Atchade's SA)**: $\lambda_j = e^{s_j}$, where $s_j^{(t)} = s_j^{(t-1)} + a_t(2 - e^{2s_j^{(t-1)}} \tau_j^2)$.

---

## File Structure

- `balasso.R`: Core implementation including MCMC Gibbs sampler, custom coordinate descent weighted Lasso solver, and wrappers for competing models (Lasso, aLasso, BLasso).
- `simulations.R`: Script to run Examples 1–4 from the paper, generating CSV results tables.
- `real_data_analysis.R`: Script to run the Body Fat (Example 5) and Prostate Cancer (Example 6) analyses, generating figures and tables.
- `test_balasso.R`: Unit tests verifying the samplers, solvers, and prediction functions.
- `generate_readme.R`: Script to compile this `README.md` dynamically from results.
- `figures/`: Folder containing generated diagnostic and credible interval plots.

---

## Usage Instructions

To run the entire suite and reproduce all results:

```R
# 1. Run Unit Tests to verify installation
Rscript -e "setwd('/Users/ramakrushnamishra/Documents/BADL'); source('test_balasso.R')"

# 2. Run Real Data Analysis (produces Tables 6-9, Figures 2-3)
Rscript -e "setwd('/Users/ramakrushnamishra/Documents/BADL'); source('real_data_analysis.R')"

# 3. Run Simulation Studies (produces Tables 1-5)
Rscript -e "setwd('/Users/ramakrushnamishra/Documents/BADL'); source('simulations.R')"

# 4. Re-compile README.md with final results
Rscript -e "setwd('/Users/ramakrushnamishra/Documents/BADL'); source('generate_readme.R')"
```

---

## Empirical Results

### 1. Simulation Studies (Examples 1-3: Model Selection)

#### Table 1: Frequency of Correct Model Selection (Example 1: Simple)
| n | sigma | Lasso | aLasso | BaLasso_Freq | BaLasso_Median | BaLasso_Mean | BaLasso_EB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 1 | 14 | 49 | 97 | 91 | 97 | 34 |
| 30 | 3 | 4 | 17 | 20 | 39 | 20 | 5 |
| 60 | 1 | 11 | 67 | 100 | 87 | 100 | 23 |
| 60 | 3 | 7 | 32 | 65 | 50 | 65 | 4 |
| 120 | 1 | 15 | 84 | 100 | 81 | 100 | 22 |
| 120 | 3 | 9 | 51 | 93 | 52 | 93 | 10 |

#### Table 2: Frequency of Correct Model Selection (Example 2: Difficult)
| n | sigma | Lasso | aLasso | BaLasso_Freq | BaLasso_Median | BaLasso_Mean | BaLasso_EB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | 9 | 11 | 42 | 1 | 3 | 1 | 18 |
| 120 | 5 | 9 | 65 | 67 | 59 | 65 | 46 |
| 300 | 3 | 14 | 88 | 99 | 85 | 99 | 57 |
| 300 | 1 | 23 | 100 | 100 | 97 | 100 | 81 |

#### Table 3: Frequency of Correct Model Selection (Example 3: Large p)
| n | sigma | aLasso | BaLasso_Freq | BaLasso_Median | BaLasso_Mean | BaLasso_EB |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 1 | 98 | 71 | 1 | 71 | 0 |
| 50 | 3 | 5 | 38 | 0 | 38 | 0 |
| 50 | 5 | 0 | 5 | 0 | 5 | 0 |
| 100 | 1 | 100 | 78 | 2 | 78 | 0 |
| 100 | 3 | 36 | 13 | 0 | 13 | 0 |
| 100 | 5 | 4 | 2 | 0 | 1 | 0 |
| 200 | 1 | 100 | 73 | 1 | 73 | 0 |
| 200 | 3 | 35 | 5 | 0 | 5 | 0 |
| 200 | 5 | 4 | 0 | 0 | 0 | 0 |

### 2. Prediction Performance (Example 4)

#### Table 4: Out-of-sample Prediction Squared Error (Small-p)
| nT_nP | sigma | Lasso | aLasso | BLasso | BaLasso_Mean | BaLasso_BMA |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 1 | 1.35 | 1.31 | 1.38 | 1.24 | 1.21 |
| 30 | 3 | 11.87 | 12.15 | 12.95 | 14.35 | 11.89 |
| 30 | 5 | 34.07 | 35.19 | 35.61 | 37.34 | 32.76 |
| 30 | 10 | 125.08 | 126.27 | 124.3 | 123.03 | 120.68 |
| 100 | 1 | 1.08 | 1.07 | 1.09 | 1.06 | 1.06 |
| 100 | 3 | 9.82 | 9.74 | 9.93 | 9.61 | 9.6 |
| 100 | 5 | 26.84 | 27.01 | 27.36 | 27.2 | 26.8 |
| 100 | 10 | 111.42 | 112.29 | 113.42 | 111.08 | 110.82 |
| 200 | 1 | 1.03 | 1.04 | 1.04 | 1.03 | 1.03 |
| 200 | 3 | 9.22 | 9.15 | 9.28 | 9.09 | 9.11 |
| 200 | 5 | 26.16 | 26.1 | 26.35 | 25.94 | 25.99 |
| 200 | 10 | 105.37 | 105.24 | 106.18 | 105.13 | 105.16 |

#### Table 5: Out-of-sample Prediction Squared Error (Large-p)
| nT_nP | sigma | Lasso | aLasso | BLasso | BaLasso_Mean | BaLasso_BMA |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 1 | 1.53 | 2.29 | 3.23 | 1.41 | 1.35 |
| 100 | 3 | 12.64 | 10.93 | 27.6 | 11.06 | 12.63 |
| 100 | 5 | 33.62 | 30.68 | 81.77 | 30.04 | 35.41 |
| 100 | 10 | 133.52 | 135.19 | 331.29 | 130.89 | 156.1 |
| 200 | 1 | 1.21 | 1.16 | 1.67 | 1.06 | 1.12 |
| 200 | 3 | 10.93 | 11.57 | 16.04 | 10.47 | 11.08 |
| 200 | 5 | 29.52 | 31.31 | 45.76 | 28.37 | 30.83 |
| 200 | 10 | 114.2 | 121.14 | 182.25 | 117.04 | 126.6 |

### 3. Real Data Analysis: Body Fat Dataset (Example 5)

Using $n=251$ observations (omitting 42nd outlier) to predict Brozek's body fat percentage using 13 clinical body measurements.

#### Table 6: Model Selection and BIC Comparison
| Method | Selected_Variables | BIC |
| --- | --- | --- |
| Lasso | Age, Height, Neck, Abdomen, Forearm, Wrist | 712.06 |
| aLasso | Age, Weight, Height, Neck, Abdomen, Hip, Thigh, Ankle, Biceps, Forearm, Wrist | 714.22 |
| BaLasso | Age, Weight, Neck, Abdomen, Thigh, Biceps, Forearm, Wrist | 708.92 |
*Note: BIC is calculated using the paper's formula $n \log(\text{RSS}/n) + \frac{k}{2} \log(n)$, where $k$ is the number of parameters including intercept.*

#### Table 7: Top 10 Models with Highest Posterior Model Probability (PMP)
| Model_Indices | PMP_Percent |
| --- | --- |
| 1 2 4 6 7 8 10 11 12 13 | 6.35 |
| 1 2 4 6 8 10 11 12 13 | 4.70 |
| 1 2 3 4 6 7 8 10 11 12 13 | 2.85 |
| 1 2 4 6 7 8 11 12 13 | 2.65 |
| 1 2 4 6 8 11 12 13 | 2.25 |
| 1 2 4 6 7 8 10 12 13 | 2.05 |
| 1 2 3 4 6 7 8 10 12 13 | 2.00 |
| 1 2 3 4 6 7 8 11 12 13 | 1.75 |
| 1 2 4 6 7 8 9 10 11 12 13 | 1.55 |
| 1 2 4 6 10 11 12 13 | 1.45 |

### 4. Real Data Analysis: Prostate Cancer Dataset (Example 6)

Using $n=97$ observations to predict log PSA (`lpsa`) from 8 clinical measures.

#### Table 8: Coefficient Estimates and Posterior Lambdas
| Covariate | BaLasso_EB_lambda | BaLasso_Median_lambda | BaLasso_Mean_lambda | BaLasso_EB_beta | BaLasso_Median_beta | BaLasso_Mean_beta | Lasso_beta | aLasso_beta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lcavol |  1.118 |   0.966 |   1.347 | 0.533 | 0.536 | 0.687 | 0.495 | 0.513 |
| lweight |  2.878 |   3.216 |  30.817 | 0.566 | 0.623 | 0.263 | 0.493 | 0.625 |
| age | 15.344 |  85.800 | 216.652 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lbph |  9.505 |  61.624 | 204.514 | 0.039 | 0.000 | 0.000 | 0.037 | 0.000 |
| svi |  2.760 |   3.507 |  39.855 | 0.648 | 0.613 | 0.000 | 0.556 | 0.650 |
| lcp | 47.512 | 103.529 | 235.948 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gleason | 19.026 | 114.728 | 245.533 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pgg45 | 22.447 |  98.365 | 234.928 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 |

#### Table 9: Top 10 Models with Highest Posterior Model Probability (PMP)
| Model_Indices | PMP_Percent |
| --- | --- |
| 1 2 5 | 31.60 |
| 1 2 4 5 | 12.70 |
| 1 2 5 8 |  8.60 |
| 1 2 3 5 |  6.35 |
| 1 2 3 4 5 |  4.90 |
| 1 2 5 7 |  4.70 |
| 1 4 5 |  3.75 |
| 1 2 4 5 8 |  2.70 |
| 1 2 |  2.65 |
| 1 2 8 |  2.15 |

---

## Figures

### Figure 2: Adaptive Shrinkage (lambda_2 vs beta_2)
Demonstrates that the penalty parameter $\lambda_2$ decreases as the signal strength $\beta_2$ increases (lighter penalty for stronger signals).

![Adaptive Shrinkage](figures/figure_adaptive_shrinkage.png)

### Figure 3: Credible Intervals (Prostate Cancer Example)
Shows the BaLasso-Mean posterior mean estimates and 95% equal-tailed credible intervals (solid lines) compared to the original Bayesian Lasso (dashed lines).

![Credible Intervals](figures/figure_prostate_credible_intervals.png)

