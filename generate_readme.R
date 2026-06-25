# R Script to Generate README.md dynamically from CSV results
# Run this after simulations and real data analyses are complete.

cat("Generating README.md...\n")

# Helper to read CSV and format as Markdown table
csv_to_md <- function(file_path, digits = 2) {
  if (!file.exists(file_path)) {
    return(paste0("*Data in ", file_path, " not generated yet.*\n"))
  }
  df <- read.csv(file_path)
  
  # Format numeric columns
  for (col in colnames(df)) {
    if (is.numeric(df[[col]])) {
      df[[col]] <- round(df[[col]], digits)
    }
  }
  
  # Create headers
  headers <- colnames(df)
  header_line <- paste0("| ", paste(headers, collapse = " | "), " |")
  sep_line <- paste0("| ", paste(rep("---", length(headers)), collapse = " | "), " |")
  
  # Create rows
  rows <- apply(df, 1, function(row) {
    paste0("| ", paste(row, collapse = " | "), " |")
  })
  
  paste(c(header_line, sep_line, rows), collapse = "\n")
}

# Load tables
t1 <- csv_to_md("table1_example1.csv", digits = 1)
t2 <- csv_to_md("table2_example2.csv", digits = 1)
t3 <- csv_to_md("table3_example3.csv", digits = 1)
t4 <- csv_to_md("table4_prediction_small.csv", digits = 2)
t5 <- csv_to_md("table5_prediction_large.csv", digits = 2)
t6 <- csv_to_md("table6_bodyfat_selection.csv", digits = 2)
t7 <- csv_to_md("table7_bodyfat_pmp.csv", digits = 2)
t8 <- csv_to_md("table8_prostate_coefs.csv", digits = 3)
t9 <- csv_to_md("table9_prostate_pmp.csv", digits = 2)

readme_content <- paste0("
# Bayesian Adaptive Lasso (BaLasso) in R

An elegant, from-scratch R implementation of the **Bayesian Adaptive Lasso (BaLasso)** algorithm, reproducing the methodology, simulation studies, and real data analyses described in the paper:

> **Bayesian adaptive Lasso**  
> Chenlei Leng, Minh-Ngoc Tran & David Nott (2014)  
> *Annals of the Institute of Statistical Mathematics*, 66:221–244.

---

## Mathematical Formulation

The standard Lasso penalty applies the same amount of shrinkage $\\lambda$ to all coefficients, which can lead to biased estimates for large active coefficients. The **Bayesian Adaptive Lasso (BaLasso)** addresses this by adopting different shrinkage parameters $\\lambda_j$ for different coefficients $\\beta_j$.

### Hierarchical Model
For centered response $y$ and design matrix $X$:
$$y | X, \\beta, \\sigma^2 \\sim N_n(X\\beta, \\sigma^2 I_n)$$
$$\\beta | \\sigma^2, \\tau_1^2, \\dots, \\tau_p^2 \\sim N_p(0_p, \\sigma^2 D_\\tau), \\quad D_\\tau = \\text{diag}(\\tau_1^2, \\dots, \\tau_p^2)$$

With the adaptive priors:
$$\\sigma^2 \\sim \\pi(\\sigma^2) \\propto 1/\\sigma^2$$
$$\\tau_j^2 | \\lambda_j^2 \\sim \\text{Exponential}(\\lambda_j^2 / 2), \\quad j = 1, \\dots, p$$
$$\\lambda_j^2 \\sim \\text{Gamma}(r, \\delta), \\quad j = 1, \\dots, p$$

Integrating out $\\tau_j^2$, the conditional prior on $\\beta_j | \\sigma^2$ is Laplace with coefficient-specific penalty parameter $\\lambda_j$:
$$\\pi(\\beta_j | \\sigma^2) = \\frac{\\lambda_j}{2\\sqrt{\\sigma^2}} e^{-\\lambda_j |\\beta_j| / \\sqrt{\\sigma^2}}$$

### Gibbs Sampler Updates
The full conditionals used in our MCMC Gibbs sampler are:
1. **Beta**: $\\beta | y, \\sigma^2, \\tau^2 \\sim N(A^{-1} X^T y, \\sigma^2 A^{-1})$ where $A = X^T X + D_\\tau^{-1}$.
2. **Sigma-squared**: $\\sigma^2 | y, \\beta, \\tau^2 \\sim \\text{Inverse-Gamma}\\left(\\frac{n-1+p}{2}, \\frac{\\|y - X\\beta\\|_2^2 + \\beta^T D_\\tau^{-1} \\beta}{2}\\right)$.
3. **Tau-squared**: $1/\\tau_j^2 | \\beta_j, \\sigma, \\lambda_j \\sim \\text{Inverse-Gaussian}\\left(\\mu' = \\frac{\\lambda_j \\sigma}{|\\beta_j|}, \\lambda' = \\lambda_j^2\\right)$.
4. **Lambda-squared (Hierarchical Bayes)**: $\\lambda_j^2 | \\tau_j^2 \\sim \\text{Gamma}\\left(1 + r, \\frac{\\tau_j^2}{2} + \\delta\\right)$.
5. **Lambda (Empirical Bayes / Atchade's SA)**: $\\lambda_j = e^{s_j}$, where $s_j^{(t)} = s_j^{(t-1)} + a_t(2 - e^{2s_j^{(t-1)}} \\tau_j^2)$.

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
Rscript -e \"setwd('/Users/ramakrushnamishra/Documents/BADL'); source('test_balasso.R')\"

# 2. Run Real Data Analysis (produces Tables 6-9, Figures 2-3)
Rscript -e \"setwd('/Users/ramakrushnamishra/Documents/BADL'); source('real_data_analysis.R')\"

# 3. Run Simulation Studies (produces Tables 1-5)
Rscript -e \"setwd('/Users/ramakrushnamishra/Documents/BADL'); source('simulations.R')\"

# 4. Re-compile README.md with final results
Rscript -e \"setwd('/Users/ramakrushnamishra/Documents/BADL'); source('generate_readme.R')\"
```

---

## Empirical Results

### 1. Simulation Studies (Examples 1-3: Model Selection)

#### Table 1: Frequency of Correct Model Selection (Example 1: Simple)
", t1, "

#### Table 2: Frequency of Correct Model Selection (Example 2: Difficult)
", t2, "

#### Table 3: Frequency of Correct Model Selection (Example 3: Large p)
", t3, "

### 2. Prediction Performance (Example 4)

#### Table 4: Out-of-sample Prediction Squared Error (Small-p)
", t4, "

#### Table 5: Out-of-sample Prediction Squared Error (Large-p)
", t5, "

### 3. Real Data Analysis: Body Fat Dataset (Example 5)

Using $n=251$ observations (omitting 42nd outlier) to predict Brozek's body fat percentage using 13 clinical body measurements.

#### Table 6: Model Selection and BIC Comparison
", t6, "
*Note: BIC is calculated using the paper's formula $n \\log(\\text{RSS}/n) + \\frac{k}{2} \\log(n)$, where $k$ is the number of parameters including intercept.*

#### Table 7: Top 10 Models with Highest Posterior Model Probability (PMP)
", t7, "

### 4. Real Data Analysis: Prostate Cancer Dataset (Example 6)

Using $n=97$ observations to predict log PSA (`lpsa`) from 8 clinical measures.

#### Table 8: Coefficient Estimates and Posterior Lambdas
", t8, "

#### Table 9: Top 10 Models with Highest Posterior Model Probability (PMP)
", t9, "

---

## Figures

### Figure 2: Adaptive Shrinkage (lambda_2 vs beta_2)
Demonstrates that the penalty parameter $\\lambda_2$ decreases as the signal strength $\\beta_2$ increases (lighter penalty for stronger signals).

![Adaptive Shrinkage](figures/figure_adaptive_shrinkage.png)

### Figure 3: Credible Intervals (Prostate Cancer Example)
Shows the BaLasso-Mean posterior mean estimates and 95% equal-tailed credible intervals (solid lines) compared to the original Bayesian Lasso (dashed lines).

![Credible Intervals](figures/figure_prostate_credible_intervals.png)
")

writeLines(readme_content, "README.md")
cat("README.md generated successfully!\n")
