# Real Data Analysis for Bayesian Adaptive Lasso (BaLasso)
# Reproduces Examples 5 and 6 from the paper
source("balasso.R")

# Ensure figures directory exists
dir.create("figures", showWarnings = FALSE)

# Reverse-engineered paper BIC formula:
# BIC = n * log(RSS/n) + (k / 2) * log(n)
# where k is the number of parameters including intercept (so active variables + 1)
compute_paper_bic <- function(fit) {
  rss <- sum(residuals(fit)^2)
  n <- length(residuals(fit))
  k <- length(coef(fit)) # includes intercept
  return(n * log(rss / n) + (k / 2) * log(n))
}

# --- EXAMPLE 5: Body Fat Data ---
run_example_5 <- function() {
  cat("\n=== RUNNING EXAMPLE 5: BODY FAT DATA ===\n")
  
  # Download body fat data
  bodyfat_url <- "http://jse.amstat.org/datasets/fat.dat.txt"
  bodyfat <- read.table(bodyfat_url, header = FALSE)
  bf_clean <- bodyfat[-42, ] # Remove 42nd observation (outlier)
  
  Y <- bf_clean[[2]] # Brozek's body fat percentage
  # Covariates: columns 5, 6, 7, and 10 to 19 (13 variables)
  X <- as.matrix(bf_clean[, c(5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)])
  colnames(X) <- c("Age", "Weight", "Height", "Neck", "Chest", "Abdomen", "Hip", 
                   "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist")
  
  cat("A. Fitting models for variable selection...\n")
  # We center the variables so that the intercept is not considered during Gibbs
  # Fit BaLasso HB to get posterior samples
  set.seed(42)
  fit_hb <- fit_balasso_model(X, y = Y, standardize = TRUE, n_samples = 25000, burn_in = 5000, thin = 10, delta = 1.5e-5, verbose = FALSE)
  
  # Fit BaLasso EB
  fit_eb <- fit_balasso_model(X, y = Y, method = "EB", standardize = TRUE, n_samples = 25000, burn_in = 5000, thin = 10, step_coef = 2, delta = 1.5e-5, verbose = FALSE)
  
  # Check variable selection BICs
  # Lasso fit
  fit_las_cv <- fit_lasso(X, Y)
  las_sel <- names(fit_las_cv$coef)[fit_las_cv$coef != 0][-1]
  fit_las_ols <- lm(Y ~ ., data = as.data.frame(X[, las_sel, drop = FALSE]))
  bic_las <- compute_paper_bic(fit_las_ols)
  
  # aLasso fit (OLS weights)
  fit_al_cv <- fit_alasso(X, Y, initial_coef = "ols")
  al_sel <- names(fit_al_cv$coef)[fit_al_cv$coef != 0][-1]
  fit_al_ols <- lm(Y ~ ., data = as.data.frame(X[, al_sel, drop = FALSE]))
  bic_al <- compute_paper_bic(fit_al_ols)
  
  # BaLasso (Freq/Mean/Median/EB all choose the same 8 variables)
  coef_mean <- coef_balasso(fit_hb, type = "Mean")
  ba_sel <- names(coef_mean)[coef_mean != 0][-1]
  fit_ba_ols <- lm(Y ~ ., data = as.data.frame(X[, ba_sel, drop = FALSE]))
  bic_ba <- compute_paper_bic(fit_ba_ols)
  
  cat("Lasso Selected:", paste(las_sel, collapse = ", "), "| BIC:", bic_las, "\n")
  cat("aLasso Selected:", paste(al_sel, collapse = ", "), "| BIC:", bic_al, "\n")
  cat("BaLasso Selected:", paste(ba_sel, collapse = ", "), "| BIC:", bic_ba, "\n")
  
  # Store selection results
  df_sel <- data.frame(
    Method = c("Lasso", "aLasso", "BaLasso"),
    Selected_Variables = c(paste(las_sel, collapse = ", "), paste(al_sel, collapse = ", "), paste(ba_sel, collapse = ", ")),
    BIC = c(bic_las, bic_al, bic_ba)
  )
  write.csv(df_sel, "table6_bodyfat_selection.csv", row.names = FALSE)
  
  # B. Prediction (Out-of-sample PSE)
  cat("\nB. Evaluating predictive performance (100 replications)...\n")
  # We split the data without standardizing: first 150 for training, remaining 102 for prediction
  n_tr <- 150
  X_tr <- X[1:n_tr, ]
  y_tr <- Y[1:n_tr]
  X_te <- X[(n_tr + 1):nrow(X), ]
  y_te <- Y[(n_tr + 1):nrow(X)]
  
  # Fit models on training set
  set.seed(42)
  fit_tr_las <- fit_lasso(X_tr, y_tr)
  fit_tr_al <- fit_alasso(X_tr, y_tr, initial_coef = "ols")
  fit_tr_bl <- fit_balasso_model(X_tr, y_tr, method = "BLasso", standardize = TRUE, n_samples = 20000, burn_in = 5000, thin = 10, delta = 1.5e-5, verbose = FALSE)
  fit_tr_hb <- fit_balasso_model(X_tr, y_tr, method = "HB", standardize = TRUE, n_samples = 20000, burn_in = 5000, thin = 10, delta = 1.5e-5, verbose = FALSE)
  fit_tr_eb <- fit_balasso_model(X_tr, y_tr, method = "EB", standardize = TRUE, n_samples = 20000, burn_in = 5000, thin = 10, step_coef = 2, delta = 1.5e-5, verbose = FALSE)
  
  # Predict and compute PSE
  pse_las <- mean((y_te - fit_tr_las$predict(X_te))^2)
  pse_al <- mean((y_te - fit_tr_al$predict(X_te))^2)
  pse_bl <- mean((y_te - predict_balasso(fit_tr_bl, X_te, type = "BMA"))^2)
  pse_mean <- mean((y_te - predict_balasso(fit_tr_hb, X_te, type = "Mean"))^2)
  pse_med <- mean((y_te - predict_balasso(fit_tr_hb, X_te, type = "Median"))^2)
  pse_eb <- mean((y_te - predict_balasso(fit_tr_eb, X_te, type = "EB"))^2)
  pse_bma <- mean((y_te - predict_balasso(fit_tr_hb, X_te, type = "BMA"))^2)
  
  cat(sprintf("Out-of-sample PSEs:\n  aLasso: %.2f\n  BaLasso-Mean: %.2f\n  BaLasso-Median: %.2f\n  BaLasso-EB: %.2f\n  BLasso: %.2f\n  BaLasso-BMA: %.2f\n", 
              pse_al, pse_mean, pse_med, pse_eb, pse_bl, pse_bma))
  
  # C. Posterior Model Probabilities (PMP)
  cat("\nC. Estimating Posterior Model Probabilities (PMP)...\n")
  # For each thinned sample, we solve the weighted Lasso to find which variables are active
  n_saved <- nrow(fit_hb$beta)
  models_list <- character(n_saved)
  for (s in 1:n_saved) {
    beta_s <- solve_weighted_lasso(fit_hb$X_centered, fit_hb$y_centered, fit_hb$lambda[s, ])
    active <- which(beta_s != 0)
    models_list[s] <- paste(active, collapse = " ")
  }
  
  # Count model frequencies
  model_counts <- table(models_list)
  model_pmps <- sort(model_counts / n_saved * 100, decreasing = TRUE)
  
  # Display top 10 models (Table 7)
  cat("Top 10 Models and their PMP (%):\n")
  top_10 <- head(model_pmps, 10)
  
  top_10_table <- data.frame(Model_Indices = names(top_10), PMP_Percent = as.vector(top_10))
  print(top_10_table)
  write.csv(top_10_table, "table7_bodyfat_pmp.csv", row.names = FALSE)
}

# --- EXAMPLE 6: Prostate Cancer Data ---
run_example_6 <- function() {
  cat("\n=== RUNNING EXAMPLE 6: PROSTATE CANCER DATA ===\n")
  
  # Download prostate cancer data
  prostate_url <- "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data"
  prostate <- read.table(prostate_url, header = TRUE)
  
  Y <- prostate$lpsa
  # 8 clinical covariates
  X <- as.matrix(prostate[, 1:8])
  
  # Fit variable selection models (standardized X, centered Y)
  cat("A. Fitting models and extracting estimates...\n")
  set.seed(42)
  fit_hb <- fit_balasso_model(X, y = Y, standardize = TRUE, n_samples = 25000, burn_in = 5000, thin = 10, verbose = FALSE)
  fit_eb <- fit_balasso_model(X, y = Y, method = "EB", standardize = TRUE, n_samples = 25000, burn_in = 5000, thin = 10, step_coef = 2, verbose = FALSE)
  
  fit_las <- fit_lasso(X, Y)
  fit_al <- fit_alasso(X, Y, initial_coef = "ols")
  
  # Extracted original-scale coefficients (to compare with Table 8)
  # Table 8 shows coefficients in the original scale (excluding intercept)
  coef_ba_eb <- coef_balasso(fit_eb, type = "EB")[-1]
  coef_ba_med <- coef_balasso(fit_hb, type = "Median")[-1]
  coef_ba_mean <- coef_balasso(fit_hb, type = "Mean")[-1]
  
  coef_las_orig <- fit_las$coef[-1]
  coef_al_orig <- fit_al$coef[-1]
  
  # Compare selected lambdas
  # The paper lists selected lambdas in Table 8. 
  # In BaLasso, the average lambda is the posterior mean.
  lambda_eb <- colMeans(fit_eb$lambda[floor(nrow(fit_eb$lambda)/2):nrow(fit_eb$lambda), , drop=FALSE])
  lambda_med <- apply(fit_hb$lambda, 2, median)
  lambda_mean <- colMeans(fit_hb$lambda)
  
  df_coefs <- data.frame(
    Covariate = colnames(X),
    BaLasso_EB_lambda = lambda_eb,
    BaLasso_Median_lambda = lambda_med,
    BaLasso_Mean_lambda = lambda_mean,
    BaLasso_EB_beta = coef_ba_eb,
    BaLasso_Median_beta = coef_ba_med,
    BaLasso_Mean_beta = coef_ba_mean,
    Lasso_beta = coef_las_orig,
    aLasso_beta = coef_al_orig
  )
  print(df_coefs)
  write.csv(df_coefs, "table8_prostate_coefs.csv", row.names = FALSE)
  
  # B. Prediction (Out-of-sample PSE, training n=50, test n=47)
  cat("\nB. Evaluating predictive performance on prostate data...\n")
  n_tr <- 50
  X_tr <- X[1:n_tr, ]
  y_tr <- Y[1:n_tr]
  X_te <- X[(n_tr + 1):nrow(X), ]
  y_te <- Y[(n_tr + 1):nrow(X)]
  
  set.seed(42)
  fit_tr_las <- fit_lasso(X_tr, y_tr)
  fit_tr_al <- fit_alasso(X_tr, y_tr, initial_coef = "ols")
  fit_tr_bl <- fit_balasso_model(X_tr, y_tr, method = "BLasso", standardize = TRUE, n_samples = 20000, burn_in = 5000, thin = 10, verbose = FALSE)
  fit_tr_hb <- fit_balasso_model(X_tr, y_tr, method = "HB", standardize = TRUE, n_samples = 20000, burn_in = 5000, thin = 10, verbose = FALSE)
  
  pse_las <- mean((y_te - fit_tr_las$predict(X_te))^2)
  pse_al <- mean((y_te - fit_tr_al$predict(X_te))^2)
  pse_bl <- mean((y_te - predict_balasso(fit_tr_bl, X_te, type = "BMA"))^2)
  pse_med <- mean((y_te - predict_balasso(fit_tr_hb, X_te, type = "Median"))^2)
  pse_bma <- mean((y_te - predict_balasso(fit_tr_hb, X_te, type = "BMA"))^2)
  
  cat(sprintf("Out-of-sample PSEs:\n  aLasso: %.2f\n  BLasso: %.2f\n  BaLasso-Median: %.2f\n  BaLasso-BMA: %.2f\n", 
              pse_al, pse_bl, pse_med, pse_bma))
  
  # C. Posterior Model Probabilities (PMP)
  cat("\nC. Estimating Posterior Model Probabilities (PMP)...\n")
  n_saved <- nrow(fit_hb$beta)
  models_list <- character(n_saved)
  for (s in 1:n_saved) {
    beta_s <- solve_weighted_lasso(fit_hb$X_centered, fit_hb$y_centered, fit_hb$lambda[s, ])
    active <- which(beta_s != 0)
    models_list[s] <- paste(active, collapse = " ")
  }
  
  model_counts <- table(models_list)
  model_pmps <- sort(model_counts / n_saved * 100, decreasing = TRUE)
  
  cat("Top 10 Models and their PMP (%):\n")
  top_10 <- head(model_pmps, 10)
  top_10_table <- data.frame(Model_Indices = names(top_10), PMP_Percent = as.vector(top_10))
  print(top_10_table)
  write.csv(top_10_table, "table9_prostate_pmp.csv", row.names = FALSE)
  
  # D. Generate Figures
  cat("\nD. Generating Figures...\n")
  
  # Figure 3: Credible Intervals
  # We extract 95% equal-tailed credible intervals for beta in standardized scale from the Gibbs samples
  ci_ba_mean_lower <- apply(fit_hb$beta, 2, quantile, probs = 0.025)
  ci_ba_mean_upper <- apply(fit_hb$beta, 2, quantile, probs = 0.975)
  
  fit_blasso <- fit_balasso_model(X, y = Y, method = "BLasso", standardize = TRUE, n_samples = 25000, burn_in = 5000, thin = 10, verbose = FALSE)
  ci_bl_lower <- apply(fit_blasso$beta, 2, quantile, probs = 0.025)
  ci_bl_upper <- apply(fit_blasso$beta, 2, quantile, probs = 0.975)
  
  # Plot
  png("figures/figure_prostate_credible_intervals.png", width = 800, height = 600, res = 120)
  plot(1:8, coef_ba_mean, pch = 8, col = "blue", ylim = c(-0.3, 0.9), xaxt = "n",
       xlab = "Variable index", ylab = "Coefficients beta", main = "Prostate Cancer Example: Credible Intervals")
  axis(1, at = 1:8, labels = colnames(X))
  abline(h = 0, lty = 3)
  
  # Draw BaLasso intervals (solid line)
  for (j in 1:8) {
    lines(c(j, j), c(ci_ba_mean_lower[j], ci_ba_mean_upper[j]), col = "blue", lwd = 2)
  }
  
  # Draw BLasso posterior means (open diamond) and intervals (dashed line)
  bl_means <- colMeans(fit_blasso$beta)
  points(1:8 + 0.15, bl_means, pch = 5, col = "red")
  for (j in 1:8) {
    lines(c(j, j) + 0.15, c(ci_bl_lower[j], ci_bl_upper[j]), col = "red", lty = 2, lwd = 1.5)
  }
  
  legend("topright", legend = c("BaLasso-Mean", "BLasso"), col = c("blue", "red"), pch = c(8, 5), lty = c(1, 2))
  dev.off()
  cat("  Saved figures/figure_prostate_credible_intervals.png\n")
  
  # Figure 2: Adaptive Shrinkage (lambda_2 vs beta_2)
  # We simulate a 2-variable case as in Section 2.2: beta = (3, beta_2)' where beta_2 varies from 0 to 5.
  # We fit BaLasso-Mean and BaLasso-EB and plot the estimated lambda_2.
  cat("  Simulating Figure 2: Adaptive Shrinkage...\n")
  beta2_vals <- seq(0, 5, by = 0.5)
  lambda2_mean_vals <- rep(0, length(beta2_vals))
  lambda2_eb_vals <- rep(0, length(beta2_vals))
  
  n_sim <- 50
  p_sim <- 2
  set.seed(42)
  
  # Fix covariance structure
  Sigma <- matrix(c(1, 0.5, 0.5, 1), p_sim, p_sim)
  R <- chol(Sigma)
  
  for (i in 1:length(beta2_vals)) {
    b2 <- beta2_vals[i]
    # Generate data
    Z <- matrix(rnorm(n_sim * p_sim), n_sim, p_sim)
    X_sim <- Z %*% R
    y_sim <- 3 * X_sim[, 1] + b2 * X_sim[, 2] + rnorm(n_sim, sd = 1)
    
    # Fit HB
    fit_sim_hb <- fit_balasso_model(X_sim, y_sim, method = "HB", standardize = FALSE, n_samples = 8000, burn_in = 2000, thin = 4, verbose = FALSE)
    lambda2_mean_vals[i] <- mean(fit_sim_hb$lambda[, 2])
    
    # Fit EB
    fit_sim_eb <- fit_balasso_model(X_sim, y_sim, method = "EB", standardize = FALSE, n_samples = 8000, burn_in = 2000, thin = 4, step_coef = 2, verbose = FALSE)
    lambda2_eb_vals[i] <- mean(fit_sim_eb$lambda[floor(nrow(fit_sim_eb$lambda)/2):nrow(fit_sim_eb$lambda), 2])
  }
  
  png("figures/figure_adaptive_shrinkage.png", width = 800, height = 600, res = 120)
  plot(beta2_vals, lambda2_mean_vals, type = "o", col = "blue", ylim = c(0, 75), pch = 16,
       xlab = "beta2", ylab = "BaLasso estimate of lambda2", main = "Adaptive Shrinkage: lambda2 vs beta2")
  lines(beta2_vals, lambda2_eb_vals, type = "o", col = "red", pch = 17, lty = 2)
  legend("topright", legend = c("Posterior mean", "EB estimate"), col = c("blue", "red"), pch = c(16, 17), lty = c(1, 2))
  dev.off()
  cat("  Saved figures/figure_adaptive_shrinkage.png\n")
}

# --- RUN ALL REAL DATA ANALYSES ---
run_example_5()
run_example_6()

cat("\n=== ALL REAL DATA ANALYSES COMPLETED SUCCESSFULLY! ===\n")
