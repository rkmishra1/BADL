# Unit tests for BaLasso implementation
source("balasso.R")

cat("=== RUNNING UNIT TESTS ===\n")

# 1. Test rinvgauss_vec
cat("Testing rinvgauss_vec...\n")
mu <- c(1.5, 10, Inf, 1000)
lambda <- c(2.0, 5.0, 1.0, 2.0)
samples <- rinvgauss_vec(mu, lambda)
stopifnot(length(samples) == 4)
stopifnot(all(samples > 0))
cat("  rinvgauss_vec runs successfully and returns positive values.\n")

# 2. Test solve_weighted_lasso vs glmnet
cat("Testing solve_weighted_lasso vs glmnet...\n")
set.seed(42)
n <- 50
p <- 8
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(3, 1.5, 0, 0, 2, 0, 0, 0)
y <- X %*% beta_true + rnorm(n)

# Center X and y
y_c <- y - mean(y)
X_c <- scale(X, center = TRUE, scale = FALSE)

lambda_vec <- c(10, 5, 20, 20, 5, 20, 20, 20)

# Run our solver
beta_our <- solve_weighted_lasso(X_c, y_c, lambda_vec)

# Run glmnet solver
library(glmnet)
# glmnet minimizes (1/(2n)) ||y - X beta||_2^2 + lambda_glmnet * sum(w_j |beta_j|)
# Set penalty.factor = lambda_vec and lambda = 1/(2n)
fit_net <- glmnet(X_c, y_c, alpha = 1, penalty.factor = lambda_vec, 
                  lambda = (1 / (2 * n)) * mean(lambda_vec), intercept = FALSE, standardize = FALSE)
beta_net <- as.vector(coef(fit_net))[-1]

# Compare
diff_max <- max(abs(beta_our - beta_net))
cat(sprintf("  Max difference between our solver and glmnet: %e\n", diff_max))
if (diff_max > 1e-4) {
  stop("Solver discrepancy too large!")
}
cat("  Weighted Lasso solvers are consistent!\n")

# 3. Test MCMC pipeline
cat("Testing balasso_mcmc...\n")
fit_hb <- fit_balasso_model(X, y, method = "HB", n_samples = 1000, burn_in = 200, thin = 2, verbose = FALSE)
stopifnot(nrow(fit_hb$beta) == 400)
stopifnot(ncol(fit_hb$beta) == p)
stopifnot(length(fit_hb$sigma2) == 400)
cat("  HB MCMC runs successfully.\n")

fit_eb <- fit_balasso_model(X, y, method = "EB", n_samples = 1000, burn_in = 200, thin = 2, verbose = FALSE)
stopifnot(nrow(fit_eb$beta) == 400)
cat("  EB MCMC runs successfully.\n")

fit_bl <- fit_balasso_model(X, y, method = "BLasso", n_samples = 1000, burn_in = 200, thin = 2, verbose = FALSE)
stopifnot(nrow(fit_bl$beta) == 400)
cat("  BLasso MCMC runs successfully.\n")

# 4. Test predictions
cat("Testing predict_balasso and coef_balasso...\n")
coef_mean <- coef_balasso(fit_hb, type = "Mean")
stopifnot(length(coef_mean) == p + 1)

pred_mean <- predict_balasso(fit_hb, X, type = "Mean")
stopifnot(length(pred_mean) == n)

pred_bma <- predict_balasso(fit_hb, X, type = "BMA")
stopifnot(length(pred_bma) == n)

cat("  Coefficient extraction and predictions run successfully.\n")
cat("=== ALL TESTS PASSED SUCCESSFULLY! ===\n")
