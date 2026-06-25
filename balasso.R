# Bayesian Adaptive Lasso (BaLasso) Implementation in R
# Reference: Chenlei Leng, Minh-Ngoc Tran, David Nott (2014)

#' Vectorized Numerically Stable Inverse Gaussian Sampler
#'
#' Draw samples from Inverse-Gaussian(mu, lambda) using the method of
#' Michael, Schucany, and Haas (1976), with robust limiting behavior as mu -> infinity.
rinvgauss_vec <- function(mu, lambda) {
  p <- length(mu)
  z <- rnorm(p)
  y <- z^2
  
  # Identify indices where mu is extremely large or infinite
  inf_idx <- is.infinite(mu) | (mu > 1e10)
  x <- rep(0, p)
  
  # 1. Standard Case: Finite, moderate mu
  if (any(!inf_idx)) {
    mu_sub <- mu[!inf_idx]
    lambda_sub <- lambda[!inf_idx]
    y_sub <- y[!inf_idx]
    
    term1 <- mu_sub + (mu_sub^2 * y_sub) / (2 * lambda_sub)
    term2 <- (mu_sub / (2 * lambda_sub)) * sqrt(4 * mu_sub * lambda_sub * y_sub + mu_sub^2 * y_sub^2)
    x_val <- term1 - term2
    
    u <- runif(sum(!inf_idx))
    prob <- mu_sub / (mu_sub + x_val)
    
    res <- ifelse(u <= prob, x_val, mu_sub^2 / x_val)
    x[!inf_idx] <- res
  }
  
  # 2. Limiting Case: mu -> infinity
  # The density f(x; inf, lambda) corresponds to 1/Y where Y ~ Gamma(1/2, lambda/2)
  if (any(inf_idx)) {
    lambda_sub <- lambda[inf_idx]
    y_sub <- rgamma(sum(inf_idx), shape = 0.5, rate = lambda_sub / 2)
    x[inf_idx] <- 1 / y_sub
  }
  
  return(x)
}

#' Coordinate Descent Solver for Weighted Lasso
#'
#' Solve: min_beta ||y - X beta||_2^2 + sum(lambda_j |beta_j|)
solve_weighted_lasso_cov <- function(Sigma, r0, X_diag, lambda_vec, beta_init = NULL, tol = 1e-8, max_iter = 5000) {
  p <- length(r0)
  if (is.null(beta_init)) {
    beta_init <- rep(0, p)
  }
  beta <- beta_init
  s <- as.vector(Sigma %*% beta)
  
  for (iter in 1:max_iter) {
    beta_old <- beta
    for (j in 1:p) {
      if (X_diag[j] == 0) next
      
      zj <- r0[j] - s[j] + X_diag[j] * beta[j]
      val <- abs(zj) - lambda_vec[j] / 2
      if (val > 0) {
        beta_new <- sign(zj) * val / X_diag[j]
      } else {
        beta_new <- 0
      }
      
      diff_beta <- beta_new - beta[j]
      if (diff_beta != 0) {
        s <- s + Sigma[, j] * diff_beta
        beta[j] <- beta_new
      }
    }
    if (sum((beta - beta_old)^2) < tol) {
      break
    }
  }
  return(beta)
}

solve_weighted_lasso <- function(X, y, lambda_vec, beta_init = NULL, tol = 1e-8, max_iter = 5000) {
  Sigma <- t(X) %*% X
  r0 <- as.vector(t(X) %*% y)
  X_diag <- diag(Sigma)
  solve_weighted_lasso_cov(Sigma, r0, X_diag, lambda_vec, beta_init, tol, max_iter)
}

#' Core Gibbs Sampler for BaLasso and BLasso
balasso_mcmc <- function(X, y, n_samples = 15000, burn_in = 5000, 
                         method = c("HB", "EB", "BLasso"), 
                         r = 0.1, delta = 1e-6, step_coef = 1, 
                         thin = 1, verbose = TRUE) {
  method <- match.arg(method)
  n <- nrow(X)
  p <- ncol(X)
  
  # Compute X'X and X'y
  XtX <- t(X) %*% X
  Xty <- t(X) %*% y
  
  # Storage
  n_saved <- floor((n_samples - burn_in) / thin)
  if (n_saved <= 0) stop("n_samples must be greater than burn_in, and thinned samples must be > 0.")
  
  beta_samples <- matrix(0, nrow = n_saved, ncol = p)
  sigma2_samples <- rep(0, n_saved)
  tau2_samples <- matrix(0, nrow = n_saved, ncol = p)
  
  if (method == "BLasso") {
    lambda_samples <- rep(0, n_saved)
  } else {
    lambda_samples <- matrix(0, nrow = n_saved, ncol = p)
  }
  
  # Initialization
  beta_curr <- as.vector(solve(XtX + diag(1e-4, p)) %*% Xty)
  sigma2_curr <- sum((y - X %*% beta_curr)^2) / n
  if (sigma2_curr <= 0 || !is.finite(sigma2_curr)) sigma2_curr <- 1
  tau2_curr <- rep(1, p)
  
  if (method == "BLasso") {
    lambda_curr <- 1
  } else if (method == "EB") {
    lambda_curr <- rep(1, p)
    s_curr <- log(lambda_curr)
  } else { # HB
    lambda_curr <- rep(1, p)
  }
  
  for (iter in 1:n_samples) {
    A <- XtX
    diag(A) <- diag(A) + 1 / tau2_curr
    
    L <- tryCatch({
      chol(A)
    }, error = function(e) {
      diag(A) <- diag(A) + 1e-6
      chol(A)
    })
    
    mu_beta <- backsolve(L, forwardsolve(t(L), Xty))
    z <- rnorm(p)
    beta_curr <- mu_beta + sqrt(sigma2_curr) * backsolve(L, z)
    
    # 2. Update sigma2: Inv-Gamma((n-1+p)/2, (RSS + beta' D_tau^-1 beta)/2)
    shape_sig <- (n - 1) / 2 + p / 2
    rss <- sum((y - X %*% beta_curr)^2)
    pen_norm <- sum(beta_curr^2 / tau2_curr)
    scale_sig <- (rss + pen_norm) / 2
    
    sigma2_curr <- 1 / rgamma(1, shape = shape_sig, rate = scale_sig)
    
    # 3. Update tau_j^2: 1/tau_j^2 ~ Inv-Gaussian(mean = lambda_j * sigma / |beta_j|, shape = lambda_j^2)
    if (method == "BLasso") {
      mu_tau <- lambda_curr * sqrt(sigma2_curr) / abs(beta_curr)
      shape_tau <- rep(lambda_curr^2, p)
    } else {
      mu_tau <- lambda_curr * sqrt(sigma2_curr) / abs(beta_curr)
      shape_tau <- lambda_curr^2
    }
    
    inv_tau2_curr <- rinvgauss_vec(mu_tau, shape_tau)
    inv_tau2_curr[inv_tau2_curr < 1e-12] <- 1e-12
    tau2_curr <- 1 / inv_tau2_curr
    
    # 4. Update lambda
    if (method == "BLasso") {
      shape_lam <- p + r
      rate_lam <- sum(tau2_curr) / 2 + delta
      lambda2_curr <- rgamma(1, shape = shape_lam, rate = rate_lam)
      lambda_curr <- sqrt(lambda2_curr)
    } else if (method == "HB") {
      shape_lam <- 1 + r
      rate_lam <- tau2_curr / 2 + delta
      lambda2_curr <- rgamma(p, shape = shape_lam, rate = rate_lam)
      lambda_curr <- sqrt(lambda2_curr)
    } else if (method == "EB") {
      # Atchade's stochastic approximation update
      a_t <- step_coef / iter
      s_curr <- s_curr + a_t * (2 - exp(2 * s_curr) * tau2_curr)
      s_curr <- pmax(pmin(s_curr, 15), -15) # prevent overflow
      lambda_curr <- exp(s_curr)
    }
    
    # Save thinned samples after burn-in
    if (iter > burn_in && (iter - burn_in) %% thin == 0) {
      idx <- (iter - burn_in) %/% thin
      beta_samples[idx, ] <- beta_curr
      sigma2_samples[idx] <- sigma2_curr
      tau2_samples[idx, ] <- tau2_curr
      if (method == "BLasso") {
        lambda_samples[idx] <- lambda_curr
      } else {
        lambda_samples[idx, ] <- lambda_curr
      }
    }
    
    if (verbose && iter %% 5000 == 0) {
      cat(sprintf("MCMC Progress: Iteration %d / %d completed\n", iter, n_samples))
    }
  }
  
  list(
    beta = beta_samples,
    sigma2 = sigma2_samples,
    tau2 = tau2_samples,
    lambda = lambda_samples,
    method = method,
    burn_in = burn_in,
    n_samples = n_samples,
    thin = thin
  )
}

#' Fit BaLasso or BLasso Model with Centering & Standardization
fit_balasso_model <- function(X, y, standardize = FALSE, n_samples = 15000, burn_in = 5000,
                              method = c("HB", "EB", "BLasso"), r = 0.1, delta = 1e-6,
                              step_coef = 1, thin = 1, verbose = TRUE) {
  method <- match.arg(method)
  X <- as.matrix(X)
  y <- as.vector(y)
  p <- ncol(X)
  
  # Center response
  y_mean <- mean(y)
  y_c <- y - y_mean
  
  # Center design matrix
  X_mean <- colMeans(X)
  X_c <- scale(X, center = X_mean, scale = FALSE)
  
  # Standardize design matrix if requested
  if (standardize) {
    X_scale <- apply(X_c, 2, sd)
    X_scale[X_scale == 0] <- 1
    X_input <- scale(X_c, center = FALSE, scale = X_scale)
  } else {
    X_scale <- rep(1, p)
    X_input <- X_c
  }
  
  # Run MCMC sampler
  fit <- balasso_mcmc(X_input, y_c, n_samples = n_samples, burn_in = burn_in,
                      method = method, r = r, delta = delta, step_coef = step_coef,
                      thin = thin, verbose = verbose)
  
  # Add meta information
  fit$X_raw <- X
  fit$y_raw <- y
  fit$X_centered <- X_input
  fit$y_centered <- y_c
  fit$X_mean <- X_mean
  fit$y_mean <- y_mean
  fit$X_scale <- X_scale
  fit$standardize <- standardize
  
  return(fit)
}

#' Extract Coefficients in Original Scale
coef_balasso <- function(fit, type = c("Mean", "Median", "EB", "Freq")) {
  type <- match.arg(type)
  p <- length(fit$X_mean)
  
  coef_res <- balasso_select_and_estimate(fit, fit$X_centered, fit$y_centered, type = type)
  beta_scaled <- coef_res$beta
  
  # Convert beta back to the original scale
  beta_orig <- beta_scaled / fit$X_scale
  intercept <- fit$y_mean - sum(beta_orig * fit$X_mean)
  
  coefs <- c(intercept, beta_orig)
  names(coefs) <- c("(Intercept)", colnames(fit$X_raw))
  return(coefs)
}

#' Make Predictions on New Data
predict_balasso <- function(fit, X_new, type = c("Mean", "Median", "EB", "Freq", "BMA")) {
  type <- match.arg(type)
  X_new <- as.matrix(X_new)
  p <- ncol(X_new)
  
  # Center and scale X_new using fit's parameters
  X_new_c <- scale(X_new, center = fit$X_mean, scale = fit$X_scale)
  
  if (type %in% c("Mean", "Median", "EB", "Freq")) {
    coef_res <- balasso_select_and_estimate(fit, fit$X_centered, fit$y_centered, type = type)
    beta_scaled <- coef_res$beta
    y_pred <- fit$y_mean + as.vector(X_new_c %*% beta_scaled)
    return(y_pred)
    
  } else if (type == "BMA") {
    n_samples <- nrow(fit$beta)
    # Thin for prediction speed
    step_bma <- max(1, floor(n_samples / 100))
    idx_subset <- seq(1, n_samples, by = step_bma)
    
    Sigma <- t(fit$X_centered) %*% fit$X_centered
    r0 <- as.vector(t(fit$X_centered) %*% fit$y_centered)
    X_diag <- diag(Sigma)
    
    preds <- matrix(0, nrow = nrow(X_new), ncol = length(idx_subset))
    for (i in 1:length(idx_subset)) {
      s <- idx_subset[i]
      if (fit$method == "BLasso") {
        lam_s <- rep(fit$lambda[s], p)
      } else {
        lam_s <- fit$lambda[s, ]
      }
      beta_s <- solve_weighted_lasso_cov(Sigma, r0, X_diag, lam_s)
      preds[, i] <- fit$y_mean + as.vector(X_new_c %*% beta_s)
    }
    return(rowMeans(preds))
  }
}

#' Internal Helper: Select Variables and Estimate Centered Coefficients
balasso_select_and_estimate <- function(fit, X, y, type = c("Mean", "Median", "EB", "Freq")) {
  type <- match.arg(type)
  p <- ncol(X)
  
  if (type == "Mean") {
    if (fit$method == "BLasso") {
      mean_lambda <- rep(mean(fit$lambda), p)
    } else {
      mean_lambda <- colMeans(fit$lambda)
    }
    beta_est <- solve_weighted_lasso(X, y, mean_lambda)
    return(list(beta = beta_est, lambda = mean_lambda))
    
  } else if (type == "Median") {
    if (fit$method == "BLasso") {
      med_lambda <- rep(median(fit$lambda), p)
    } else {
      med_lambda <- apply(fit$lambda, 2, median)
    }
    beta_est <- solve_weighted_lasso(X, y, med_lambda)
    return(list(beta = beta_est, lambda = med_lambda))
    
  } else if (type == "EB") {
    if (fit$method != "EB") {
      stop("Type 'EB' requires a fit obtained with method = 'EB'")
    }
    eb_lambda <- colMeans(fit$lambda[floor(nrow(fit$lambda)/2):nrow(fit$lambda), , drop=FALSE])
    beta_est <- solve_weighted_lasso(X, y, eb_lambda)
    return(list(beta = beta_est, lambda = eb_lambda))
    
  } else if (type == "Freq") {
    n_samples <- nrow(fit$beta)
    # Thin for speed
    step_freq <- max(1, floor(n_samples / 100))
    idx_subset <- seq(1, n_samples, by = step_freq)
    
    Sigma <- t(X) %*% X
    r0 <- as.vector(t(X) %*% y)
    X_diag <- diag(Sigma)
    
    beta_matrix <- matrix(0, nrow = length(idx_subset), ncol = p)
    for (i in 1:length(idx_subset)) {
      s <- idx_subset[i]
      if (fit$method == "BLasso") {
        lam_s <- rep(fit$lambda[s], p)
      } else {
        lam_s <- fit$lambda[s, ]
      }
      beta_matrix[i, ] <- solve_weighted_lasso_cov(Sigma, r0, X_diag, lam_s)
    }
    
    # Calculate selection frequencies
    freqs <- colMeans(beta_matrix != 0)
    selected_vars <- which(freqs >= 0.5)
    
    # Re-estimate with mean lambda
    if (fit$method == "BLasso") {
      mean_lambda <- rep(mean(fit$lambda), p)
    } else {
      mean_lambda <- colMeans(fit$lambda)
    }
    beta_est <- solve_weighted_lasso(X, y, mean_lambda)
    
    # Force non-selected variables to 0
    beta_est_freq <- rep(0, p)
    beta_est_freq[selected_vars] <- beta_est[selected_vars]
    
    return(list(beta = beta_est_freq, frequencies = freqs, selected = selected_vars))
  }
}

# --- Competing Lasso & Adaptive Lasso Methods ---

fit_lasso <- function(X, y, nfolds = 5) {
  library(glmnet)
  cv_fit <- cv.glmnet(as.matrix(X), y, alpha = 1, nfolds = nfolds)
  best_lambda <- cv_fit$lambda.min
  fit <- glmnet(as.matrix(X), y, alpha = 1, lambda = best_lambda)
  coefs <- as.vector(coef(fit))
  names(coefs) <- c("(Intercept)", colnames(X))
  
  list(
    coef = coefs,
    best_lambda = best_lambda,
    predict = function(X_new) {
      as.vector(predict(fit, newx = as.matrix(X_new)))
    }
  )
}

fit_alasso <- function(X, y, initial_coef = c("ols", "lasso"), nfolds = 5) {
  library(glmnet)
  initial_coef <- match.arg(initial_coef)
  X <- as.matrix(X)
  p <- ncol(X)
  
  if (initial_coef == "ols") {
    if (p >= nrow(X)) {
      cv_ridge <- cv.glmnet(X, y, alpha = 0, nfolds = nfolds)
      beta_init <- as.vector(coef(cv_ridge, s = "lambda.min"))[-1]
    } else {
      beta_init <- as.vector(coef(lm(y ~ X)))[-1]
    }
  } else {
    cv_lasso <- cv.glmnet(X, y, alpha = 1, nfolds = nfolds)
    beta_init <- as.vector(coef(cv_lasso, s = "lambda.min"))[-1]
  }
  
  # Compute weights
  w <- 1 / abs(beta_init)
  w[is.infinite(w) | is.na(w)] <- 1e6
  w[w > 1e6] <- 1e6
  
  cv_fit <- cv.glmnet(X, y, alpha = 1, penalty.factor = w, nfolds = nfolds)
  best_lambda <- cv_fit$lambda.min
  fit <- glmnet(X, y, alpha = 1, penalty.factor = w, lambda = best_lambda)
  coefs <- as.vector(coef(fit))
  names(coefs) <- c("(Intercept)", colnames(X))
  
  list(
    coef = coefs,
    best_lambda = best_lambda,
    weights = w,
    predict = function(X_new) {
      as.vector(predict(fit, newx = as.matrix(X_new)))
    }
  )
}
