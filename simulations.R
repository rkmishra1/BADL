# Simulation Studies for Bayesian Adaptive Lasso (BaLasso)
# Reproduces Examples 1, 2, 3, and 4 from the paper
source("balasso.R")
library(parallel)

run_parallel <- function(n_reps, n_cores, run_rep_func, export_vars = NULL, env = parent.frame()) {
  cl <- makeCluster(n_cores)
  on.exit(stopCluster(cl))
  if (!is.null(export_vars)) {
    clusterExport(cl, varlist = export_vars, envir = env)
  }
  clusterEvalQ(cl, {
    library(Matrix)
    library(glmnet)
    source("balasso.R")
  })
  res <- parLapply(cl, 1:n_reps, run_rep_func)
  return(do.call(rbind, res))
}

# Setup
QUICK_RUN <- FALSE # Set to TRUE for a fast test, FALSE for the paper's full run
n_reps <- if (QUICK_RUN) 5 else 100
n_cores <- max(1, detectCores() - 1)

cat(sprintf("Running simulations with %d replications on %d cores...\n", n_reps, n_cores))

# Utility function to generate AR(1) design matrix
generate_ar1_X <- function(n, p, rho = 0.5) {
  Sigma <- outer(1:p, 1:p, function(i, j) rho^abs(i - j))
  R <- chol(Sigma) # R^T R = Sigma, R is upper triangular
  Z <- matrix(rnorm(n * p), n, p)
  X <- Z %*% R
  colnames(X) <- paste0("X", 1:p)
  return(X)
}

# Utility function to generate Zou (2006) design matrix
generate_zou_X <- function(n) {
  p <- 4
  Sigma <- matrix(0, p, p)
  for (i in 1:3) {
    for (j in 1:3) {
      if (i == j) Sigma[i, j] <- 1 else Sigma[i, j] <- -0.39
    }
    Sigma[i, 4] <- 0.23
    Sigma[4, i] <- 0.23
  }
  Sigma[4, 4] <- 1
  R <- chol(Sigma)
  Z <- matrix(rnorm(n * p), n, p)
  X <- Z %*% R
  colnames(X) <- paste0("X", 1:p)
  return(X)
}

# --- EXAMPLE 1: Simple Example ---
run_example_1 <- function() {
  cat("\n=== RUNNING EXAMPLE 1 ===\n")
  p <- 8
  beta_true <- c(3, 1.5, 0, 0, 2, 0, 0, 0)
  active_idx <- c(1, 2, 5)
  
  settings <- list(
    list(n = 30, sigma = 1),
    list(n = 30, sigma = 3),
    list(n = 60, sigma = 1),
    list(n = 60, sigma = 3),
    list(n = 120, sigma = 1),
    list(n = 120, sigma = 3)
  )
  
  results <- data.frame()
  
  for (set in settings) {
    n <- set$n
    sig <- set$sigma
    cat(sprintf("Setting: n = %d, sigma = %d\n", n, sig))
    
    # Define a single replication function for mclapply
    run_rep <- function(rep_id) {
      # Generate data
      X <- generate_ar1_X(n, p, rho = 0.5)
      y <- X %*% beta_true + rnorm(n, sd = sig)
      
      # 1. Lasso
      fit_las <- fit_lasso(X, y)
      las_ok <- all((fit_las$coef[-1] != 0) == (beta_true != 0))
      
      # 2. aLasso (OLS weights)
      fit_al <- fit_alasso(X, y, initial_coef = "ols")
      al_ok <- all((fit_al$coef[-1] != 0) == (beta_true != 0))
      
      # 3. BaLasso HB
      # We run 12000 MCMC samples, 4000 burn-in, thin by 8
      fit_hb <- fit_balasso_model(X, y, method = "HB", n_samples = 4000, burn_in = 1000, thin = 3, verbose = FALSE)
      
      # BaLasso-Freq
      sel_freq <- balasso_select_and_estimate(fit_hb, fit_hb$X_centered, fit_hb$y_centered, type = "Freq")
      freq_ok <- all((sel_freq$beta != 0) == (beta_true != 0))
      
      # BaLasso-Median
      sel_med <- balasso_select_and_estimate(fit_hb, fit_hb$X_centered, fit_hb$y_centered, type = "Median")
      med_ok <- all((sel_med$beta != 0) == (beta_true != 0))
      
      # BaLasso-Mean
      sel_mean <- balasso_select_and_estimate(fit_hb, fit_hb$X_centered, fit_hb$y_centered, type = "Mean")
      mean_ok <- all((sel_mean$beta != 0) == (beta_true != 0))
      
      # 4. BaLasso EB
      fit_eb <- fit_balasso_model(X, y, method = "EB", n_samples = 4000, burn_in = 1000, thin = 3, step_coef = 2, verbose = FALSE)
      sel_eb <- balasso_select_and_estimate(fit_eb, fit_eb$X_centered, fit_eb$y_centered, type = "EB")
      eb_ok <- all((sel_eb$beta != 0) == (beta_true != 0))
      
      return(c(las = las_ok, al = al_ok, freq = freq_ok, med = med_ok, mean = mean_ok, eb = eb_ok))
    }
    
    rep_res <- run_parallel(n_reps, n_cores, run_rep,
                            export_vars = c("n", "sig", "beta_true", "p", "generate_ar1_X"))
    
    # Calculate frequencies of correct selection
    freqs <- colMeans(rep_res) * 100
    
    row_res <- data.frame(
      n = n,
      sigma = sig,
      Lasso = freqs["las"],
      aLasso = freqs["al"],
      BaLasso_Freq = freqs["freq"],
      BaLasso_Median = freqs["med"],
      BaLasso_Mean = freqs["mean"],
      BaLasso_EB = freqs["eb"]
    )
    results <- rbind(results, row_res)
  }
  
  rownames(results) <- NULL
  write.csv(results, "table1_example1.csv", row.names = FALSE)
  print(results)
}

# --- EXAMPLE 2: Difficult Example ---
run_example_2 <- function() {
  cat("\n=== RUNNING EXAMPLE 2 ===\n")
  p <- 4
  beta_true <- c(5.6, 5.6, 5.6, 0)
  
  settings <- list(
    list(n = 60, sigma = 9),
    list(n = 120, sigma = 5),
    list(n = 300, sigma = 3),
    list(n = 300, sigma = 1)
  )
  
  results <- data.frame()
  
  for (set in settings) {
    n <- set$n
    sig <- set$sigma
    cat(sprintf("Setting: n = %d, sigma = %d\n", n, sig))
    
    run_rep <- function(rep_id) {
      X <- generate_zou_X(n)
      y <- X %*% beta_true + rnorm(n, sd = sig)
      
      # 1. Lasso
      fit_las <- fit_lasso(X, y)
      las_ok <- all((fit_las$coef[-1] != 0) == (beta_true != 0))
      
      # 2. aLasso (OLS weights)
      fit_al <- fit_alasso(X, y, initial_coef = "ols")
      al_ok <- all((fit_al$coef[-1] != 0) == (beta_true != 0))
      
      # 3. BaLasso HB
      fit_hb <- fit_balasso_model(X, y, method = "HB", n_samples = 4000, burn_in = 1000, thin = 3, verbose = FALSE)
      
      sel_freq <- balasso_select_and_estimate(fit_hb, fit_hb$X_centered, fit_hb$y_centered, type = "Freq")
      freq_ok <- all((sel_freq$beta != 0) == (beta_true != 0))
      
      sel_med <- balasso_select_and_estimate(fit_hb, fit_hb$X_centered, fit_hb$y_centered, type = "Median")
      med_ok <- all((sel_med$beta != 0) == (beta_true != 0))
      
      sel_mean <- balasso_select_and_estimate(fit_hb, fit_hb$X_centered, fit_hb$y_centered, type = "Mean")
      mean_ok <- all((sel_mean$beta != 0) == (beta_true != 0))
      
      # 4. BaLasso EB
      fit_eb <- fit_balasso_model(X, y, method = "EB", n_samples = 4000, burn_in = 1000, thin = 3, step_coef = 2, verbose = FALSE)
      sel_eb <- balasso_select_and_estimate(fit_eb, fit_eb$X_centered, fit_eb$y_centered, type = "EB")
      eb_ok <- all((sel_eb$beta != 0) == (beta_true != 0))
      
      return(c(las = las_ok, al = al_ok, freq = freq_ok, med = med_ok, mean = mean_ok, eb = eb_ok))
    }
    
    rep_res <- run_parallel(n_reps, n_cores, run_rep,
                            export_vars = c("n", "sig", "beta_true", "generate_zou_X"))
    
    freqs <- colMeans(rep_res) * 100
    
    row_res <- data.frame(
      n = n,
      sigma = sig,
      Lasso = freqs["las"],
      aLasso = freqs["al"],
      BaLasso_Freq = freqs["freq"],
      BaLasso_Median = freqs["med"],
      BaLasso_Mean = freqs["mean"],
      BaLasso_EB = freqs["eb"]
    )
    results <- rbind(results, row_res)
  }
  
  rownames(results) <- NULL
  write.csv(results, "table2_example2.csv", row.names = FALSE)
  print(results)
}

# --- EXAMPLE 3: Large p Example ---
run_example_3 <- function() {
  cat("\n=== RUNNING EXAMPLE 3 ===\n")
  p <- 100
  beta_true <- rep(0, p)
  active_idx <- seq(10, 100, by = 10)
  beta_true[active_idx] <- 5
  
  settings <- list(
    list(n = 50, sigma = 1),
    list(n = 50, sigma = 3),
    list(n = 50, sigma = 5),
    list(n = 100, sigma = 1),
    list(n = 100, sigma = 3),
    list(n = 100, sigma = 5),
    list(n = 200, sigma = 1),
    list(n = 200, sigma = 3),
    list(n = 200, sigma = 5)
  )
  
  results <- data.frame()
  
  for (set in settings) {
    n <- set$n
    sig <- set$sigma
    cat(sprintf("Setting: n = %d, sigma = %d\n", n, sig))
    
    run_rep <- function(rep_id) {
      X <- generate_ar1_X(n, p, rho = 0.5)
      y <- X %*% beta_true + rnorm(n, sd = sig)
      
      # 1. aLasso (Lasso weights since p >= n)
      fit_al <- fit_alasso(X, y, initial_coef = "lasso")
      al_ok <- all((fit_al$coef[-1] != 0) == (beta_true != 0))
      
      # 2. BaLasso HB
      fit_hb <- fit_balasso_model(X, y, method = "HB", n_samples = 4000, burn_in = 1000, thin = 3, verbose = FALSE)
      
      sel_freq <- balasso_select_and_estimate(fit_hb, fit_hb$X_centered, fit_hb$y_centered, type = "Freq")
      freq_ok <- all((sel_freq$beta != 0) == (beta_true != 0))
      
      sel_med <- balasso_select_and_estimate(fit_hb, fit_hb$X_centered, fit_hb$y_centered, type = "Median")
      med_ok <- all((sel_med$beta != 0) == (beta_true != 0))
      
      sel_mean <- balasso_select_and_estimate(fit_hb, fit_hb$X_centered, fit_hb$y_centered, type = "Mean")
      mean_ok <- all((sel_mean$beta != 0) == (beta_true != 0))
      
      # 3. BaLasso EB
      fit_eb <- fit_balasso_model(X, y, method = "EB", n_samples = 4000, burn_in = 1000, thin = 3, step_coef = 2, verbose = FALSE)
      sel_eb <- balasso_select_and_estimate(fit_eb, fit_eb$X_centered, fit_eb$y_centered, type = "EB")
      eb_ok <- all((sel_eb$beta != 0) == (beta_true != 0))
      
      return(c(al = al_ok, freq = freq_ok, med = med_ok, mean = mean_ok, eb = eb_ok))
    }
    
    rep_res <- run_parallel(n_reps, n_cores, run_rep,
                            export_vars = c("n", "sig", "beta_true", "p", "generate_ar1_X"))
    
    freqs <- colMeans(rep_res) * 100
    
    row_res <- data.frame(
      n = n,
      sigma = sig,
      aLasso = freqs["al"],
      BaLasso_Freq = freqs["freq"],
      BaLasso_Median = freqs["med"],
      BaLasso_Mean = freqs["mean"],
      BaLasso_EB = freqs["eb"]
    )
    results <- rbind(results, row_res)
  }
  
  rownames(results) <- NULL
  write.csv(results, "table3_example3.csv", row.names = FALSE)
  print(results)
}

# --- EXAMPLE 4: Prediction ---
run_example_4 <- function() {
  cat("\n=== RUNNING EXAMPLE 4: PREDICTION ===\n")
  
  # A. Small p case
  cat("Running Small-p case...\n")
  p_small <- 8
  beta_small <- c(3, 1.5, 0.1, 0.1, 2, 0, 0, 0)
  
  settings_small <- list(
    list(nT = 30, nP = 30, sigma = 1),
    list(nT = 30, nP = 30, sigma = 3),
    list(nT = 30, nP = 30, sigma = 5),
    list(nT = 30, nP = 30, sigma = 10),
    list(nT = 100, nP = 100, sigma = 1),
    list(nT = 100, nP = 100, sigma = 3),
    list(nT = 100, nP = 100, sigma = 5),
    list(nT = 100, nP = 100, sigma = 10),
    list(nT = 200, nP = 200, sigma = 1),
    list(nT = 200, nP = 200, sigma = 3),
    list(nT = 200, nP = 200, sigma = 5),
    list(nT = 200, nP = 200, sigma = 10)
  )
  
  results_small <- data.frame()
  
  for (set in settings_small) {
    nT <- set$nT
    nP <- set$nP
    sig <- set$sigma
    cat(sprintf("Setting Small-p: nT = %d, nP = %d, sigma = %d\n", nT, nP, sig))
    
    run_rep_small <- function(rep_id) {
      # Generate training and prediction sets
      # Combined size is nT + nP
      X_all <- generate_ar1_X(nT + nP, p_small, rho = 0.5)
      y_all <- X_all %*% beta_small + rnorm(nT + nP, sd = sig)
      
      X_tr <- X_all[1:nT, ]
      y_tr <- y_all[1:nT]
      X_pr <- X_all[(nT + 1):(nT + nP), ]
      y_pr <- y_all[(nT + 1):(nT + nP)]
      
      # 1. Lasso
      fit_las <- fit_lasso(X_tr, y_tr)
      pred_las <- fit_las$predict(X_pr)
      pse_las <- mean((y_pr - pred_las)^2)
      
      # 2. aLasso (OLS weights)
      fit_al <- fit_alasso(X_tr, y_tr, initial_coef = "ols")
      pred_al <- fit_al$predict(X_pr)
      pse_al <- mean((y_pr - pred_al)^2)
      
      # 3. BLasso
      fit_bl <- fit_balasso_model(X_tr, y_tr, method = "BLasso", n_samples = 4000, burn_in = 1000, thin = 3, verbose = FALSE)
      pred_bl <- predict_balasso(fit_bl, X_pr, type = "BMA")
      pse_bl <- mean((y_pr - pred_bl)^2)
      
      # 4. BaLasso-Mean
      fit_hb <- fit_balasso_model(X_tr, y_tr, method = "HB", n_samples = 4000, burn_in = 1000, thin = 3, verbose = FALSE)
      pred_mean <- predict_balasso(fit_hb, X_pr, type = "Mean")
      pse_mean <- mean((y_pr - pred_mean)^2)
      
      # 5. BaLasso-BMA
      pred_bma <- predict_balasso(fit_hb, X_pr, type = "BMA")
      pse_bma <- mean((y_pr - pred_bma)^2)
      
      return(c(las = pse_las, al = pse_al, bl = pse_bl, mean = pse_mean, bma = pse_bma))
    }
    
    rep_res <- run_parallel(n_reps, n_cores, run_rep_small,
                            export_vars = c("nT", "nP", "p_small", "beta_small", "sig", "generate_ar1_X"))
    
    pse_avg <- colMeans(rep_res)
    
    row_res <- data.frame(
      nT_nP = nT,
      sigma = sig,
      Lasso = pse_avg["las"],
      aLasso = pse_avg["al"],
      BLasso = pse_avg["bl"],
      BaLasso_Mean = pse_avg["mean"],
      BaLasso_BMA = pse_avg["bma"]
    )
    results_small <- rbind(results_small, row_res)
  }
  
  write.csv(results_small, "table4_prediction_small.csv", row.names = FALSE)
  print(results_small)
  
  # B. Large p case
  cat("\nRunning Large-p case...\n")
  p_large <- 100
  beta_large <- rep(0, p_large)
  active_idx <- seq(10, 100, by = 10)
  beta_large[active_idx] <- 5
  # Insert model uncertainty by setting first 5 active coefficients to 0.5
  beta_large[seq(10, 50, by = 10)] <- 0.5
  
  settings_large <- list(
    list(nT = 100, nP = 100, sigma = 1),
    list(nT = 100, nP = 100, sigma = 3),
    list(nT = 100, nP = 100, sigma = 5),
    list(nT = 100, nP = 100, sigma = 10),
    list(nT = 200, nP = 200, sigma = 1),
    list(nT = 200, nP = 200, sigma = 3),
    list(nT = 200, nP = 200, sigma = 5),
    list(nT = 200, nP = 200, sigma = 10)
  )
  
  results_large <- data.frame()
  
  for (set in settings_large) {
    nT <- set$nT
    nP <- set$nP
    sig <- set$sigma
    cat(sprintf("Setting Large-p: nT = %d, nP = %d, sigma = %d\n", nT, nP, sig))
    
    run_rep_large <- function(rep_id) {
      X_all <- generate_ar1_X(nT + nP, p_large, rho = 0.5)
      y_all <- X_all %*% beta_large + rnorm(nT + nP, sd = sig)
      
      X_tr <- X_all[1:nT, ]
      y_tr <- y_all[1:nT]
      X_pr <- X_all[(nT + 1):(nT + nP), ]
      y_pr <- y_all[(nT + 1):(nT + nP)]
      
      # 1. Lasso
      fit_las <- fit_lasso(X_tr, y_tr)
      pred_las <- fit_las$predict(X_pr)
      pse_las <- mean((y_pr - pred_las)^2)
      
      # 2. aLasso (Lasso weights since p >= n)
      fit_al <- fit_alasso(X_tr, y_tr, initial_coef = "lasso")
      pred_al <- fit_al$predict(X_pr)
      pse_al <- mean((y_pr - pred_al)^2)
      
      # 3. BLasso
      fit_bl <- fit_balasso_model(X_tr, y_tr, method = "BLasso", n_samples = 4000, burn_in = 1000, thin = 3, verbose = FALSE)
      pred_bl <- predict_balasso(fit_bl, X_pr, type = "BMA")
      pse_bl <- mean((y_pr - pred_bl)^2)
      
      # 4. BaLasso-Mean
      fit_hb <- fit_balasso_model(X_tr, y_tr, method = "HB", n_samples = 4000, burn_in = 1000, thin = 3, verbose = FALSE)
      pred_mean <- predict_balasso(fit_hb, X_pr, type = "Mean")
      pse_mean <- mean((y_pr - pred_mean)^2)
      
      # 5. BaLasso-BMA
      pred_bma <- predict_balasso(fit_hb, X_pr, type = "BMA")
      pse_bma <- mean((y_pr - pred_bma)^2)
      
      return(c(las = pse_las, al = pse_al, bl = pse_bl, mean = pse_mean, bma = pse_bma))
    }
    
    rep_res <- run_parallel(n_reps, n_cores, run_rep_large,
                            export_vars = c("nT", "nP", "p_large", "beta_large", "sig", "generate_ar1_X"))
    
    pse_avg <- colMeans(rep_res)
    
    row_res <- data.frame(
      nT_nP = nT,
      sigma = sig,
      Lasso = pse_avg["las"],
      aLasso = pse_avg["al"],
      BLasso = pse_avg["bl"],
      BaLasso_Mean = pse_avg["mean"],
      BaLasso_BMA = pse_avg["bma"]
    )
    results_large <- rbind(results_large, row_res)
  }
  
  write.csv(results_large, "table5_prediction_large.csv", row.names = FALSE)
  print(results_large)
}

# --- RUN ALL EXAMPLES ---
run_example_1()
run_example_2()
run_example_3()
run_example_4()

cat("\n=== ALL SIMULATIONS COMPLETED SUCCESSFULLY! ===\n")
