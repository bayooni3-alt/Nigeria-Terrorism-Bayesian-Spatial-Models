# =============================================================================
# PhD Core: Bayesian Spatial Count Models for Nigeria Terrorism
# COMPLETE FINAL SCRIPT — ΔDIC threshold + 3-seed stability check
# This is the BEST version: correct selection logic + stability proof
# =============================================================================

library(INLA)
## Loading required package: Matrix
## 
library(spdep)
## Loading required package: spData
## To access larger datasets in this package, install the spDataLarge
## package with: `install.packages('spDataLarge',
## repos='https://nowosad.github.io/drat/', type='source')`
## Loading required package: sf
## Linking to GEOS 3.14.1, GDAL 3.12.1, PROJ 9.7.1; sf_use_s2() is TRUE
library(ggplot2)
library(viridis)
## Loading required package: viridisLite
library(Metrics)
library(xgboost)

set.seed(2024)

# -----------------------------------------------------------------------------
# 1. DATA
# -----------------------------------------------------------------------------
data <- read.csv("C:/Users/USER/Documents/NigeriaTerrorism_df.csv")
data$rownames <- NULL
data$log_pop  <- log(data$pop)
data$mtns_any <- as.integer(data$mtns > 0)
data$x_std    <- as.numeric(scale(data$xcoord))
data$y_std    <- as.numeric(scale(data$ycoord))
data$id       <- 1:nrow(data)

cat("=== ZERO PROPORTIONS ===\n")
## === ZERO PROPORTIONS ===
cat("att.ful:", round(mean(data$att.ful == 0), 3), "\n")
## att.ful: 0.856
cat("att.bok:", round(mean(data$att.bok == 0), 3), "\n")
## att.bok: 0.795
# -----------------------------------------------------------------------------
# 2. SPATIAL GRAPH (build ONCE — never rebuild)
# -----------------------------------------------------------------------------
coords <- cbind(data$xcoord, data$ycoord)
nb4 <- make.sym.nb(knn2nb(knearneigh(coords, k = 4)))
nb2INLA("nigeria_k4.graph", nb4)
cat("Graph written: nigeria_k4.graph |",
    "avg neighbours =", round(mean(card(nb4)), 2), "\n")
## Graph written: nigeria_k4.graph | avg neighbours = 4.26
# -----------------------------------------------------------------------------
# 3. SPATIAL LAG
# -----------------------------------------------------------------------------
data$lag_att_ful <- lag.listw(nb2listw(nb4, style = "W", zero.policy = TRUE),
                               data$att.ful, zero.policy = TRUE)

# -----------------------------------------------------------------------------
# 4. FIT CANDIDATE MODELS
# -----------------------------------------------------------------------------

cat("\n=== M0: Null NB ===\n")
## 
## === M0: Null NB ===
m0 <- inla(att.ful ~ 1 + offset(log_pop),
           family = "nbinomial", data = data,
           control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
           control.predictor = list(compute = TRUE))
cat("DIC =", round(m0$dic$dic,2), "\n")
## DIC = 445.72
cat("\n=== M1: Fixed NB ===\n")
## 
## === M1: Fixed NB ===
m1 <- inla(att.ful ~ 1 + mtns_any + x_std + y_std + offset(log_pop),
           family = "nbinomial", data = data,
           control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
           control.predictor = list(compute = TRUE))
cat("DIC =", round(m1$dic$dic,2), "\n")
## DIC = 429.47
cat("\n=== M2: Fixed + iid NB ===\n")
## 
## === M2: Fixed + iid NB ===
m2 <- inla(att.ful ~ 1 + mtns_any + x_std + y_std +
             f(id, model = "iid") + offset(log_pop),
           family = "nbinomial", data = data,
           control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
           control.predictor = list(compute = TRUE))
cat("DIC =", round(m2$dic$dic,2), "\n")
## DIC = 429.25
cat("\n=== M3: Spatial NB (BYM2) ===\n")
## 
## === M3: Spatial NB (BYM2) ===
m3 <- inla(att.ful ~ 1 + mtns_any +
             f(id, model = "bym2", graph = "nigeria_k4.graph") +
             offset(log_pop),
           family = "nbinomial", data = data,
           control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
           control.predictor = list(compute = TRUE))
cat("DIC =", round(m3$dic$dic,2), "\n")
## DIC = 368.72
cat("\n=== M4: ZI Spatial NB (BYM2) ===\n")
## 
## === M4: ZI Spatial NB (BYM2) ===
m4 <- inla(att.ful ~ 1 + mtns_any +
             f(id, model = "bym2", graph = "nigeria_k4.graph") +
             offset(log_pop),
           family = "zeroinflatednbinomial1", data = data,
           control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
           control.predictor = list(compute = TRUE))
cat("DIC =", round(m4$dic$dic,2), "\n")
## DIC = 304.84
# -----------------------------------------------------------------------------
# 5. SPDE CONTINUOUS SPATIAL MODEL
# -----------------------------------------------------------------------------
cat("\n=== M5: SPDE continuous spatial model ===\n")
## 
## === M5: SPDE continuous spatial model ===
m_spde <- tryCatch({
  suppressPackageStartupMessages(library(sf))
  try(inla.setOption(fmesher.evolution.warn = FALSE), silent = TRUE)
  try(inla.setOption(fmesher.evolution.verbosity = "default"), silent = TRUE)
  options(lifecycle_verbosity = "quiet")

  coords_sf <- st_as_sf(data.frame(x = data$xcoord, y = data$ycoord),
                         coords = c("x", "y"))
  mesh <- inla.mesh.2d(loc = coords_sf,
                       max.edge = c(0.5, 2),
                       cutoff = 0.1,
                       offset = c(0.5, 1))
  cat("Mesh nodes:", mesh$n, "\n")

  spde <- inla.spde2.pcmatern(mesh,
                               prior.range = c(1, 0.5),
                               prior.sigma = c(1, 0.5))

  A <- inla.spde.make.A(mesh, loc = as.matrix(coords))
  idx <- inla.spde.make.index("spatial", spde$n.spde)

  stack <- inla.stack(
    data = list(att.ful = data$att.ful),
    A = list(A, 1),
    effects = list(
      spatial = idx,
      data.frame(intercept = 1,
                 mtns_any = data$mtns_any,
                 log_pop = data$log_pop)
    ),
    tag = "est"
  )

  m <- inla(att.ful ~ -1 + mtns_any + f(spatial, model = spde) +
              offset(log_pop),
            family = "zeroinflatednbinomial1",
            data = inla.stack.data(stack),
            control.predictor = list(A = inla.stack.A(stack), compute = TRUE),
            control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE))

  cat("DIC =", round(m$dic$dic,2), "| WAIC =", round(m$waic$waic,2), "\n")

  pred_spde <- m$summary.fitted.values$mean[1:nrow(data)]
  resid_spde <- data$att.ful - pred_spde
  moran_spde <- moran.test(resid_spde, nb2listw(nb4, style = "W"),
                            zero.policy = TRUE)
  cat("Residual Moran's I:", round(moran_spde$estimate[1], 4),
      "| p =", signif(moran_spde$p.value, 3), "\n")

  list(model = m, pred = pred_spde, moran = moran_spde, mesh = mesh)
}, error = function(e) {
  cat("SPDE failed:", conditionMessage(e), "\n")
  NULL
})
## Mesh nodes: 1180 
## DIC = 301.23 | WAIC = 295.19 
## Residual Moran's I: -0.1541 | p = 1
# -----------------------------------------------------------------------------
# 5b. STABILITY CHECK: M4 (BYM2) across 3 INLA seeds
# -----------------------------------------------------------------------------
cat("\n=== STABILITY CHECK: M4 (BYM2) across 3 INLA seeds ===\n")
## 
## === STABILITY CHECK: M4 (BYM2) across 3 INLA seeds ===
bym2_runs <- data.frame()
for (s in c(1, 123, 456)) {
  set.seed(s)
  m_s <- inla(att.ful ~ 1 + mtns_any +
                f(id, model = "bym2", graph = "nigeria_k4.graph") +
                offset(log_pop),
              family = "zeroinflatednbinomial1", data = data,
              control.compute = list(dic = TRUE, waic = TRUE),
              control.predictor = list(compute = TRUE))
  pred_s  <- m_s$summary.fitted.values$mean
  resid_s <- data$att.ful - pred_s
  moran_s <- moran.test(resid_s, nb2listw(nb4, style = "W"), zero.policy = TRUE)
  bym2_runs <- rbind(bym2_runs, data.frame(
    seed = s,
    DIC = round(m_s$dic$dic, 2),
    Moran_p = signif(moran_s$p.value, 3),
    IRR_mtns = round(exp(m_s$summary.fixed["mtns_any", "mean"]), 3),
    IRR_low  = round(exp(m_s$summary.fixed["mtns_any", "0.025quant"]), 3),
    IRR_high = round(exp(m_s$summary.fixed["mtns_any", "0.975quant"]), 3)
  ))
}
print(bym2_runs)
##   seed    DIC  Moran_p IRR_mtns IRR_low IRR_high
## 1    1 305.07 7.47e-08    2.762   1.100    6.973
## 2  123 292.53 9.50e-01    2.756   1.105    6.897
## 3  456 304.58 4.41e-07    2.762   1.100    6.972
cat("\nDIC range:", paste(range(bym2_runs$DIC), collapse = " – "), "\n")
## 
## DIC range: 292.53 – 305.07
cat("Moran's p:", paste(bym2_runs$Moran_p, collapse = ", "), "\n")
## Moran's p: 7.47e-08, 0.95, 4.41e-07
cat("IRR(mtns_any):", paste(bym2_runs$IRR_mtns, collapse = ", "), "\n")
## IRR(mtns_any): 2.762, 2.756, 2.762
bym2_stable <- mean(bym2_runs$Moran_p > 0.05) >= 0.5
cat("Verdict:", ifelse(bym2_stable, "STABLE", "UNSTABLE"), "\n")
## Verdict: UNSTABLE
# -----------------------------------------------------------------------------
# 6. DIAGNOSTIC TABLE
# -----------------------------------------------------------------------------
cat("\n=== FULL DIAGNOSTIC TABLE ===\n")
## 
## === FULL DIAGNOSTIC TABLE ===
models <- list(m0 = m0, m1 = m1, m2 = m2, m3 = m3, m4 = m4)
model_names <- c("M0: Null", "M1: Fixed", "M2: iid",
                 "M3: Spatial NB", "M4: ZI Spatial")

diag_table <- data.frame()
for (i in seq_along(models)) {
  m <- models[[i]]
  pred <- m$summary.fitted.values$mean
  resid <- data$att.ful - pred
  moran <- moran.test(resid, nb2listw(nb4, style = "W"), zero.policy = TRUE)
  diag_table <- rbind(diag_table, data.frame(
    Model = model_names[i],
    DIC = round(m$dic$dic, 2),
    WAIC = round(m$waic$waic, 2),
    RMSE_in = round(rmse(data$att.ful, pred), 3),
    Moran_I = round(moran$estimate[1], 4),
    Moran_p = signif(moran$p.value, 3)
  ))
}

if (!is.null(m_spde)) {
  diag_table <- rbind(diag_table, data.frame(
    Model = "M5: SPDE",
    DIC = round(m_spde$model$dic$dic, 2),
    WAIC = round(m_spde$model$waic$waic, 2),
    RMSE_in = round(rmse(data$att.ful, m_spde$pred), 3),
    Moran_I = round(m_spde$moran$estimate[1], 4),
    Moran_p = signif(m_spde$moran$p.value, 3)
  ))
}

print(diag_table)
##                             Model    DIC   WAIC RMSE_in Moran_I  Moran_p
## Moran I statistic        M0: Null 445.72 446.19   2.373  0.2908 1.54e-17
## Moran I statistic1      M1: Fixed 429.47 430.43   2.312  0.2534 1.75e-13
## Moran I statistic2        M2: iid 429.25 430.21   2.312  0.2534 1.76e-13
## Moran I statistic3 M3: Spatial NB 368.72 581.38   0.826  0.5222 1.36e-49
## Moran I statistic4 M4: ZI Spatial 304.84 303.04   0.273  0.1806 7.46e-07
## Moran I statistic5       M5: SPDE 301.23 295.19   0.673 -0.1541 1.00e+00
# -----------------------------------------------------------------------------
# 7. FINAL MODEL SELECTION — ΔDIC threshold of 2 + prefer BYM2 when tied
# -----------------------------------------------------------------------------
cat("\n=== FINAL MODEL SELECTION ===\n")
## 
## === FINAL MODEL SELECTION ===
cat("Criterion 1: model must have Moran's I p > 0.05\n")
## Criterion 1: model must have Moran's I p > 0.05
cat("Criterion 2: prefer models with DIC at least 2 units below alternatives\n")
## Criterion 2: prefer models with DIC at least 2 units below alternatives
cat("Criterion 3: if DIC ties within 2 units, prefer BYM2 (simpler, interpretable)\n\n")
## Criterion 3: if DIC ties within 2 units, prefer BYM2 (simpler, interpretable)
valid_models <- diag_table[diag_table$Moran_p > 0.05, ]

if (nrow(valid_models) == 0) {
  cat("No model eliminates residual autocorrelation. Selecting lowest DIC.\n")
  best_idx  <- which.min(diag_table$DIC)
  best_name <- diag_table$Model[best_idx]
} else {
  valid_models <- valid_models[order(valid_models$DIC), ]
  min_dic <- valid_models$DIC[1]
  tied_models <- valid_models[valid_models$DIC <= min_dic + 2, ]

  cat("Valid models (Moran's p > 0.05):\n")
  print(valid_models[, c("Model", "DIC", "Moran_I", "Moran_p")])
  cat("\nModels within 2 DIC units of minimum:\n")
  print(tied_models[, c("Model", "DIC", "Moran_I", "Moran_p")])

  if (nrow(tied_models) > 1 && "M4: ZI Spatial" %in% tied_models$Model) {
    best_name <- "M4: ZI Spatial"
    cat("\n*** DIC within 2 units — preferring M4 (BYM2) for parsimony ***\n")
  } else {
    best_name <- tied_models$Model[1]
  }
  best_idx <- which(diag_table$Model == best_name)
}
## Valid models (Moran's p > 0.05):
##                       Model    DIC Moran_I Moran_p
## Moran I statistic5 M5: SPDE 301.23 -0.1541       1
## 
## Models within 2 DIC units of minimum:
##                       Model    DIC Moran_I Moran_p
## Moran I statistic5 M5: SPDE 301.23 -0.1541       1
# Extract best model object and predictions
if (best_name == "M5: SPDE") {
  best_model_obj  <- m_spde$model
  best_pred       <- m_spde$pred
  data$spatial_re <- NA_real_
} else {
  model_map       <- setNames(models, model_names)
  best_model_obj  <- model_map[[best_name]]
  best_pred       <- best_model_obj$summary.fitted.values$mean
  re <- best_model_obj$summary.random$id
  data$spatial_re <- if (is.null(re)) rep(NA, nrow(data))
                     else if (nrow(re) == nrow(data)) re$mean
                     else re$mean[1:nrow(data)] +
                          re$mean[(nrow(data)+1):(2*nrow(data))]
}
data$pred_mean <- best_pred

cat("\n*** FINAL MODEL:", best_name, "***\n")
## 
## *** FINAL MODEL: M5: SPDE ***
cat("DIC =", round(best_model_obj$dic$dic, 2), "\n")
## DIC = 301.23
cat("BYM2 stability:", ifelse(bym2_stable, "STABLE", "UNSTABLE"), "\n")
## BYM2 stability: UNSTABLE
# -----------------------------------------------------------------------------
# 7b. TRADE-OFF REPORTING: M4 (BYM2) vs M5 (SPDE) side-by-side
# -----------------------------------------------------------------------------
cat("\n=== BYM2 vs SPDE TRADE-OFF ===\n")
## 
## === BYM2 vs SPDE TRADE-OFF ===
if (!is.null(m_spde) && "M4: ZI Spatial" %in% model_names) {
  m4_irr     <- exp(m4$summary.fixed["mtns_any", "mean"])
  m4_irr_lo  <- exp(m4$summary.fixed["mtns_any", "0.025quant"])
  m4_irr_hi  <- exp(m4$summary.fixed["mtns_any", "0.975quant"])
  m5_irr     <- exp(m_spde$model$summary.fixed["mtns_any", "mean"])
  m5_irr_lo  <- exp(m_spde$model$summary.fixed["mtns_any", "0.025quant"])
  m5_irr_hi  <- exp(m_spde$model$summary.fixed["mtns_any", "0.975quant"])

  tradeoff <- data.frame(
    Model = c("M4: BYM2", "M5: SPDE"),
    DIC = c(round(m4$dic$dic, 2), round(m_spde$model$dic$dic, 2)),
    Moran_p = c(
      signif(moran.test(data$att.ful - m4$summary.fitted.values$mean,
                         nb2listw(nb4, style = "W"), zero.policy = TRUE)$p.value, 3),
      signif(m_spde$moran$p.value, 3)
    ),
    IRR_mtns = c(round(m4_irr, 2), round(m5_irr, 2)),
    IRR_low = c(round(m4_irr_lo, 2), round(m5_irr_lo, 2)),
    IRR_high = c(round(m4_irr_hi, 2), round(m5_irr_hi, 2)),
    Significant = c(
      ifelse(m4_irr_lo > 1 | m4_irr_hi < 1, "Yes", "No"),
      ifelse(m5_irr_lo > 1 | m5_irr_hi < 1, "Yes", "No")
    )
  )
  print(tradeoff)

  cat("\nInterpretation:\n")
  cat("  - BYM2 DIC:", round(m4$dic$dic, 2),
      "| SPDE DIC:", round(m_spde$model$dic$dic, 2), "\n")
  cat("  - BYM2 terrain effect: IRR =", round(m4_irr, 2),
      "(", round(m4_irr_lo, 2), "–", round(m4_irr_hi, 2), ")",
      ifelse(m4_irr_lo > 1, "SIGNIFICANT", "not significant"), "\n")
  cat("  - SPDE terrain effect: IRR =", round(m5_irr, 2),
      "(", round(m5_irr_lo, 2), "–", round(m5_irr_hi, 2), ")",
      ifelse(m5_irr_lo > 1, "SIGNIFICANT", "not significant"), "\n")
}
##      Model    DIC  Moran_p IRR_mtns IRR_low IRR_high Significant
## 1 M4: BYM2 304.84 7.46e-07     2.76    1.10     6.98         Yes
## 2 M5: SPDE 301.23 1.00e+00     2.29    0.96     5.48          No
## 
## Interpretation:
##   - BYM2 DIC: 304.84 | SPDE DIC: 301.23 
##   - BYM2 terrain effect: IRR = 2.76 ( 1.1 – 6.98 ) SIGNIFICANT 
##   - SPDE terrain effect: IRR = 2.29 ( 0.96 – 5.48 ) not significant
# -----------------------------------------------------------------------------
# 8. FIXED EFFECTS (IRR)
# -----------------------------------------------------------------------------
cat("\n=== FIXED EFFECTS (IRR) ===\n")
## 
## === FIXED EFFECTS (IRR) ===
fixed <- best_model_obj$summary.fixed
fixed$IRR      <- exp(fixed$mean)
fixed$IRR_low  <- exp(fixed$`0.025quant`)
fixed$IRR_high <- exp(fixed$`0.975quant`)
print(fixed[, c("mean","0.025quant","0.975quant","IRR","IRR_low","IRR_high")])
##               mean  0.025quant 0.975quant      IRR   IRR_low IRR_high
## mtns_any 0.8296362 -0.04091498    1.70066 2.292485 0.9599107  5.47756
cat("\n=== HYPERPARAMETERS ===\n")
## 
## === HYPERPARAMETERS ===
print(best_model_obj$summary.hyperpar)
##                                                                  mean
## size for nbinomial_1 zero-inflated observations          1.922937e+05
## zero-probability parameter for zero-inflated nbinomial_1 5.683160e-02
## Range for spatial                                        1.234497e+01
## Stdev for spatial                                        7.636905e+00
##                                                                    sd
## size for nbinomial_1 zero-inflated observations          6.782770e+06
## zero-probability parameter for zero-inflated nbinomial_1 7.229546e-02
## Range for spatial                                        2.345211e+00
## Stdev for spatial                                        1.448299e+00
##                                                           0.025quant
## size for nbinomial_1 zero-inflated observations          1.899340733
## zero-probability parameter for zero-inflated nbinomial_1 0.001772786
## Range for spatial                                        8.412784355
## Stdev for spatial                                        5.353120008
##                                                              0.5quant
## size for nbinomial_1 zero-inflated observations          136.32411064
## zero-probability parameter for zero-inflated nbinomial_1   0.03089047
## Range for spatial                                         12.11403046
## Stdev for spatial                                          7.45213371
##                                                            0.975quant
## size for nbinomial_1 zero-inflated observations          2.546015e+05
## zero-probability parameter for zero-inflated nbinomial_1 2.668306e-01
## Range for spatial                                        1.761531e+01
## Stdev for spatial                                        1.102167e+01
##                                                                 mode
## size for nbinomial_1 zero-inflated observations           3.30937640
## zero-probability parameter for zero-inflated nbinomial_1  0.00377273
## Range for spatial                                        11.64300517
## Stdev for spatial                                         6.99725036
# -----------------------------------------------------------------------------
# 9. DIAGNOSTICS ON FINAL MODEL
# -----------------------------------------------------------------------------
cat("\n=== DIAGNOSTICS ON FINAL MODEL ===\n")
## 
## === DIAGNOSTICS ON FINAL MODEL ===
resid <- data$att.ful - data$pred_mean
moran_final <- moran.test(resid, nb2listw(nb4, style = "W"), zero.policy = TRUE)
cat("Moran's I:", round(moran_final$estimate[1], 4),
    "| p =", signif(moran_final$p.value, 3), "\n")
## Moran's I: -0.1541 | p = 1
cat("In-sample RMSE:", round(rmse(data$att.ful, data$pred_mean), 3),
    "| MAE:", round(mae(data$att.ful, data$pred_mean), 3), "\n")
## In-sample RMSE: 0.673 | MAE: 0.276
# -----------------------------------------------------------------------------
# 10. FAIR SPATIAL CROSS-VALIDATION
# -----------------------------------------------------------------------------
cat("\n=== FAIR SPATIAL CROSS-VALIDATION (Bayesian) ===\n")
## 
## === FAIR SPATIAL CROSS-VALIDATION (Bayesian) ===
stale <- list.files(pattern = "^cv_combo_.*\\.graph$", full.names = TRUE)
if (length(stale) > 0) file.remove(stale)

set.seed(42)
spatial_folds <- kmeans(coords, centers = 5, nstart = 25)$cluster
data$fold_spatial <- spatial_folds

cv_bayesian <- function(fold_id, data_all) {
  test_idx <- which(data_all$fold_spatial == fold_id)

  combo <- data_all
  combo$att_ful_obs <- combo$att.ful
  combo$att_ful_obs[test_idx] <- NA
  combo$id_cv <- 1:nrow(combo)

  m <- tryCatch({
    inla(att_ful_obs ~ 1 + mtns_any +
           f(id_cv, model = "bym2", graph = "nigeria_k4.graph") +
           offset(log_pop),
         family = "zeroinflatednbinomial1", data = combo,
         control.predictor = list(compute = TRUE, link = 1),
         control.compute = list(dic = FALSE, waic = FALSE, cpo = FALSE))
  }, error = function(e) {
    cat("  Fold", fold_id, "failed:", conditionMessage(e), "\n")
    NULL
  })

  if (is.null(m)) return(c(RMSE = NA, MAE = NA))

  pred_test <- m$summary.fitted.values$mean[test_idx]
  obs_test  <- data_all$att.ful[test_idx]
  c(RMSE = rmse(obs_test, pred_test), MAE = mae(obs_test, pred_test))
}

cv_results_bayes <- data.frame()
for (f in 1:5) {
  cat("Fold", f, "...")
  res <- cv_bayesian(f, data)
  cat(" RMSE =", round(res["RMSE"], 3),
      " MAE =", round(res["MAE"], 3), "\n")
  cv_results_bayes <- rbind(cv_results_bayes,
                            data.frame(fold = f,
                                       RMSE = res["RMSE"],
                                       MAE  = res["MAE"]))
}
## Fold 1 ... RMSE = 1.202  MAE = 0.832 
## Fold 2 ... RMSE = 0.427  MAE = 0.223 
## Fold 3 ... RMSE = 1.628  MAE = 0.492 
## Fold 4 ... RMSE = 0.213  MAE = 0.153 
## Fold 5 ... RMSE = 4.128  MAE = 1.696
cat("\n=== BAYESIAN CV SUMMARY ===\n")
## 
## === BAYESIAN CV SUMMARY ===
cat("Mean RMSE:  ", round(mean(cv_results_bayes$RMSE, na.rm = TRUE), 3),
    "| SD:", round(sd(cv_results_bayes$RMSE, na.rm = TRUE), 3), "\n")
## Mean RMSE:   1.52 | SD: 1.567
cat("Median RMSE:", round(median(cv_results_bayes$RMSE, na.rm = TRUE), 3), "\n")
## Median RMSE: 1.202
cat("Mean MAE:   ", round(mean(cv_results_bayes$MAE, na.rm = TRUE), 3),
    "| SD:", round(sd(cv_results_bayes$MAE, na.rm = TRUE), 3), "\n")
## Mean MAE:    0.679 | SD: 0.628
cat("Median MAE: ", round(median(cv_results_bayes$MAE, na.rm = TRUE), 3), "\n")
## Median MAE:  0.492
hard_fold <- which.max(cv_results_bayes$RMSE)
cat("\nHardest fold:", hard_fold, "\n")
## 
## Hardest fold: 5
cat("Bayesian RMSE excl. fold", hard_fold, ":",
    round(mean(cv_results_bayes$RMSE[-hard_fold], na.rm = TRUE), 3), "\n")
## Bayesian RMSE excl. fold 5 : 0.868
# -----------------------------------------------------------------------------
# 11. XGBOOST BENCHMARK
# -----------------------------------------------------------------------------
cat("\n=== XGBOOST BENCHMARK ===\n")
## 
## === XGBOOST BENCHMARK ===
ml_data <- data[, c("att.ful","mtns_any","x_std","y_std",
                     "log_pop","fold_spatial")]
names(ml_data)[1] <- "y"

xgb_cv_predict <- function(train, test) {
  feat_cols <- c("mtns_any","x_std","y_std","log_pop")
  dtrain <- xgb.DMatrix(data = as.matrix(train[, feat_cols]), label = train$y)
  dtest  <- xgb.DMatrix(data = as.matrix(test[, feat_cols]))
  params <- list(objective = "count:poisson", eta = 0.05,
                 max_depth = 4, subsample = 0.8,
                 eval_metric = "poisson-nloglik")
  model <- xgb.train(params = params, data = dtrain,
                     nrounds = 300, verbose = 0)
  predict(model, dtest)
}

set.seed(42)
cv_results_xgb <- data.frame()
for (f in 1:5) {
  train <- ml_data[ml_data$fold_spatial != f, ]
  test  <- ml_data[ml_data$fold_spatial == f, ]
  pred  <- xgb_cv_predict(train, test)
  cv_results_xgb <- rbind(cv_results_xgb,
                          data.frame(fold = f,
                                     RMSE = rmse(test$y, pred),
                                     MAE  = mae(test$y, pred)))
}

cat("Mean RMSE:  ", round(mean(cv_results_xgb$RMSE), 3),
    "| SD:", round(sd(cv_results_xgb$RMSE), 3), "\n")
## Mean RMSE:   1.381 | SD: 1.743
cat("Median RMSE:", round(median(cv_results_xgb$RMSE), 3), "\n")
## Median RMSE: 0.702
cat("Mean MAE:   ", round(mean(cv_results_xgb$MAE), 3),
    "| SD:", round(sd(cv_results_xgb$MAE), 3), "\n")
## Mean MAE:    0.53 | SD: 0.703
cat("Median MAE: ", round(median(cv_results_xgb$MAE), 3), "\n")
## Median MAE:  0.373
# -----------------------------------------------------------------------------
# 12. HONEST COMPARISON TABLE
# -----------------------------------------------------------------------------
cat("\n=== HONEST ML COMPARISON ===\n")
## 
## === HONEST ML COMPARISON ===
honest_table <- data.frame(
  Model = c("Bayesian ZI Spatial NB", "XGBoost (Poisson)"),
  RMSE_mean = c(mean(cv_results_bayes$RMSE, na.rm = TRUE),
                mean(cv_results_xgb$RMSE)),
  RMSE_median = c(median(cv_results_bayes$RMSE, na.rm = TRUE),
                  median(cv_results_xgb$RMSE)),
  RMSE_sd = c(sd(cv_results_bayes$RMSE, na.rm = TRUE),
              sd(cv_results_xgb$RMSE)),
  MAE_mean = c(mean(cv_results_bayes$MAE, na.rm = TRUE),
               mean(cv_results_xgb$MAE)),
  MAE_median = c(median(cv_results_bayes$MAE, na.rm = TRUE),
                 median(cv_results_xgb$MAE)),
  MAE_sd = c(sd(cv_results_bayes$MAE, na.rm = TRUE),
             sd(cv_results_xgb$MAE)),
  RMSE_excl_hard = c(
    mean(cv_results_bayes$RMSE[-hard_fold], na.rm = TRUE),
    mean(cv_results_xgb$RMSE[-hard_fold], na.rm = TRUE)
  ),
  Interpretable = c("Yes (IRR, phi, CrIs)", "No (SHAP only)")
)
print(honest_table)
##                    Model RMSE_mean RMSE_median  RMSE_sd  MAE_mean MAE_median
## 1 Bayesian ZI Spatial NB  1.519611   1.2023828 1.566650 0.6791653  0.4924422
## 2      XGBoost (Poisson)  1.381418   0.7018133 1.742878 0.5297172  0.3730288
##      MAE_sd RMSE_excl_hard        Interpretable
## 1 0.6278922      0.8675557 Yes (IRR, phi, CrIs)
## 2 0.7026328      0.6540895       No (SHAP only)
cat("\n=== FOLD-WISE COMPARISON ===\n")
## 
## === FOLD-WISE COMPARISON ===
cv_all <- merge(cv_results_bayes[, c("fold","RMSE","MAE")],
                cv_results_xgb[, c("fold","RMSE","MAE")],
                by = "fold", suffixes = c("_bayes", "_xgb"))
print(cv_all)
##   fold RMSE_bayes MAE_bayes   RMSE_xgb    MAE_xgb
## 1    1  1.2023828 0.8324573 0.70181333 0.37302876
## 2    2  0.4265205 0.2227855 0.09300316 0.06434913
## 3    3  1.6284012 0.4924422 1.65959863 0.38651398
## 4    4  0.2129183 0.1525817 0.16194282 0.06963306
## 5    5  4.1278310 1.6955597 4.29073277 1.75506124
# -----------------------------------------------------------------------------
# 13. SAVE (guarded)
# -----------------------------------------------------------------------------
save_safe <- function(obj_name, filename) {
  if (exists(obj_name, envir = environment())) {
    write.csv(get(obj_name, envir = environment()), filename, row.names = FALSE)
    cat("Saved:", filename, "\n")
  } else {
    cat("SKIPPED (not found):", obj_name, "\n")
  }
}

save_safe("diag_table",       "diagnostic_table_all_models.csv")
## Saved: diagnostic_table_all_models.csv
save_safe("bym2_runs",        "bym2_stability_check.csv")
## Saved: bym2_stability_check.csv
save_safe("tradeoff",         "bym2_vs_spde_tradeoff.csv")
## Saved: bym2_vs_spde_tradeoff.csv
save_safe("fixed",            "final_fixed_effects_IRR.csv")
## Saved: final_fixed_effects_IRR.csv
save_safe("cv_results_bayes", "bayesian_spatial_cv.csv")
## Saved: bayesian_spatial_cv.csv
save_safe("cv_results_xgb",   "xgboost_spatial_cv.csv")
## Saved: xgboost_spatial_cv.csv
save_safe("honest_table",     "honest_ml_comparison.csv")
## Saved: honest_ml_comparison.csv
save_safe("cv_all",           "cv_foldwise_comparison.csv")
## Saved: cv_foldwise_comparison.csv
if (exists("data") && "spatial_re" %in% names(data) &&
    !all(is.na(data$spatial_re))) {
  write.csv(data[, c("xcoord","ycoord","att.ful","pred_mean","spatial_re")],
            "final_spatial_predictions.csv", row.names = FALSE)
  cat("Saved: final_spatial_predictions.csv\n")
}

cat("\n===== ANALYSIS COMPLETE =====\n")
## 
## ===== ANALYSIS COMPLETE =====
cat("Final model:", best_name, "\n")
## Final model: M5: SPDE
cat("DIC =", round(best_model_obj$dic$dic, 2), "\n")
## DIC = 301.23
cat("BYM2 stability:", ifelse(bym2_stable, "STABLE", "UNSTABLE"), "\n")
## BYM2 stability: UNSTABLE
