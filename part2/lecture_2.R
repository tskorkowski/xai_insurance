library(tidyverse)
library(splitTools)
library(withr)
library(mgcv)
library(lightgbm)
library(MetricsWeighted)
library(flashlight)
library(gridExtra)
library(OpenML)
library(farff)

# Reload data and model-related objects
main <- "french_motor"
# main <- file.path("r", "french_motor")
reload <- TRUE
refit <- TRUE
setwd("G:/My Drive/aktuariusz/SAV/Responsible Machine Learning with Insurance Applications/Lecture_2")

if (refit) {
  # Download and save dataset
  raw_file <- file.path(main, "raw.rds")
  if (!file.exists(raw_file)) {
    raw <- tibble(getOMLDataSet(data.id = 41214)$data)
    saveRDS(raw, file = raw_file)
  } else {
    raw <- readRDS(raw_file)
  }

  # Preprocessing
  prep <- raw %>% 
    # Identify rows that might belong to the same policy
    group_by_at(
      c("Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", 
        "VehGas", "Density", "Region")
    ) %>% 
    mutate(
      group_id = cur_group_id(),
      group_size = n()
    ) %>% 
    ungroup() %>% 
    arrange(group_id) %>%
    # Usual preprocessing
    mutate(
      Exposure = pmin(1, Exposure),
      Freq = pmin(15, ClaimNb / Exposure),
      VehPower = pmin(12, VehPower),
      VehAge = pmin(20, VehAge),
      VehGas = factor(VehGas),
      DrivAge = pmin(85, DrivAge),
      logDensity = log(Density),
      VehBrand = relevel(fct_lump_n(VehBrand, n = 3), "B12"),
      PolicyRegion = relevel(fct_lump_n(Region, n = 5), "R24"),
      AreaCode = Area
    )
  
  # Covariates, response, weight
  x <- c("VehPower", "VehAge",  "VehBrand", "VehGas", "DrivAge",
         "logDensity", "PolicyRegion")
  y <- "Freq"
  w <- "Exposure"
  save(prep, file = file.path(main, "intro.RData"))
} else {
  load(file.path(main, "intro.RData"))
}  
  
fit_lgb <- lgb.load(file.path(main, "fit_lgb.txt"))

# Prediction functions used by the explainer objects
pred_exp <- function(fit, X) predict(fit, X, type = "response")
pred_lgb <- function(fit, X) predict(fit, data.matrix(X[x]))

fl_glm <- flashlight(model = fit_glm, label = "GLM", predict_function = pred_exp)
fl_gam <- flashlight(model = fit_gam, label = "GAM", predict_function = pred_exp)
fl_lgb <- flashlight(model = fit_lgb, label = "LGB", predict_function = pred_lgb)

# Combine them and add common elements like exposure weights and some metrics
metrics <- list(
  `Average deviance` = deviance_poisson, 
  `Relative deviance reduction` = r_squared_poisson
)

fls <- multiflashlight(
  list(fl_glm, fl_gam, fl_lgb), data = test, y = y, w = w, metrics = metrics
)

# Version on the link scale (ask the author of flashlight 
# why the option is called "linkinv" rather than "transformation"...)
fls_log <- multiflashlight(fls, linkinv = log)