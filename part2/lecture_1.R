library(tidyverse)
library(gridExtra)
library(OpenML)
library(farff)
library(splitTools)
library(mgcv)
library(lightgbm)


main <- "french_motor"
# main <- "r/french_motor"
fillc <- "#E69F00"
refit <- FALSE

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
} else {
  load(file.path(main, "intro.RData"))
}

# Description
nrow(prep)
head(prep[, c("IDpol", "group_id", "ClaimNb", x, w, y)], 8)
summary(prep[, c(x, w, y)])

# Histograms
prep %>%
  select_at(c("Freq", "Exposure", "DrivAge", "VehAge", "VehPower", "logDensity")) %>% 
  pivot_longer(everything()) %>% 
  ggplot(aes(x = value)) +
  geom_histogram(bins = 19, fill = fillc) +
  facet_wrap(~ name, scales = "free") +
  ggtitle("Histograms of numeric columns")

#Bar charts for discrete fetures
c("ClaimNb", "VehBrand", "VehGas", "PolicyRegion") %>% 
  lapply(function(v) ggplot(prep, aes_string(v)) + geom_bar(fill = fillc)) %>% 
  grid.arrange(grobs = ., top = "Barplots of discrete columns")

#Modeling
set.seed(22)  # Seeds the data split as well as the boosted trees model

ind <- partition(prep[["group_id"]], p = c(train = 0.8, test = 0.2), type = "grouped")
train <- prep[ind$train, ]
test <- prep[ind$test, ]

# 1) GLM
if (refit) {
  fit_glm <- glm(
    reformulate(x, y), data = train, family = quasipoisson(), weights = train[[w]]
  )
}
summary(fit_glm)

# 2) GAM with penalized regression smooths and maximum k-1 df. Takes long to fit
if (refit) {
  fit_gam <- gam(
    Freq ~ s(VehAge, k = 7) + s(DrivAge, k = 7) + s(logDensity, k = 3) + 
      s(VehPower, k = 3) + PolicyRegion + VehBrand + VehGas, 
    data = train, 
    family = quasipoisson(), 
    weights = train[[w]]
  )
}
summary(fit_gam)

# Visualizing the effect of an additive component on the scale of the linear predictor
plot(fit_gam, select = 2, main = "Additive effect of 'DrivAge' on log scale")

# 3) Boosted trees
if (refit) {
  # Data interface of LightGBM (LGM)
  dtrain <- lgb.Dataset(
    data.matrix(train[x]), 
    label = train[[y]], 
    weight = train[[w]], 
    params = list(feature_pre_filter = FALSE)
  )
  
  # Parameters found by CV randomized search, see later
  params <- list(
    learning_rate = 0.05, 
    objective = "poisson", 
    metric = "poisson", 
    num_leaves = 63, 
    min_data_in_leaf = 100, 
    min_sum_hessian_in_leaf = 0, 
    colsample_bynode = 0.8, 
    bagging_fraction = 0.8, 
    lambda_l1 = 4, 
    lambda_l2 = 0, 
    num_threads = 7 # adapt
  )
  
  # Fit model ("nrounds" is an optimized parameter as well, see later)
  fit_lgb <- lgb.train(params = params, data = dtrain, nrounds = 174)  
  lgb.save(fit_lgb, file.path(main, "fit_lgb.txt"))
} else{
  fit_lgb <- lgb.load(file.path(main, "fit_lgb.txt"))
}

# Save everything important
if (refit) {
  save(x, y, w, fit_glm, fit_gam, prep, train, test, fillc, 
       file = file.path(main, "intro.RData"))
}