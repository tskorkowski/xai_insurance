library(withr)
library(tidyverse)
library(insuranceData)
library(ranger)
library(xgboost)
library(MetricsWeighted)
library(flashlight)
library(shapviz)

data(dataCar)

# Data preparation
prep <- dataCar %>% 
  mutate(
    Freq = numclaims / exposure,
    Sev = claimcst0 / numclaims,
    veh_age = factor(veh_age),
    agecat = factor(agecat),
    veh_body = fct_lump_prop(veh_body, 0.1),
    log_value = log(pmax(0.3, pmin(15, veh_value)))
  )

# Modeling
y_var <- "Freq"
w_var <- "exposure"
x_vars <- c("log_value", "agecat", "veh_age", "area", "veh_body", "gender")

with_seed(304, 
          ix <- sample(nrow(prep), 0.8 * nrow(prep))
)
train <- prep[ix, ]
test <- prep[-ix, ]

form <- reformulate(x_vars, y_var)
fit_glm <- glm(form, data = train, weights = train[[w_var]], family = quasipoisson())
fit_rf <- ranger(
  form, data = train, case.weights = train[[w_var]], max.depth = 4, seed = 400
)

# XGBoost
dtrain <- xgb.DMatrix(
  data.matrix(train[, x_vars]), 
  label = train[[y_var]], 
  weight = train[[w_var]]
)

# Hyperparameters
params <- list(
  learning_rate = 0.05,
  objective = "count:poisson",
  max_depth = 2,
  colsample_bynode = 0.8,
  subsample = 0.8,
  reg_alpha = 1,
  reg_lambda = 0,
  min_split_loss = 0.001
)

with_seed(345, 
          fit_xgb <- xgb.train(params = params, data = dtrain, nrounds = 196)
)


#imporving GLM
#no very strong interactions, mainly log+value and agecat
# we add natural spline to the GLM with the 

#in case of improving GLM we'd need to provide two strategies to imporve the model
    #one way is have another model (trees of network) and use shap values
      #subsample training set 
      #calc variable importance
      #
    #use basic techniques if we are missing any effect of structure in the  