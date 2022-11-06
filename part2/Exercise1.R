library(tidyverse)
library(insuranceData)
library(gridExtra)
library(splitTools)
library(ranger)
library(MetricsWeighted)

data(dataCar)
head(dataCar)

fillc <- "#E69F00"
refit <- TRUE

# Data preparation
prep <- dataCar %>% 
  mutate(
    Id = 1:count(dataCar)[[1]],
    Freq = numclaims / exposure,
    Sev = claimcst0 / numclaims,
    veh_age = factor(veh_age),
    agecat = factor(agecat),
    veh_body = fct_lump_prop(veh_body, 0.1),
    log_value = log(pmax(0.3, pmin(15, veh_value))),
    pure_prem = Sev / exposure
  )

# Variable groups
y_var <- "Freq"
w_var <- "exposure"
x_vars <- c("log_value", "agecat", "veh_age", "area", "veh_body", "gender")

# XGBoost parameters for the last exercise
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

??dataCar
summary(prep)

# Histograms
prep %>%
  select_at(c("veh_value", "exposure", "claimcst0", "Freq", "Sev")) %>% 
  pivot_longer(everything()) %>% 
  ggplot(aes(x = value)) +
  geom_histogram(bins = 19, fill = fillc) +
  facet_wrap(~ name, scales = "free") +
  ggtitle("Histograms of numeric columns")

#Bar charts for discrete fetures
c("agecat", "veh_age", "area", "veh_body", "gender") %>% 
  lapply(function(v) ggplot(prep, aes_string(v)) + geom_bar(fill = fillc)) %>% 
  grid.arrange(grobs = ., top = "Barplots of discrete columns")

##Split set into test and validation sets
set.seed(22)  # Seeds the data split as well as the boosted trees model

#Exercise2

ind <- partition(prep[["Id"]], p = c(train = 0.8, test = 0.2), type = "grouped")
train <- prep[ind$train, ]
test <- prep[ind$test, ]

# 1) GLM
if (refit) {
  fit_glm <- glm(
    reformulate(x_vars, y_var), data = train, family = quasipoisson(), weights = train[[w_var]]
  )
}
summary(fit_glm)

#interpretation  of the features “log_value” and “gender”
# change of one point increase in log_value is associated with change in log expected claims frequency of:
fit_glm$coefficients['log_value']
# or on frequency scale [%]
(exp(fit_glm$coefficients['log_value'])-1) * 100

#on the other hand in comparison to female drivers on the log expected claims frequency male drivers:
fit_glm$coefficients['genderM']
# or on frequency scale [%]
(exp(fit_glm$coefficients['genderM'])-1) * 100

#2) Random forest
poisson_deviance = deviance_poisson(train[[y_var]],fit_glm$fitted.values, train[[w_var]])

max(fit_glm$fitted.values)

#Select the optimal tree depth by minimizing exposure-weighted Poisson deviance calculated from the out-of-bag (OOB) predictions on the training data.
#what it means that we change depth parameter of the trees and select depth that corresponds to the lowest observed poisson deviance

#the dataset is quite poor, predictions are not too good even when compared to intercept model only

# Save everything important
if (refit) {
  save(x_vars, y_var, w_var, fit_glm, prep, train, test, fillc, 
       file = file.path(main, "Exercises.RData"))
}