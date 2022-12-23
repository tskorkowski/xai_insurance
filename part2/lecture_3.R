library(OpenML)
library(farff)
library(tidyverse)
library(lubridate)
library(withr)
library(xgboost)
library(MetricsWeighted)
library(flashlight)
library(shapviz)
library(keras)
# install_keras() # if no Python with TensorFlow is installed yet

#===============================
# Download and save dataset
#===============================

main <- file.path("r", "workers_compensation")

raw_file <- file.path(main, "raw.rds")
if (!file.exists(raw_file)) {
  if (!dir.exists(main)) {
    dir.create(main)
  }
  raw <- tibble(getOMLDataSet(data.id = 42876)$data)
  saveRDS(raw, file = raw_file)
} else {
  raw <- readRDS(raw_file)
}

#===============================
# Preprocessing
#===============================

# Clip small and/or large values
clip <- function(x, low = -Inf, high = Inf) {
  pmax(pmin(x, high), low)  
}

prep <- raw %>% 
  filter(WeeklyPay >= 200, HoursWorkedPerWeek >= 20) %>% 
  mutate(
    Ultimate = clip(UltimateIncurredClaimCost, high = 1e6),
    LogInitial = log(clip(InitialCaseEstimate, 1e2, 1e5)),
    LogWeeklyPay = log(clip(WeeklyPay, 100, 2000)),
    LogAge = log(clip(Age, 17, 70)),
    Female = (Gender == "F") * 1L,
    Married = (MaritalStatus == "M") * 1L,
    PartTime = (PartTimeFullTime == "P") * 1L,
    
    DateTimeOfAccident = as_datetime(DateTimeOfAccident),
    
    LogDelay = log1p(as.numeric(ymd(DateReported) - as_date(DateTimeOfAccident))),
    DateNum = decimal_date(DateTimeOfAccident),
    WeekDay = wday(DateTimeOfAccident, week_start = 1),   # 1 = Monday
    Hour = hour(DateTimeOfAccident)
  )

# Lost claim amount -> in practice, use as correction or "smearing" factor
(smearing_factor <- 1 - with(prep, sum(Ultimate) / sum(UltimateIncurredClaimCost)))

#===============================
# Variable groups
#===============================

y_var <- "Ultimate"
x_continuous <- c("LogInitial", "LogWeeklyPay", "LogDelay", "LogAge", "DateNum")
x_discrete <- c("PartTime", "Female", "Married", "WeekDay", "Hour")
x_vars <- c(x_continuous, x_discrete)

#===============================
# Univariate analysis
#===============================

# Some rows
prep %>%
  select(all_of(c(y_var, x_vars))) %>%
  head()

# Histogram of the response on log scale
ggplot(prep, aes(.data[[y_var]])) +
  geom_histogram(bins = 19, fill = "darkred") +
  scale_x_log10()

# Histograms of continuous predictors
prep %>% 
  select(all_of(x_continuous)) %>% 
  pivot_longer(everything()) %>% 
  ggplot(aes(value)) +
  geom_histogram(bins = 19, fill = "darkred") +
  facet_wrap(~ name, scales = "free", ncol = 2)

# Barplots of discrete predictors
prep %>% 
  select(all_of(x_discrete)) %>% 
  pivot_longer(everything()) %>% 
  ggplot(aes(factor(value))) +
  geom_bar(fill = "darkred") +
  facet_wrap(~ name, scales = "free", ncol = 2)

#===============================
# Data splits for models
#===============================

with_seed(656, 
          .in <- sample(nrow(prep), 0.8 * nrow(prep), replace = FALSE)
)

train <- prep[.in, ]
test <- prep[-.in, ]
y_train <- train[[y_var]]
y_test <- test[[y_var]]
X_train <- prep[.in, x_vars]
X_test <- prep[-.in, x_vars]

#===============================
# XGBoost: Additive model 
# (Exercise 3)
#===============================

nrounds <- 484

params <- list(
  learning_rate = 0.1, 
  objective = "reg:gamma", 
  max_depth = 1, 
  colsample_bynode = 1, 
  subsample = 1, 
  reg_alpha = 3, 
  reg_lambda = 1, 
  tree_method = "hist", 
  min_split_loss = 0.001, 
  nthread = 7 # adapt
)

#===============================
# XGBoost: Partly additive model
# # (Exercise 4)
#===============================

# Build interaction constraint vector
additive <- c("Female", "DateNum")
ic <- c(
  list(which(!(x_vars %in% additive)) - 1),
  as.list(which(x_vars %in% additive) - 1)
)
ic

nrounds2 <- 202

params2 <- list(
  learning_rate = 0.1, 
  objective = "reg:gamma", 
  max_depth = 2, 
  colsample_bynode = 0.8, 
  subsample = 1, 
  reg_alpha = 0, 
  reg_lambda = 2, 
  min_split_loss = 0, 
  tree_method = "hist", 
  nthread = 7, 
  interaction_constraints = ic
)

#null deviance is the model with intercept only
# deviance improvement
# drop one test with F test to see how the model changs with removing one variable

#parwise interactions are achivable with one hidden layer (normal order), for higher order we'd need more hidden layers
#for trees we'd create two layer trees (depth 2) x1 at the root, x2 at the level 2
#boosted tree model with additive - same as above but you specify interactions constrains where you have as separate subsets for variabels that are supposed to be additive (no interactions).
#We stay with detph 2, as we'd like to have pariwise interactions only

#the most important part of this lecture is to be able to interpret results and have strategies to improve model GLM or other :)

#Missing two weeks of Christinas lectures plus exercises

#u can always look at machine learning as an source of inspiration to imporve GLM

#whenever you do tree do the shap analysis


#vertical scatter indicates that model is (not?) additive


#Partial additive model XGBoost why does that make sense?
  #sometimes you need to be able to desribe the model fully, it's useful when wanting to incorporate trends
  #there are only very small interactoins
  # vertical spllit means there are interactions


#it's important to know how nn cen be structured, by drawing structure and what kind of effect we'd observe based on architecture
#conceptual things we need to know and how to interpret the model output - not necessarily how to code the exercises in
#study exercises