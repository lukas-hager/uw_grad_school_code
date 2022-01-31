rm(list = ls())

gc()
gc()

library(data.table)
library(tidyverse)

# source functions

dir_path <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dir_path)
source('functions.R')

# read in the airline data

file_path <- '/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/ECON 532/Assignments/Assignment 1/pset1_upload/airline.txt'

data_og <- fread(file_path) %>% 
  rename_all(tolower)

# one-hot encode the day of week

data_one_hot <- data_og %>% 
  pivot_longer(cols = day_of_week) %>% 
  arrange(value) %>% 
  mutate(name = str_c(name, value, sep = '_'),
         value = 1,
         row_num = row_number()) %>% 
  pivot_wider(id_cols = c(row_num, arr_delay, dep_delay, distance)) %>% 
  select(-row_num, -day_of_week_1) %>% 
  mutate_at(vars(day_of_week_2:day_of_week_7), list(~replace_na(., 0))) %>% 
  mutate(cons = 1) %>% 
  select(arr_delay, cons, distance, dep_delay, everything())

# define variables

y <- data_one_hot %>% 
  pull(arr_delay)

X <- data_one_hot %>% 
  select(-arr_delay) %>% 
  as.matrix()

# get OLS coefficients with formula

beta_hat_form <- solve(t(X) %*% X) %*% t(X) %*% y

cat('Formula beta:\n')
create_latex_matrix(beta_hat_form, 5)

# get OLS coefficients with base R method to check

data_base_r <- data_og %>% 
  mutate(day_of_week = factor(day_of_week))

ols_reg <- lm(arr_delay ~ distance + dep_delay + day_of_week, 
              data = data_base_r)

# write a function to return squared error

get_squared_error <- function(b_hat,X_val,y_val){
  if (length(b_hat) != dim(X_val)[2] | length(y_val) != dim(X_val)[1]){
    stop('Dimension mismatch.')
  }
  
  y_hat <- X_val %*% b_hat
  error <- y_val - y_hat
  return(sum(error^2))
}

# optimize to get the minimum

optimization <- optim(rep(0,dim(X)[2]), get_squared_error, X_val = X, y_val = y, 
                      control = list(maxit = 1000000,
                                     reltol = 1e-30))
beta_hat_optim <- optimization$par
names(beta_hat_optim) <- dimnames(X)[[2]]

cat('Optimized beta:\n')
create_latex_matrix(beta_hat_optim, 5)

# find the differences

diff <- beta_hat_optim - beta_hat_form
diff_perc <- (get_squared_error(beta_hat_optim,X,y) - get_squared_error(beta_hat_form,X,y)) / get_squared_error(beta_hat_form,X,y)

cat('Difference:\n')
create_latex_matrix(diff, 5)

cat('Criterion difference:\n')
create_latex_matrix(diff_perc, 5)