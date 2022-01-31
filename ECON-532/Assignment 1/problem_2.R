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

data_binary <- data_og %>% 
  mutate(late = as.numeric(arr_delay > 15),
         cons = 1)

# define variables

y <- data_binary %>% 
  pull(late)

X <- data_binary %>% 
  select(cons, distance, dep_delay) %>% 
  as.matrix()

# write a function to return simplified log likelihood, which is
# more numerically stable, apparently

get_ll <- function(b_hat,X_val,y_val){
  if (length(b_hat) != dim(X_val)[2] | length(y_val) != dim(X_val)[1]){
    stop('Dimension mismatch.')
  }
  
  prod <- X_val %*% b_hat
  exp_term <- exp(prod)
  return(-1 * sum(y_val * prod - log(1 + exp_term)))
}

# run a regression to get starting coefficients

lm_coef <- lm(late ~ distance + dep_delay, data = data_binary)$coefficients

# optimize to get the minimum

optimization <- optim(lm_coef, get_ll, X_val = X, y_val = y, control = list(maxit = 1000000))
beta_hat_optim <- optimization$par
names(beta_hat_optim) <- dimnames(X)[[2]]

# print results

cat(str_interp('Algorithm ${ifelse(optimization$convergence == 0, "Converged", "Did Not Converge")}\n'))
cat('Variables:\n')
cat(str_c(str_c(names(beta_hat_optim), collapse = '\n'), '\n'))
cat('Point Estimates:\n')
create_latex_matrix(beta_hat_optim, 5)

# get final df

final_df <- data_binary %>% 
  mutate(prob_1 = exp(X %*% beta_hat_optim) / (1 + exp(X %*% beta_hat_optim)),
         prob_0 = 1-prob_1)

# compare to base r formulation

model <- glm(late ~ distance + dep_delay, 
             data = data_binary, 
             family=binomial(link='logit'))
