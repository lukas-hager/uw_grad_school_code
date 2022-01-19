rm(list = ls())

gc()
gc()

library(data.table)
library(R.matlab)
library(tidyverse)

# source functions

dir_path <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dir_path)
source('functions.R')

# read in the airline data

file_path <- '/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/ECON 532/Assignments/Assignment 1/pset1_upload/IV.mat'
data_list <- R.matlab::readMat(file_path) 
list2env(data_list, envir = .GlobalEnv)

# write g function

g_i <- function(b_val, X_val, Y_val, Z_val){
  # get the g_is
  g_i <- t(Z_val) %*% (Y_val - X_val %*% b_val)
  return(g_i)
}

# write objective function

gmm_crit <- function(b_val,X_val,Y_val,Z_val,W_val){
  g_n_val <- g_i(b_val, X_val, Y_val, Z_val) / dim(X_val)[1]
  return(t(g_n_val) %*% W_val %*% g_n_val)
}

# write variance/covariance function

v_cov <- function(b_val,X_val,Y_val,Z_val ,W_val){
  n <- dim(X_val)[1]
  g_i_mat <- Z_val * as.numeric(Y_val - X_val %*% b_val)
  G <- -t(Z_val) %*% X_val / n
  omega <- t(g_i_mat) %*% g_i_mat / n
  inv_term <- solve(t(G) %*% W_val %*% G)
  return(inv_term %*% t(G) %*% W_val %*% omega %*% t(W_val) %*% G %*% inv_term)
}

# write SE function

get_se <- function(v_cov, n){
  return(sqrt(diag(v_cov / n)))
}

# first step estimation

W_1 <- diag(dim(Z)[2])

optimization_1 <- optim(rep(0,dim(X)[2]), 
                        gmm_crit, 
                        X_val = X, 
                        Y_val = Y, 
                        Z_val = Z,
                        W_val = W_1,
                        control = list(maxit = 1000000))
beta_hat_optim_1 <- optimization_1$par
v_cov_1 <- v_cov(beta_hat_optim_1,X,Y,Z,W_1)

# print results
cat(str_interp('Step 1 Algorithm ${ifelse(optimization_1$convergence == 0, "Converged", "Did Not Converge")}\n'))
cat('Step 1 Beta:\n')
create_latex_matrix(beta_hat_optim_1, 5)
cat('Step 1 Variance-Covariance Matrix:\n')
create_latex_matrix(v_cov_1, 5)
cat('Step 1 SEs:\n')
create_latex_matrix(get_se(v_cov_1, dim(X)[1]), 5)

# create the new weighting matrix

resids_sq <- (Y - X %*% beta_hat_optim_1)^2
W_2_inv <- t(Z) %*% (Z * as.numeric(resids_sq)) / dim(Z)[1]
W_2 <- solve(W_2_inv)

# second step optimization

optimization_2 <- optim(beta_hat_optim_1, 
                        gmm_crit, 
                        X_val = X, 
                        Y_val = Y, 
                        Z_val = Z,
                        W_val = W_2,
                        control = list(maxit = 1000000))
beta_hat_optim_2 <- optimization_2$par
v_cov_2 <- v_cov(beta_hat_optim_2,X,Y,Z,W_2)

# print results
cat(str_interp('Step 2 Algorithm ${ifelse(optimization_2$convergence == 0, "Converged", "Did Not Converge")}\n'))
cat('Step 2 Beta:\n')
create_latex_matrix(beta_hat_optim_2, 5)
cat('Step 2 Variance-Covariance Matrix:\n')
create_latex_matrix(v_cov_2, 5)
cat('Step 2 SEs:\n')
create_latex_matrix(get_se(v_cov_2, dim(X)[1]), 5)

# # get g
# 
# G <- t(Z) %*% X / dim(X)[1]
# 
# # get omega
# 
# g_i_mat <- Z * as.numeric(Y - X %*% beta_hat_optim_2)
# 
# omega <- t(g_i_mat) %*% g_i_mat / dim(X)[1]
# 
# # get v-cov

# library(gmm)
# 
# g <- function(b_val, data){
#   Y <- data[,1]
#   X <- data[,2:5]
#   Z <- data[,6:10]
#   
#   return(t(Z) %*% (Y - X %*% b_val))
# }
# 
# gmm(g = g, 
#          x = do.call(cbind, list(Y,X,Z)),
#          t0 = rep(0,3),
#          type = 'twostep')
