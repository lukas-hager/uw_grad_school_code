rm(list = ls())

gc()
gc()

library(data.table)
library(broom)
library(knitr)
library(sandwich)
library(MASS)
library(tidyverse)

set.seed(2021)

#########################################################
# Exercise 9.26
#########################################################

# read in the data

link <- 'https://www.ssc.wisc.edu/~bhansen/econometrics/Nerlove1963.txt'

data_og <- fread(link) %>% 
  rename_all(tolower)

# add features

data_reg <- data_og %>% 
  mutate_all(list('log' = log))

# run OLS

basic_reg <- lm(cost_log ~ output_log + plabor_log + pcapital_log + pfuel_log,
                data = data_reg)

# get robust SEs

V_HC1 <- vcovHC(basic_reg, 
                type = "HC1")

# output results

basic_reg_results <- basic_reg %>% 
  coeftest(., vcov = V_HC1) %>% 
  tidy()

basic_reg_results %>%
  kable(., 
        format = 'latex',
        digits = 4,
        align = 'lrrrr',
        caption = 'Ordinary Least Squares')

# define relevant vectors and matrices

beta_ols <- basic_reg_results$estimate
R <- c(0, 0, 1, 1, 1)
c <- 1
X <- data_reg %>% 
  select(ends_with('_log')) %>% 
  select(-cost_log) %>% 
  mutate(constant = 1) %>% 
  select(constant, everything()) %>% 
  as.matrix()

Y <- data_reg %>% 
  select(cost_log) %>% 
  as.matrix()

Xe_ols <- X * rep(basic_reg$residuals, times = k)

# create Q_XX inverted
Q <- ginv(crossprod(X))

# calculate CLS estimate

inv <- ginv(t(R) %*% Q %*% R)
constraint <- (t(R) %*% beta_ols -c)
beta_cls <- beta_ols - Q %*% R %*% inv %*% constraint %>% 
  as.vector()

# get cls residuals

cls_resid <- Y - X %*% beta_cls %>% 
  as.vector()

# calculate variance-covariance matrix

n <- dim(X)[1]
k <- dim(X)[2]
q <- length(c)

Xe_cls <- X * rep(cls_resid, times = k)
V_cls_cond <- (n/(n-k+1)) * Q %*% (t(Xe_cls) %*% Xe_cls) %*% Q
ugly_term <- Q %*% R %*% ginv(t(R) %*% Q %*% R) %*% t(R)
cov_term <- (ugly_term %*% V_cls_cond) + (V_cls_cond %*% t(ugly_term))

V_cls <- V_cls_cond - cov_term + (ugly_term %*% V_cls_cond %*% t(ugly_term))

se_cls <- sqrt(diag(V_cls))

# report coefficients and SEs

data.table('term' = basic_reg_results$term,
           'estimate' = beta_cls,
           'std.error' = se_cls) %>% 
  kable(.,
        format = 'latex',
        digits = 4,
        align = 'lrr',
        caption = 'Constrained Least Squares')

# we now compute efficient minimum distance

inv <- ginv(t(R) %*% V_HC1 %*% R)
beta_emd <- beta_ols - V_HC1 %*% R %*% inv %*% constraint %>% 
  as.vector()

V_term <- V_HC1 %*% R %*% ginv(t(R) %*% V_HC1 %*% R) %*% t(R)
beta_mle <- beta_ols - V_term %*% beta_ols
Xe_emd <- X * rep((Y - X %*% beta_emd), times=k)
V2 <- (n/(n-k+1)) * Q %*% (t(Xe_emd) %*% Xe_emd) %*% Q
V_emd <- V2 - V2 %*% R %*% ginv(t(R) %*% V2 %*% R) %*% t(R) %*% V2 

se_emd <- sqrt(diag(V_emd))

# report coefficients and SEs

data.table('term' = basic_reg_results$term,
           'estimate' = beta_emd,
           'std.error' = se_emd) %>% 
  kable(.,
        format = 'latex',
        digits = 4,
        align = 'lrr',
        caption = 'Minimum Distance Estimation')

# wald test

w <- t(constraint) %*% ginv(t(R) %*% V_HC1 %*% R) %*% constraint %>% 
  as.vector()

p_wald <- round(1 - pchisq(w, df = q), 2)
wald_text <- 'The p-value of the Wald test is ${p_wald}'
str_interp(wald_text)

# emd test

j <- t(beta_ols - beta_emd) %*% ginv(V_HC1) %*% (beta_ols - beta_emd)

p_emd <- round(1 - pchisq(j, df = q), 2)
emd_text <- 'The p-value of the Wald test is ${p_emd}'
str_interp(emd_text)

#########################################################
# Exercise 10.28
#########################################################

# construct jackknife estimator for ols

data_id <- data_reg %>% 
  mutate(id = row_number())

beta_jk <- map_dfr(c(1:nrow(data_id)), function(x){
  reg_jk <- lm(cost_log ~ output_log + plabor_log + 
                 pcapital_log + pfuel_log,
               data = data_id %>% 
                 filter(id != x))
  return(reg_jk$coefficients)
}) %>% 
  rename_all(list(~ tolower(str_replace_all(., 
                                            '\\(|\\)', 
                                            '')))) %>% 
  mutate(id = row_number())

means_jk <- beta_jk %>% 
  select(-id) %>% 
  summarise_all(list('mean' = mean)) %>% 
  pivot_longer(everything()) %>% 
  pull(value)

mats <- map(c(1:nrow(data_id)), function(x){
  df <- beta_jk %>% 
    filter(id == x) %>% 
    select(-id) %>% 
    pivot_longer(everything()) %>% 
    pull(value)
  
  return((df - means_jk) %*% t(df - means_jk))
  
})

var_jk <- ((n-1)/n) * Reduce('+', mats)
se_jk <- sqrt(diag(var_jk))

# report results

data.table('term' = basic_reg_results$term,
           'estimate' = beta_ols,
           'std.error' = se_jk) %>% 
  kable(.,
        format = 'latex',
        digits = 4,
        align = 'lrr',
        caption = 'OLS with Jackknife SEs')

# construct bootstrap estimator

B <- 1000

beta_boot <- map_dfr(c(1:B), function(x){
  reg_jk <- lm(cost_log ~ output_log + plabor_log + 
                 pcapital_log + pfuel_log,
               data = data_id %>% 
                 dplyr::sample_n(size = nrow(data_id),
                                 replace = TRUE))
  return(reg_jk$coefficients)
}) %>% 
  rename_all(list(~ tolower(str_replace_all(., 
                                            '\\(|\\)', 
                                            '')))) %>% 
  mutate(id = row_number())

means_boot <- beta_boot %>% 
  select(-id) %>% 
  summarise_all(list('mean' = mean)) %>% 
  pivot_longer(everything()) %>% 
  pull(value)

mats <- map(c(1:B), function(x){
  df <- beta_boot %>% 
    filter(id == x) %>% 
    select(-id) %>% 
    pivot_longer(everything()) %>% 
    pull(value)
  
  return((df - means_boot) %*% t(df - means_boot))
  
})

var_boot <- (1/(B-1)) * Reduce('+', mats)
se_boot <- sqrt(diag(var_boot))

# report results

data.table('Term' = basic_reg_results$term,
           'Estimate' = beta_ols,
           'SE Asymptotic' = basic_reg_results$std.error,
           'SE Jackknife' = se_jk,
           'SE Bootstrap' = se_boot) %>% 
  kable(.,
        format = 'latex',
        digits = 4,
        align = 'lrr',
        caption = 'OLS with Bootstrap SEs')

# estimate theta and SEs

theta <- basic_reg_results %>% 
  filter(term %in% c('plabor_log', 'pcapital_log', 'pfuel_log')) %>% 
  pull(estimate) %>% 
  sum()

get_theta_var <- function(vcov_mat){
  return(sqrt(sum(vcov_mat[3:5,3:5])))
}

data.table('Theta' = theta,
           'SE Asymptotic' = get_theta_var(V_HC1),
           'SE Jackknife' = get_theta_var(var_jk),
           'SE Bootstrap' = get_theta_var(var_boot)) %>% 
  kable(.,
        format = 'latex',
        digits = 4,
        align = 'lrrr',
        caption = 'Estimate of Theta in MRW Model')

# bootstrapped CIs using percentile method

theta_boot <- beta_boot %>% 
  mutate(theta = plabor_log + pcapital_log + pfuel_log) %>% 
  pull()

quantile(theta_boot, c(.025, .975)) %>% 
  t() %>% 
  kable(.,
        format = 'latex',
        digits = 4,
        align = 'rr',
        caption = 'Bootstrapped CIs Using Percentile Method')

# bootstrapped CIs using BC alpha

# define parameters z_0 and a

theta_jk <- beta_jk %>% 
  mutate(theta = plabor_log + pcapital_log + pfuel_log) %>% 
  pull()

p_star <- sum(theta_boot >= theta) / length(theta_boot)
z_0 <- qnorm(p_star)

theta_bar <- mean(theta_jk)
a <- sum((theta_bar - theta_jk)^3) / (6 * (sum((theta_bar - theta_jk)^2)^(3/2)))

x_alpha <- function(alpha){
  z_a <- qnorm(alpha)
  return(pnorm(z_0 + (z_a + z_0) / (1 + a * (z_a + z_0))))
}

quantile(theta_boot, c(x_alpha(.025), 
                       x_alpha(.975))) %>% 
  t() %>% 
  kable(.,
        col.names = c('2.5%', '97.5%'),
        format = 'latex',
        digits = 4,
        align = 'rr',
        caption = 'Bootstrapped CIs Using BC Alpha Method')