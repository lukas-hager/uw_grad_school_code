library(data.table)
library(broom)
library(knitr)
library(sandwich)
library(MASS)
library(tidyverse)

set.seed(2021)

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

# output results

basic_reg_results <- tidy(basic_reg)

basic_reg_results %>%
  kable(., 
        format = 'latex',
        digits = 2,
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

Xe_ols <- X * rep(basic_reg$residuals, times = k)

# create Q_XX inverted
Q <- ginv(crossprod(X))

# calculate CLS estimate

inv <- ginv(t(R) %*% Q %*% R)
constraint <- (t(R) %*% beta_ols -c)
beta_cls <- beta_ols - Q %*% R %*% inv %*% constraint %>% 
  as.vector()

# get cls residuals

cls_resid <- data_reg$cost_log - X %*% beta_cls %>% 
  as.vector()

# calculate variance-covariance matrix

n <- dim(X)[1]
k <- dim(X)[2]
q <- length(c)

Xe_cls <- X * rep(cls_resid, times = k)
V_cls_cond <- (n/(n-k+q)) * Q %*% (t(Xe_cls) %*% Xe_cls) %*% Q
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
        digits = 2,
        align = 'lrr',
        caption = 'Constrained Least Squares')

# for minimum distance estimation, we need V_beta

resid_ols <- basic_reg$residuals

V_ols <- (n/(n-k)) * Q %*% (t(Xe_ols) %*% Xe_ols) %*% Q

# plug this into the MDE formulas

inv <- ginv(t(R) %*% V_beta %*% R)
beta_mle <- beta_ols - V_beta %*% R %*% inv %*% constraint %>% 
  as.vector()

V_mle <- V_beta - V_beta %*% R %*% inv %*% t(R) %*% V_beta

se_mle <- sqrt(diag(V_mle))

# report coefficients and SEs

data.table('term' = basic_reg_results$term,
           'estimate' = beta_mle,
           'std.error' = se_mle) %>% 
  kable(.,
        format = 'latex',
        digits = 2,
        align = 'lrr',
        caption = 'Minimum Distance Estimation')

# wald test, assuming conditional homoskedasticity again

w <- t(constraint) %*% ginv(t(R) %*% V_beta %*% R) %*% constraint %>% 
  as.vector()

p_wald <- round(1 - pchisq(w, df = 1), 2)
wald_text <- 'The p-value of the Wald test is ${p_wald}'
str_interp(wald_text)

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
        digits = 2,
        align = 'lrr',
        caption = 'OLS with Jackknife SEs')

# construct bootstrap estimator

beta_boot <- map_dfr(c(1:nrow(data_id)), function(x){
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

mats <- map(c(1:nrow(data_id)), function(x){
  df <- beta_boot %>% 
    filter(id == x) %>% 
    select(-id) %>% 
    pivot_longer(everything()) %>% 
    pull(value)
  
  return((df - means_boot) %*% t(df - means_boot))
  
})

var_jk <- (1/(n-1)) * Reduce('+', mats)
se_jk <- sqrt(diag(var_jk))

# report results

data.table('term' = basic_reg_results$term,
           'estimate' = beta_ols,
           'std.error' = se_jk) %>% 
  kable(.,
        format = 'latex',
        digits = 2,
        align = 'lrr',
        caption = 'OLS with Bootstrap SEs')

###################
# Exercise 9.27
###################

# read in data

mrw_reg <- fread('https://www.ssc.wisc.edu/~bhansen/econometrics/MRW1992.txt')

# estimate unrestricted model

mrw_reg <- mrw %>% 
  filter(N == 1) %>% 
  mutate(log_growth = log(Y85) - log(Y60),
         log_gdp = log(Y60),
         log_i = log(invest/100),
         log_vars = log(.05 + pop_growth/100),
         log_school = log(school/100))

mrw_basic_reg <- lm(log_growth ~ log_gdp + log_i + log_vars + log_school,
                    data = mrw_reg)

mrw_basic_reg_summary <- tidy(mrw_basic_reg)

# get heteroskedastic-consistent SEs from sandwich

V_mrw <- vcovHC(mrw_basic_reg, 
                type = "HC1")

se_mrw <- V_mrw %>% 
  diag() %>% 
  sqrt()

# report to show the output matches

data.table('term' = mrw_basic_reg_summary$term,
           'estimate' = mrw_basic_reg_summary$estimate,
           'std.error' = se_mrw) %>% 
  kable(.,
        format = 'latex',
        digits = 2,
        align = 'lrr',
        caption = 'Estimates of Solow Growth Model')

# define parameters for test

beta_ols <- mrw_basic_reg_summary$estimate
R <- c(0,0,1,1,1)
c <- 0
X <- mrw_reg %>% 
  mutate(cons = 1) %>% 
  select(cons, log_gdp, log_i, log_vars, log_school) %>% 
  as.matrix()
  
# run the Wald test

constraint <- t(R) %*% beta_ols - c
w_n <- t(constraint) %*% ginv(t(R) %*% V_mrw %*% R) %*% constraint %>% 
  as.vector()

p_wald <- round(1 - pchisq(w_n, df = 1), 2)
wald_text <- 'The p-value of the Wald test is ${p_wald}.'
cat(str_interp(wald_text))
