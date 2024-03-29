rm(list = ls())

gc()
gc()

library(data.table)
library(broom)
library(knitr)
library(sandwich)
library(lmtest)
library(scales)
library(openxlsx)
library(MASS)
library(tidyverse)

set.seed(2021)

#########################################################
# Exercise 4.26
#########################################################

data_ddk <- 'https://www.ssc.wisc.edu/~bhansen/econometrics/DDK2011.xlsx' %>% 
  read.xlsx() %>% 
  mutate_all(list(~ ifelse(. == '.', NA, .)))

data_reg <- data_ddk %>% 
  mutate(totalscore_z = (totalscore - mean(totalscore)) / sd(totalscore)) %>% 
  select(totalscore_z, tracking,  agetest, girl, etpteacher, schoolid, percentile) %>% 
  na.omit() %>% 
  mutate_all(as.numeric)

basic_reg <- lm(totalscore_z ~ tracking + agetest + girl + etpteacher + percentile,
                data = data_reg)

basic_reg_results <- basic_reg %>% 
  tidy()

V_HC1 <- vcovHC(basic_reg, 
                type = "HC1")

V_cluster <- vcovCL(basic_reg, 
                    cluster = ~ schoolid)

data.table('Term' = basic_reg_results$term,
           'Estimate' = basic_reg_results$estimate,
           'SE Robust' = sqrt(diag(V_HC1)),
           'SE Clustered' = sqrt(diag(V_cluster)),
           'SE Difference' = sqrt(diag(V_HC1)) - sqrt(diag(V_cluster))) %>% 
  kable(.,
        format = 'latex',
        digits = 4,
        align = 'lrrr',
        caption = 'Regression with Robust and Clustered SEs')

#########################################################
# Exercise 10.31
#########################################################
  
# run bootstrap

boot_data <- data_reg %>% 
  group_nest(schoolid)

B <- 1000

beta_boot <- map_dfr(c(1:B), function(val){
  reg <- lm(totalscore_z ~ tracking + agetest + girl + etpteacher + percentile,
            data = boot_data %>% 
              sample_n(nrow(boot_data), 
                       replace = TRUE) %>% 
              unnest(data))
  
  return(reg$coefficients)
})

means_boot <- beta_boot %>% 
  summarise_all(list('mean' = mean)) %>% 
  pivot_longer(everything()) %>% 
  pull(value)

mats <- map(c(1:nrow(beta_boot)), function(x){
  df <- beta_boot %>% 
    filter(row_number() == x) %>% 
    pivot_longer(everything()) %>% 
    pull(value)
  
  return((df - means_boot) %*% t(df - means_boot))
  
})

var_boot <- (1/(B-1)) * Reduce('+', mats)
se_boot <- sqrt(diag(var_boot))

data.table('Term' = basic_reg_results$term,
           'Estimate' = basic_reg_results$estimate,
           'SE Robust' = sqrt(diag(V_HC1)),
           'SE Clustered' = sqrt(diag(V_cluster)),
           'SE Bootstrap' = se_boot) %>% 
  kable(.,
        format = 'latex',
        digits = 4,
        align = 'lrrr',
        caption = 'Regression with Robust, Clustered, and Bootstrapped SEs')

# estimate jackknife for BC alpha

beta_jk <- map_dfr(c(1:nrow(data_reg)), function(x){
  reg_jk <- lm(totalscore_z ~ tracking + agetest + girl + etpteacher + percentile,
               data = data_reg %>% 
                 filter(row_number() != x))
  return(reg_jk$coefficients)
}) %>% 
  rename_all(list(~ tolower(str_replace_all(., 
                                            '\\(|\\)', 
                                            '')))) %>% 
  mutate(id = row_number())

# define parameters z_0 and a

x_alpha <- function(alpha, z_0, a){
  z_a <- qnorm(alpha)
  return(pnorm(z_0 + (z_a + z_0) / (1 + a * (z_a + z_0))))
}

vars <- colnames(beta_boot)

cis <- map_dfr(vars, function(var){
  beta_val <- beta_boot %>% 
    pull(var)
  
  beta <- basic_reg$coefficients[var]
  p_star <- sum(beta_val >= beta) / length(beta_val)
  z_0 <- qnorm(p_star)
  
  beta_bar <- mean(beta_val)
  a <- sum((beta_bar - beta_val)^3) / (6 * (sum((beta_bar - beta_val)^2)^(3/2)))
  
  out <- c(var,
           x_alpha(.025, z_0, a),
           x_alpha(.975, z_0, a),
           unname(quantile(beta_val, 
                           c(x_alpha(.025, z_0, a),
                             x_alpha(.975, z_0, a)))))
  
  names(out) <- c('Term', 'Low', 'High', 'Lower Bound', 'Upper Bound')
  return(out)
}) %>% 
  mutate_at(vars(-Term), as.numeric) %>% 
  mutate_at(vars(c(Low, High)), list(~ percent(., accuracy = .0001)))

cis %>% 
  kable(.,
        format = 'latex',
        digits = 4,
        align = 'lrrrr',
        caption = 'Bootstrapped CIs Using BC Alpha Method')
