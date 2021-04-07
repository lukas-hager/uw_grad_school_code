rm(list = ls())

gc()
gc()

library(data.table)
library(broom)
library(knitr)
library(sandwich)
library(lmtest)
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
  select(totalscore_z, tracking,  agetest, girl, etpteacher, schoolid) %>% 
  na.omit() %>% 
  mutate_all(as.numeric)

basic_reg <- lm(totalscore_z ~ tracking + agetest + girl + etpteacher,
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
  
