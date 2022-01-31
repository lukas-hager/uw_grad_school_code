rm(list = ls())
gc()
gc()

library(data.table)
library(evd)
library(tidyverse)

options(dplyr.summarise.inform = FALSE)

yuya_rvs <- fread('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/ECON 525/Problem Sets/Assignment 2-3/draw.csv') %>% 
  mutate(lambda = as.numeric(V2 >= .2))

v_final <- fread('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/ECON 525/Problem Sets/Assignment 2-3/v_final.csv')

x <- 0

sims <- bind_rows(
  lapply(c(1:5000),function(i){
    if(i == 1){
      x_old <<- x
      return(c('i_t' = 0, 'x_t' = x))
    }else{
      lambda <- yuya_rvs %>% 
        slice(i) %>% 
        pull(lambda)
      
      x_new <- min(x_old + lambda, 10)
      
      prob <- v_final %>% 
        slice(x_new+1) %>% 
        pull(p)
      
      i_t <- yuya_rvs %>% 
        slice(i) %>% 
        mutate(i = as.numeric(V1 >= 1-prob)) %>% 
        pull(i)
      
      if(i_t == 1){
        x_old <<- 0
      }else{
        x_old <<- x_new
      }
      return(c('i_t' = i_t, 'x_t' = x_new))
      
    }
  })
)

# sims <- bind_rows(
#   lapply(c(1:20000),function(i){
#     if(i == 1){
#       x_old <<- x
#       return(c('i_t' = 0, 'x_t' = x))
#     }else{
#       lambda <- rbinom(1,1,.8)
#       
#       x_new <- min(x_old + lambda, 10)
#       
#       prob <- v_final %>% 
#         slice(x_new+1) %>% 
#         pull(p)
#       
#       i_t <- rbinom(1,1,prob)
#       
#       if(i_t == 1){
#         x_old <<- 0
#       }else{
#         x_old <<- x_new
#       }
#       return(c('i_t' = i_t, 'x_t' = x_new))
#       
#     }
#   })
# )

# we have to permute some of the values away from 1 so that we can take a log

P_xt <- sims %>% 
  group_by(x_t) %>% 
  summarise(prob = mean(i_t)) %>% 
  ungroup() %>% 
  arrange(x_t) %>% 
  mutate(prob = ifelse(prob == 1, .99, prob)) %>% 
  pull(prob)

p_hat <- sims %>% 
  mutate(x_t1 = lag(x_t),
         i_t1 = lag(i_t)) %>% 
  group_by(x_t1, i_t1, x_t) %>% 
  summarise(n = n()) %>% 
  ungroup() %>% 
  group_by(x_t1, i_t1) %>% 
  mutate(perc = n / sum(n)) %>% 
  ungroup() %>% 
  arrange(x_t1, i_t1, x_t)

G_0 <- matrix(sapply(c(0:10), 
                     function(x){
                       vals <- data.table('x_t' = c(0:10)) %>% 
                         left_join(p_hat %>% 
                                     filter(x_t1 == x, i_t1 == 0),
                                   by = 'x_t') %>% 
                         mutate(perc = replace_na(perc, 0)) %>% 
                         arrange(x_t) %>% 
                         pull(perc)
                       
                       return(vals)
                     }),
              nrow = 11,
              ncol = 11,
              byrow = TRUE)

# need to set the last element to 1 because we don't have data

G_0[11,11] <- 1

G_1 <- matrix(sapply(c(0:10), 
                     function(x){
                       vals <- data.table('x_t' = c(0:10)) %>% 
                         left_join(p_hat %>% 
                                     filter(x_t1 == x, i_t1 == 1),
                                   by = 'x_t') %>% 
                         mutate(perc = replace_na(perc, 0)) %>% 
                         arrange(x_t) %>% 
                         pull(perc)
                       
                       return(vals)
                     }),
              nrow = 11,
              ncol = 11,
              byrow = TRUE)

G <- P_xt * G_1 + (1-P_xt) * G_0

mu = .5772
beta = .95

e1 <- mu - log(P_xt)
e0 <- mu - log(1-P_xt)

get_likelihood <- function(t1_vals, t2_vals, t3_vals){
  grid <- CJ('t1' = t1_vals, 't2' = t2_vals) %>% 
    left_join(CJ('t2' = t2_vals, 't3' = t3_vals),
              by = 't2') %>% 
    data.table()
  
  lapply(c(1:nrow(grid)), function(idx){
    row <- grid[idx]
    t1 <- row %>% pull(t1) 
    t2 <- row %>% pull(t2)
    t3 <- row %>% pull(t3)
    
    Pi <- P_xt * (-t3 + e1) + (1-P_xt) * (-t1 * c(0:10) -t2 * c(0:10) + e0)
    
    V <- solve(diag(11) - beta * G) %*% Pi
    
    V_0 <- as.numeric(-t1 * c(0:10) -t2 * c(0:10)^2 + beta * G_0 %*% V)
    V_1 <- as.numeric(-t3 + beta * G_1 %*% V)
    
    probs <- data.table('x_t' = c(0:10), 'p' = 1/(1+exp(V_0-V_1)))
    
    likelihood <- sims %>% 
      left_join(probs %>% 
                  select(x_t, p),
                by = 'x_t') %>% 
      mutate(event_prob = (1-i_t) * (1-p) + i_t * p) %>% 
      summarise(ll = sum(log(event_prob), na.rm = TRUE)) %>% 
      pull(ll)
    
    return(list('t1' = t1, 't2' = t2, 't3' = t3, 'll' = likelihood))
  })
}

likelihoods <- bind_rows(
  get_likelihood(seq(.1,1,by =.1), seq(0,.1,by=.01), seq(3,4,by=.1))
) %>% 
  arrange(desc(ll))

theta1 <- likelihoods %>% slice(1) %>% pull(t1)
theta2 <- likelihoods %>% slice(1) %>% pull(t2)
theta3 <- likelihoods %>% slice(1) %>% pull(t3)

Pi <- P_xt * (-theta3 + e1) + (1-P_xt) * (-theta1 * c(0:10) -theta2 * c(0:10) + e0)

V <- solve(diag(11) - beta * G) %*% Pi

V_0 <- as.numeric(-theta1 * c(0:10) -theta2 * c(0:10)^2 + beta * G_0 %*% V)
V_1 <- as.numeric(-theta3 + beta * G_1 %*% V)

v_data <- data.table(V_0, V_1) %>% 
  mutate(v_diff = V_1 - V_0) %>% 
  pull(v_diff)

# simulate

v_diff <- log(P_xt) - log(1-P_xt)

set.seed(2022)

all_sims <- bind_rows(
  lapply(c(1:110),function(sim_no){
    x_old <<- sim_no %% 11
    e_0 <<- rgumbel(5000)
    e_1 <<- rgumbel(5000)
    e_diff <<-  e_0 - e_1
    sims <- bind_rows(
      lapply(c(1:5000), function(i){
        i_t <- as.numeric(v_data[x_old + 1] > sim_data$e_diff[i])
        if(i_t == 0){
          x_new <<- sample(c(0:10), 1, replace = FALSE, prob = G_0[x_old+1,])
        }else{
          x_new <<- sample(c(0:10), 1, replace = FALSE, prob = G_1[x_old+1,])
        }
        x_t <<- x_old
        x_old <<- x_new
        return(c('i_t' = i_t, 'x_t' = x_t))
        
      })
    ) %>% 
      mutate(e_0 = e_0,
             e_1 = e_1,
             sim_no = sim_no,
             start_val = sim_no %% 11)
  })
)

A <- all_sims %>% 
  group_by(start_val) %>% 
  summarise(A_val = (1 / n_distinct(sim_no)) * sum((x_t * (1-i_t))* beta^(row_number()-1))) %>% 
  ungroup()
  
B <- all_sims %>% 
  group_by(start_val) %>% 
  summarise(B_val = (1 / n_distinct(sim_no)) * sum(i_t * beta^(row_number()-1))) %>% 
  ungroup()

C <- all_sims %>% 
  group_by(start_val) %>% 
  summarise(C_val = (1 / n_distinct(sim_no)) * sum((e_1 * i_t + e_0 * (1-i_t)) * beta^(row_number()-1))) %>% 
  ungroup()

V1_estimated <- -theta1 * A$A_val -theta3 * B$B_val + C$C_val
