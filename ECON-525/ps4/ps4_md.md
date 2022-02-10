---
title: "Assignment 4"
author: "Lukas Hager"
date: "2/1/2022"
output: 
  html_document:
    keep_md: true
always_allow_html: true
---



#### Frequency Estimation of $\hat{P}$ and $\hat{p}$

The code to create the frequency estimator is below:


```r
suppressMessages(library(data.table))
suppressMessages(library(evd))
suppressMessages(library(tidyverse))

options(dplyr.summarise.inform = FALSE)

set.seed(2022)

yuya_rvs <- fread('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/ECON 525/Problem Sets/Assignment 2-3/draw.csv') %>% 
  mutate(lambda = as.numeric(V2 >= .2))

v_final <- fread('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/ECON 525/Problem Sets/Assignment 2-3/v_final.csv')

x <- 0

sims <- bind_rows(
  lapply(c(1:500000),function(i){
    if(i == 1){
      x_old <<- x
      return(c('i_t' = 0, 'x_t' = x))
    }else{
      lambda <- rbinom(1,1,.8)

      x_new <- min(x_old + lambda, 10)

      prob <- v_final %>%
        slice(x_new+1) %>%
        pull(p)

      i_t <- rbinom(1,1,prob)

      if(i_t == 1){
        x_old <<- 0
      }else{
        x_old <<- x_new
      }
      return(c('i_t' = i_t, 'x_t' = x_new))

    }
  })
)

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
```

This yields a value of $\hat{P}$ (which is $\mathbb{P}(i_t == 1|x_t)$) of


```
##  [1] 0.01719055 0.05478048 0.12681533 0.23506327 0.35780438 0.48466999
##  [7] 0.59663624 0.68490409 0.74181594 0.82051282 0.85000000
```

The values of $\hat{p}$ and $\hat{P}$ imply this transition matrix:

```
##             [,1]      [,2]      [,3]      [,4]      [,5]      [,6]       [,7]
##  [1,] 0.20226655 0.7977335 0.0000000 0.0000000 0.0000000 0.0000000 0.00000000
##  [2,] 0.01077456 0.2321422 0.7570833 0.0000000 0.0000000 0.0000000 0.00000000
##  [3,] 0.02651260 0.1003027 0.1748945 0.6982902 0.0000000 0.0000000 0.00000000
##  [4,] 0.04633752 0.1887257 0.0000000 0.1521010 0.6128358 0.0000000 0.00000000
##  [5,] 0.07312167 0.2846827 0.0000000 0.0000000 0.1285834 0.5136122 0.00000000
##  [6,] 0.09586398 0.3888060 0.0000000 0.0000000 0.0000000 0.1027573 0.41257266
##  [7,] 0.11727226 0.4793640 0.0000000 0.0000000 0.0000000 0.0000000 0.07927214
##  [8,] 0.14078399 0.5441201 0.0000000 0.0000000 0.0000000 0.0000000 0.00000000
##  [9,] 0.14206300 0.5997529 0.0000000 0.0000000 0.0000000 0.0000000 0.00000000
## [10,] 0.15384615 0.6666667 0.0000000 0.0000000 0.0000000 0.0000000 0.00000000
## [11,] 0.16666667 0.6833333 0.0000000 0.0000000 0.0000000 0.0000000 0.00000000
##             [,8]       [,9]      [,10]     [,11]
##  [1,] 0.00000000 0.00000000 0.00000000 0.0000000
##  [2,] 0.00000000 0.00000000 0.00000000 0.0000000
##  [3,] 0.00000000 0.00000000 0.00000000 0.0000000
##  [4,] 0.00000000 0.00000000 0.00000000 0.0000000
##  [5,] 0.00000000 0.00000000 0.00000000 0.0000000
##  [6,] 0.00000000 0.00000000 0.00000000 0.0000000
##  [7,] 0.32409161 0.00000000 0.00000000 0.0000000
##  [8,] 0.05821518 0.25688073 0.00000000 0.0000000
##  [9,] 0.00000000 0.04879555 0.20938851 0.0000000
## [10,] 0.00000000 0.00000000 0.03418803 0.1452991
## [11,] 0.00000000 0.00000000 0.00000000 0.1500000
```
Note that we needed to make two small modifications to allow for estimation of the problem, due to the simulated data. First, we needed to assign the transition probability of $x_t = 10$ if $i_t = 0$, as this does not occur in the data. Second, we need to adjust any value of $\hat{p} = 1$ to $\hat{p} = .99$ to ensure that we can estimate the model (as $\log(0)$ is undefined). 

#### Compute Likelihood

We use the above transition matrix to solve
$$\hat{V} = \left(I - \beta G\right)^{-1}\Pi$$
for any parameters $\theta_1,\theta_2,\theta_3$. We can then compute the likelihood by comparing the implied probabilities to the true probabilities from the simulation:

```r
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
```

Our top likelihood parameter choices are

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:right;"> t1 </th>
   <th style="text-align:right;"> t2 </th>
   <th style="text-align:right;"> t3 </th>
   <th style="text-align:right;"> ll </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 0.3 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 4.0 </td>
   <td style="text-align:right;"> -214494.9 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.3 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 3.9 </td>
   <td style="text-align:right;"> -214617.4 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.2 </td>
   <td style="text-align:right;"> 0.02 </td>
   <td style="text-align:right;"> 3.7 </td>
   <td style="text-align:right;"> -214874.9 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.2 </td>
   <td style="text-align:right;"> 0.02 </td>
   <td style="text-align:right;"> 3.6 </td>
   <td style="text-align:right;"> -214914.8 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.3 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 3.8 </td>
   <td style="text-align:right;"> -214945.6 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.2 </td>
   <td style="text-align:right;"> 0.02 </td>
   <td style="text-align:right;"> 3.8 </td>
   <td style="text-align:right;"> -215030.6 </td>
  </tr>
</tbody>
</table>

#### Forward Simulation

We forward simulate 1000 times per starting value of $x_t$ and subsequently compute the continuation payoff. We then get the choice specific value functions by adding the flow utility for each state, and then use these to get implied replacement probabilities. Finally, we can match those to the data to get a likelihood.


```r
v_data <- data.table(V_0, V_1) %>% 
  mutate(v_diff = V_1 - V_0) %>% 
  pull(v_diff)

v_diff <- log(P_xt) - log(1-P_xt)

all_sims <- bind_rows(
  lapply(c(1:11000),function(sim_no){
    x_old <<- sim_no %% 11
    e_0 <<- rgumbel(200)
    e_1 <<- rgumbel(200)
    e_diff <<-  e_0 - e_1
    sims <- bind_rows(
      lapply(c(1:200), function(i){
        i_t <- as.numeric(v_data[x_old + 1] > e_diff[i])
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
  group_by(sim_no, start_val) %>% 
  summarise(val = sum(x_t * (1-i_t) * beta^(row_number()))) %>% 
  ungroup() %>% 
  group_by(start_val) %>% 
  summarise(A_val = (1 / n_distinct(sim_no)) * sum(val)) %>% 
  ungroup()
  
B <- all_sims %>% 
  group_by(sim_no, start_val) %>% 
  summarise(val = sum(i_t * beta^(row_number()))) %>% 
  ungroup() %>% 
  group_by(start_val) %>% 
  summarise(B_val = (1 / n_distinct(sim_no)) * sum(val)) %>% 
  ungroup()

C <- all_sims %>% 
  group_by(sim_no, start_val) %>% 
  summarise(val = sum((e_1 * i_t + e_0 * (1-i_t)) * beta^(row_number()))) %>% 
  ungroup() %>% 
  group_by(start_val) %>% 
  summarise(C_val = (1 / n_distinct(sim_no)) * sum(val)) %>% 
  ungroup()

D <- all_sims %>% 
  group_by(sim_no, start_val) %>% 
  summarise(val = sum(x_t * (1-i_t))^2 * beta^(row_number())) %>% 
  ungroup() %>% 
  group_by(start_val) %>% 
  summarise(D_val = (1 / n_distinct(sim_no)) * sum(val)) %>% 
  ungroup()

get_likelihood_fs <- function(t1_vals, t2_vals, t3_vals){
  grid <- CJ('t1' = t1_vals, 't2' = t2_vals) %>% 
    left_join(CJ('t2' = t2_vals, 't3' = t3_vals),
              by = 't2') %>% 
    data.table()
  
  lapply(c(1:nrow(grid)), function(idx){
    row <- grid[idx]
    t1 <- row %>% pull(t1) 
    t2 <- row %>% pull(t2)
    t3 <- row %>% pull(t3)
    
    continuation <- - t1 * A$A_val -t2 * D$D_val - t3 * B$B_val + C$C_val
    
    V_0 <- - t1 * c(0:10) - t2 * c(0:10)^2 + as.numeric(G_0 %*% continuation)
    V_1 <- - t3 + as.numeric(G_1 %*% continuation)
    
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

likelihoods_fs <- bind_rows(
  get_likelihood_fs(seq(.1,1,by =.1), seq(0,.1,by=.01), seq(3,4,by=.1))
) %>% 
  arrange(desc(ll))
```

Our top likelihood parameter choices are

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:right;"> t1 </th>
   <th style="text-align:right;"> t2 </th>
   <th style="text-align:right;"> t3 </th>
   <th style="text-align:right;"> ll </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 0.3 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 4.0 </td>
   <td style="text-align:right;"> -214954.8 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.3 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 3.9 </td>
   <td style="text-align:right;"> -215442.5 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.3 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 3.8 </td>
   <td style="text-align:right;"> -216174.4 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.2 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 3.4 </td>
   <td style="text-align:right;"> -216381.3 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.2 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 3.5 </td>
   <td style="text-align:right;"> -216432.3 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.2 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 3.3 </td>
   <td style="text-align:right;"> -216576.7 </td>
  </tr>
</tbody>
</table>

#### Comparison

We can use our parameters to get the value functions and choice probabilities via forward simulation:


```r
theta1_fs <- likelihoods_fs %>% slice(1) %>% pull(t1)
theta2_fs <- likelihoods_fs %>% slice(1) %>% pull(t2)
theta3_fs <- likelihoods_fs %>% slice(1) %>% pull(t3)

continuation <- - theta1_fs * A$A_val -theta2_fs * D$D_val - theta3_fs * B$B_val + C$C_val

V_0_fs <- - theta1_fs * c(0:10) - theta2_fs *c(0:10)^2 + as.numeric(G_0 %*% continuation)
V_1_fs <- - theta3_fs + as.numeric(G_1 %*% continuation)

probs <- data.table('x_t' = c(0:10), 'V_0' = V_0_fs, 'V_1' = V_1_fs, 'p' = 1/(1+exp(V_0_fs-V_1_fs)))
```

Then we have, via forward simulation:

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:right;"> x_t </th>
   <th style="text-align:right;"> V_0 </th>
   <th style="text-align:right;"> V_1 </th>
   <th style="text-align:right;"> p </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> -5.527214 </td>
   <td style="text-align:right;"> -9.553619 </td>
   <td style="text-align:right;"> 0.0175257 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -6.708054 </td>
   <td style="text-align:right;"> -9.533170 </td>
   <td style="text-align:right;"> 0.0559819 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> -7.793092 </td>
   <td style="text-align:right;"> -9.520966 </td>
   <td style="text-align:right;"> 0.1508598 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 3 </td>
   <td style="text-align:right;"> -8.454020 </td>
   <td style="text-align:right;"> -9.532735 </td>
   <td style="text-align:right;"> 0.2537493 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 4 </td>
   <td style="text-align:right;"> -8.946663 </td>
   <td style="text-align:right;"> -9.525602 </td>
   <td style="text-align:right;"> 0.3591768 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 5 </td>
   <td style="text-align:right;"> -9.565082 </td>
   <td style="text-align:right;"> -9.532080 </td>
   <td style="text-align:right;"> 0.5082498 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 6 </td>
   <td style="text-align:right;"> -10.003484 </td>
   <td style="text-align:right;"> -9.533299 </td>
   <td style="text-align:right;"> 0.6154275 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 7 </td>
   <td style="text-align:right;"> -10.456583 </td>
   <td style="text-align:right;"> -9.524428 </td>
   <td style="text-align:right;"> 0.7175122 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8 </td>
   <td style="text-align:right;"> -11.004663 </td>
   <td style="text-align:right;"> -9.538277 </td>
   <td style="text-align:right;"> 0.8125075 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 9 </td>
   <td style="text-align:right;"> -11.257028 </td>
   <td style="text-align:right;"> -9.542228 </td>
   <td style="text-align:right;"> 0.8474578 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 10 </td>
   <td style="text-align:right;"> -11.533912 </td>
   <td style="text-align:right;"> -9.533770 </td>
   <td style="text-align:right;"> 0.8808121 </td>
  </tr>
</tbody>
</table>

The full solution yields:

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:right;"> x_t </th>
   <th style="text-align:right;"> v_0 </th>
   <th style="text-align:right;"> v_1 </th>
   <th style="text-align:right;"> p </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> -5.636115 </td>
   <td style="text-align:right;"> -9.636115 </td>
   <td style="text-align:right;"> 0.0179862 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -6.789360 </td>
   <td style="text-align:right;"> -9.636115 </td>
   <td style="text-align:right;"> 0.0548493 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> -7.713348 </td>
   <td style="text-align:right;"> -9.636115 </td>
   <td style="text-align:right;"> 0.1275534 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 3 </td>
   <td style="text-align:right;"> -8.454517 </td>
   <td style="text-align:right;"> -9.636115 </td>
   <td style="text-align:right;"> 0.2347652 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 4 </td>
   <td style="text-align:right;"> -9.061462 </td>
   <td style="text-align:right;"> -9.636115 </td>
   <td style="text-align:right;"> 0.3601638 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 5 </td>
   <td style="text-align:right;"> -9.574719 </td>
   <td style="text-align:right;"> -9.636115 </td>
   <td style="text-align:right;"> 0.4846560 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 6 </td>
   <td style="text-align:right;"> -10.023904 </td>
   <td style="text-align:right;"> -9.636115 </td>
   <td style="text-align:right;"> 0.5957504 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 7 </td>
   <td style="text-align:right;"> -10.429250 </td>
   <td style="text-align:right;"> -9.636115 </td>
   <td style="text-align:right;"> 0.6885042 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8 </td>
   <td style="text-align:right;"> -10.804056 </td>
   <td style="text-align:right;"> -9.636115 </td>
   <td style="text-align:right;"> 0.7627727 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 9 </td>
   <td style="text-align:right;"> -11.155104 </td>
   <td style="text-align:right;"> -9.636115 </td>
   <td style="text-align:right;"> 0.8203896 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 10 </td>
   <td style="text-align:right;"> -11.464407 </td>
   <td style="text-align:right;"> -9.636115 </td>
   <td style="text-align:right;"> 0.8615582 </td>
  </tr>
</tbody>
</table>

We get very close to the same results, with the exact parameters recovered (although we needed to simulate the data 5 million times).
