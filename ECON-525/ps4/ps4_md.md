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
##  [1] 0.01158301 0.05140187 0.12290970 0.23697651 0.39442815 0.48360656
##  [7] 0.59006211 0.74137931 0.72727273 0.66666667 0.99000000
```

The values of $\hat{p}$ and $\hat{P}$ imply this transition matrix:

```
##              [,1]      [,2]      [,3]      [,4]     [,5]      [,6]       [,7]
##  [1,] 0.212355212 0.7876448 0.0000000 0.0000000 0.000000 0.0000000 0.00000000
##  [2,] 0.007009346 0.2344237 0.7585670 0.0000000 0.000000 0.0000000 0.00000000
##  [3,] 0.013377926 0.1095318 0.1856187 0.6914716 0.000000 0.0000000 0.00000000
##  [4,] 0.046986721 0.1899898 0.0000000 0.1552605 0.607763 0.0000000 0.00000000
##  [5,] 0.076530836 0.3178973 0.0000000 0.0000000 0.127566 0.4780059 0.00000000
##  [6,] 0.106557377 0.3770492 0.0000000 0.0000000 0.000000 0.1092896 0.40710383
##  [7,] 0.173913043 0.4161491 0.0000000 0.0000000 0.000000 0.0000000 0.07453416
##  [8,] 0.206896552 0.5344828 0.0000000 0.0000000 0.000000 0.0000000 0.00000000
##  [9,] 0.090909091 0.6363636 0.0000000 0.0000000 0.000000 0.0000000 0.00000000
## [10,] 0.000000000 0.6666667 0.0000000 0.0000000 0.000000 0.0000000 0.00000000
## [11,] 0.000000000 0.9900000 0.0000000 0.0000000 0.000000 0.0000000 0.00000000
##             [,8]      [,9]     [,10]     [,11]
##  [1,] 0.00000000 0.0000000 0.0000000 0.0000000
##  [2,] 0.00000000 0.0000000 0.0000000 0.0000000
##  [3,] 0.00000000 0.0000000 0.0000000 0.0000000
##  [4,] 0.00000000 0.0000000 0.0000000 0.0000000
##  [5,] 0.00000000 0.0000000 0.0000000 0.0000000
##  [6,] 0.00000000 0.0000000 0.0000000 0.0000000
##  [7,] 0.33540373 0.0000000 0.0000000 0.0000000
##  [8,] 0.06896552 0.1896552 0.0000000 0.0000000
##  [9,] 0.00000000 0.0000000 0.2727273 0.0000000
## [10,] 0.00000000 0.0000000 0.0000000 0.3333333
## [11,] 0.00000000 0.0000000 0.0000000 0.0100000
```
Note that we needed to make two small modifications to allow for estimation of the problem, due to the simulated data. First, we needed to assign the transition probability of $x_t = 10$ if $i_t = 0$, as this does not occur in the data. Second, we need to adjust any value of $\hat{p} = 1$ to $\hat{p} = .99$ to ensure that we can estimate the model (as $\log(0)$ is undefined). 

#### Compute Likelihood

We use the above transition matrix to solve
$$\hat{V} = \left(I - \beta G)^{-1}\Pi$$
for any parameters $\theta_31,\theta_32,\theta_3$. We can then compute the likelihood by comparing the implied probabilities to the true probabilities from the simulation:

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
   <td style="text-align:right;"> 3.9 </td>
   <td style="text-align:right;"> -2122.800 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.3 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 3.8 </td>
   <td style="text-align:right;"> -2123.576 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.3 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 4.0 </td>
   <td style="text-align:right;"> -2124.019 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.3 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 3.7 </td>
   <td style="text-align:right;"> -2126.428 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.2 </td>
   <td style="text-align:right;"> 0.02 </td>
   <td style="text-align:right;"> 3.6 </td>
   <td style="text-align:right;"> -2128.823 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.3 </td>
   <td style="text-align:right;"> 0.01 </td>
   <td style="text-align:right;"> 4.0 </td>
   <td style="text-align:right;"> -2129.112 </td>
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

set.seed(2022)

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
  summarise(val = sum((row_number() != 1) * (x_t * (1-i_t)) * beta^(row_number()-1))) %>% 
  ungroup() %>% 
  group_by(start_val) %>% 
  summarise(A_val = (1 / n_distinct(sim_no)) * sum(val)) %>% 
  ungroup()
  
B <- all_sims %>% 
  group_by(sim_no, start_val) %>% 
  summarise(val = sum((row_number() != 1) * i_t * beta^(row_number()-1))) %>% 
  ungroup() %>% 
  group_by(start_val) %>% 
  summarise(B_val = (1 / n_distinct(sim_no)) * sum(val)) %>% 
  ungroup()

C <- all_sims %>% 
  group_by(sim_no, start_val) %>% 
  summarise(val = sum((row_number() != 1) * (e_1 * i_t + e_0 * (1-i_t)) * beta^(row_number()-1))) %>% 
  ungroup() %>% 
  group_by(start_val) %>% 
  summarise(C_val = (1 / n_distinct(sim_no)) * sum(val)) %>% 
  ungroup()

D <- all_sims %>% 
  group_by(sim_no, start_val) %>% 
  summarise(val = sum((row_number() != 1) * (x_t * (1-i_t))^2 * beta^(row_number()-1))) %>% 
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
    
    V_0 <- - t1 * c(0:10) - t2 * c(0:10)^2 + continuation 
    V_1 <- - t3 + continuation
    
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
   <td style="text-align:right;"> 0.7 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 3.4 </td>
   <td style="text-align:right;"> -2135.860 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.7 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 3.5 </td>
   <td style="text-align:right;"> -2138.693 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.6 </td>
   <td style="text-align:right;"> 0.01 </td>
   <td style="text-align:right;"> 3.2 </td>
   <td style="text-align:right;"> -2139.450 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.6 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 3.1 </td>
   <td style="text-align:right;"> -2139.461 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.7 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 3.3 </td>
   <td style="text-align:right;"> -2139.685 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.6 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 3.0 </td>
   <td style="text-align:right;"> -2141.753 </td>
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

V_0_fs <- - theta1_fs * c(0:10) - theta2_fs *c(0:10)^2 + continuation
V_1_fs <- - theta3_fs + continuation

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
   <td style="text-align:right;"> -16.33540 </td>
   <td style="text-align:right;"> -19.73540 </td>
   <td style="text-align:right;"> 0.0322955 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -18.10528 </td>
   <td style="text-align:right;"> -20.80528 </td>
   <td style="text-align:right;"> 0.0629734 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> -19.52787 </td>
   <td style="text-align:right;"> -21.52787 </td>
   <td style="text-align:right;"> 0.1192029 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 3 </td>
   <td style="text-align:right;"> -20.04357 </td>
   <td style="text-align:right;"> -21.34357 </td>
   <td style="text-align:right;"> 0.2141650 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 4 </td>
   <td style="text-align:right;"> -20.80334 </td>
   <td style="text-align:right;"> -21.40334 </td>
   <td style="text-align:right;"> 0.3543437 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 5 </td>
   <td style="text-align:right;"> -21.24745 </td>
   <td style="text-align:right;"> -21.14745 </td>
   <td style="text-align:right;"> 0.5249792 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 6 </td>
   <td style="text-align:right;"> -21.18419 </td>
   <td style="text-align:right;"> -20.38419 </td>
   <td style="text-align:right;"> 0.6899745 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 7 </td>
   <td style="text-align:right;"> -21.58281 </td>
   <td style="text-align:right;"> -20.08281 </td>
   <td style="text-align:right;"> 0.8175745 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8 </td>
   <td style="text-align:right;"> -22.52327 </td>
   <td style="text-align:right;"> -20.32327 </td>
   <td style="text-align:right;"> 0.9002495 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 9 </td>
   <td style="text-align:right;"> -23.27421 </td>
   <td style="text-align:right;"> -20.37421 </td>
   <td style="text-align:right;"> 0.9478464 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 10 </td>
   <td style="text-align:right;"> -23.90722 </td>
   <td style="text-align:right;"> -20.30722 </td>
   <td style="text-align:right;"> 0.9734030 </td>
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


We see that the value functions are fairly different, although the replacement probabilities are close (which is to be expected due to the method employed). Further, the forward simulation method fails to recover the correct parameters, returning (0.7,0,3.4) instead of (.3,0,4). I suspect this is because the data from the initial simulation is spotty for certain values of $x_t$, which will cause the forward simulations to be inexact.
