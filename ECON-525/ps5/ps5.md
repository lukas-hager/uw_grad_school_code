---
title: "Problem Set 5"
author: "Lukas Hager"
date: "2/22/2022"
output: 
  html_document:
    keep_md: true
always_allow_html: true
---



Set up environment:


```r
rm(list = ls())

library(data.table)
library(knitr)
library(tidyverse)
```


## Question 1

For Gauss-Chebyshev, we implement the quadrature given in the notes:


```r
b <- 1.96
a <- -1.96

eval_int_gc <- function(n){
  i_vals <- c(1:n)
  x_i <- cos(((2*i_vals-1) * pi) / (2*n))
  
  return(pi * (b-a) / (2*n) * sum((dnorm((x_i + 1) * (b-a) / 2 + a))*(1-x_i^2)^(1/2)))
}

results <- bind_rows(
  lapply(
    c(10^(c(1:6))), function(n){
      return(c('n' = n, 'int' = eval_int_gc(n)))
    }
  )
)

results
```

```
## # A tibble: 6 x 2
##         n   int
##     <dbl> <dbl>
## 1      10 0.951
## 2     100 0.950
## 3    1000 0.950
## 4   10000 0.950
## 5  100000 0.950
## 6 1000000 0.950
```
For Monte Carlo, we set a seed and compute:


```r
set.seed(18)

eval_int_mc <- function(n){
  draws <- runif(n, min = a, max = b)
  return(mean(dnorm(draws) * (b-a)))
}

results <- bind_rows(
  lapply(
    c(10^(c(1:6))), function(n){
      return(c('n' = n, 'int' = eval_int_mc(n)))
    }
  )
)

results
```

```
## # A tibble: 6 x 2
##         n   int
##     <dbl> <dbl>
## 1      10 0.954
## 2     100 0.918
## 3    1000 0.943
## 4   10000 0.951
## 5  100000 0.950
## 6 1000000 0.950
```


For the exact integral, we can use `R`'s built in function:


```r
pnorm(b) - pnorm(a)
```

```
## [1] 0.9500042
```


## Question 2


```r
# read in the data

file_path <- '/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/ECON 525/Problem Sets/Assignment 5/data for hw5.csv'

data_og <- fread(file_path) %>% 
  rename_all(tolower)

# define variables

y <- data_og %>% 
  pull(d)

X <- data_og %>% 
  mutate(cons = 1) %>% 
  select(cons, x) %>% 
  as.matrix()

# write a function to return simplified log likelihood

get_ll <- function(b_hat,X_val,y_val){
  prod <- X_val %*% b_hat
  phi_term <- pnorm(prod)
  return(-1 * sum(log(y_val * phi_term + (1-y_val) * (1-phi_term))))
}

# run a regression to get starting coefficients

lm_coef <- lm(d ~ x, data = data_og)$coefficients

# optimize to get the minimum

optimization <- optim(lm_coef, get_ll, X_val = X, y_val = y, control = list(maxit = 1000000))
beta_hat_optim <- optimization$par
names(beta_hat_optim) <- dimnames(X)[[2]]

print(beta_hat_optim)
```

```
##       cons          x 
##  1.6348198 -0.3005263
```

To confirm accuracy, we compare with the results from the `R` implementation of probit:


```r
# compare to base r formulation

model <- glm(d ~ x, 
             data = data_og, 
             family=binomial(link='probit'))

print(model$coefficients)
```

```
## (Intercept)           x 
##   1.6350699  -0.3005798
```

Using the forward difference, we assess using different values of $h$:


```r
alpha_hat <- beta_hat_optim[1]
beta_hat <- beta_hat_optim[2]

get_ll_dplyr <- function(a_hat, b_hat, x_val, y_val){
  prod <- a_hat + x_val * b_hat
  phi_term <- pnorm(prod)
  return(log(y_val * phi_term + (1-y_val) * (1-phi_term)))
}

forward_diff <- function(a_hat, b_hat, h){
  partials <- data_og %>% 
    mutate(h_alpha = h * pmax(1,abs(a_hat)),
           h_beta = h * pmax(1,abs(b_hat)),
           f_x_0 = get_ll_dplyr(a_hat, b_hat, x, d),
           f_x_alpha = get_ll_dplyr(a_hat + h_alpha, b_hat, x, d),
           f_x_beta = get_ll_dplyr(a_hat, b_hat + h_beta, x, d),
           d_alpha = (f_x_alpha - f_x_0) / h_alpha,
           d_beta = (f_x_beta - f_x_0) / h_beta) %>% 
    select(d_alpha, d_beta) %>% 
    as.matrix()
  
  v_cov <- solve(t(partials) %*% partials)
  return(sqrt(diag(v_cov)))
}

results <- bind_rows(
  lapply(
    10^(-c(0:6)), function(h_val){
      list(c('h' = h_val, forward_diff(alpha_hat, beta_hat, h_val)))
    }
  )
)

results
```

```
## # A tibble: 7 x 3
##          h d_alpha  d_beta
##      <dbl>   <dbl>   <dbl>
## 1 1         0.0274 0.00394
## 2 0.1       0.0512 0.0130 
## 3 0.01      0.0543 0.0140 
## 4 0.001     0.0543 0.0140 
## 5 0.0001    0.0543 0.0140 
## 6 0.00001   0.0543 0.0140 
## 7 0.000001  0.0543 0.0140
```


```r
central_diff <- function(a_hat, b_hat, h){
  partials <- data_og %>% 
    mutate(h_alpha = h * pmax(1,abs(a_hat)),
           h_beta = h * pmax(1,abs(b_hat)),
           f_x_alpha_low = get_ll_dplyr(a_hat - h_alpha, b_hat, x, d),
           f_x_alpha_high = get_ll_dplyr(a_hat + h_alpha, b_hat, x, d),
           f_x_beta_low = get_ll_dplyr(a_hat, b_hat - h_beta, x, d),
           f_x_beta_high = get_ll_dplyr(a_hat, b_hat + h_beta, x, d),
           d_alpha = (f_x_alpha_high - f_x_alpha_low) / (2 * h_alpha),
           d_beta = (f_x_beta_high - f_x_beta_low) / (2 * h_beta)) %>% 
    select(d_alpha, d_beta) %>% 
    as.matrix()
  
  v_cov <- solve(t(partials) %*% partials)
  return(sqrt(diag(v_cov)))
}

results <- bind_rows(
  lapply(
    10^(-c(0:6)), function(h_val){
      list(c('h' = h_val, central_diff(alpha_hat, beta_hat, h_val)))
    }
  )
)

results
```

```
## # A tibble: 7 x 3
##          h d_alpha  d_beta
##      <dbl>   <dbl>   <dbl>
## 1 1         0.0360 0.00572
## 2 0.1       0.0538 0.0137 
## 3 0.01      0.0543 0.0140 
## 4 0.001     0.0543 0.0140 
## 5 0.0001    0.0543 0.0140 
## 6 0.00001   0.0543 0.0140 
## 7 0.000001  0.0543 0.0140
```



To compute the closed form, we need to take a first order condition with respect to $\alpha$, $\beta$ (taking $\theta$ here to refer to all coefficients, and $X_i$ to refer to $x_i$ and the constant):
$$\frac{\partial \mathcal{L}}{\partial \alpha} = \sum_{i=1}^n\frac{y_i\phi(X_i'\theta)}{\Phi(X_i'\theta)} - \frac{(1-y_i)\phi(X_i'\theta)}{1-\Phi(X_i'\theta)}$$

$$\frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^n\frac{x_iy_i\phi(X_i'\theta)}{\Phi(X_i'\theta)} - \frac{(1-y_i)x_i\phi(X_i'\theta)}{1-\Phi(X_i'\theta)}$$

This implies that our "true" covariance matrix and standard errors are calculated by


```r
partials <- data_og %>% 
  mutate(prod = alpha_hat + beta_hat * x,
         d_alpha = (d * dnorm(prod) / pnorm(prod)) - (1-d) * dnorm(prod) / (1-pnorm(prod)),
         d_beta = (d * x * dnorm(prod) / pnorm(prod)) - (1-d) * x * dnorm(prod) / (1-pnorm(prod))) %>% 
  select(d_alpha, d_beta) %>% 
  as.matrix()

sqrt(diag(solve(t(partials) %*% partials)))
```

```
##    d_alpha     d_beta 
## 0.05430081 0.01396878
```

