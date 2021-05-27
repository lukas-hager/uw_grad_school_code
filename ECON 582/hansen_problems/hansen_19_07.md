Hansen Exercise 19.7
================
Lukas Hager
5/27/2021

### Read In Data and Filter to Relevant Observations

``` r
data_og <- 'https://www.ssc.wisc.edu/~bhansen/econometrics/DDK2011.xlsx' %>% 
  read.xlsx(., na.strings = '.')

reg_data <- data_og %>% 
  filter(!is.na(totalscore), !is.na(percentile), tracking == 1, girl == 0) %>% 
  select(totalscore, perc = percentile) %>% 
  data.table()

cat(str_interp('We are left with ${format(nrow(reg_data), big.mark = ",")} observations.'))
```

We are left with 1,473 observations.

### Calculate Rule-of-Thumb (ROT) Bandwidth

We will use the Fan and Gijbels (1996) suggestion that `q=4`, so we will
fit a fourth-order polynomial.

``` r
reg_data_poly <- reg_data %>% 
  mutate(perc_2 = perc^2,
         perc_3 = perc^3,
         perc_4 = perc^4) %>% 
  data.table()

poly_reg <- lm(totalscore ~ perc + perc_2 + perc_3 + perc_4, 
               data = reg_data_poly) 

poly_reg_2d <- poly_reg %>% 
  tidy() %>% 
  filter(str_detect(term, 'perc\\_[2-4]')) %>% 
  select(term, estimate) %>% 
  mutate(power = as.numeric(str_extract(term, '\\d')),
         estimate_adj = estimate * power * (power - 1))

fitted_2d <- reg_data_poly %>% 
  rowwise() %>% 
  mutate(fit = sum(c(1, perc, perc_2) * poly_reg_2d$estimate_adj)) %>% 
  ungroup()

B_hat <- 1 / nrow(reg_data) * sum(((1 / 2) * fitted_2d$fit)^2)

support_diff <- max(reg_data$perc) - min(reg_data$perc)

sigma_2 <- var(poly_reg$residuals) * support_diff

h_rot <- .58 * (sigma_2 / (nrow(reg_data) * B_hat))^(1/5)

cat(str_interp('We get a ROT bandwidth of `h_rot = ${round(h_rot, 2)}`.'))
```

We get a ROT bandwidth of `h_rot = 5.34`.

### Calculate Conventional Cross-Validation Bandwidth

To do this, we need to define a grid of `h` values. We will iterate over
the grid and select the value that minimizes the cross-validation loss,
which is defined as the sum of the leave-one-out estimator errors. This
is pretty time-intensive (maybe because my code isn’t efficient) because
each value of `h` that we try has \\(n^2\\)) calculations for \\(\\) and
the error.

As a methodological note: I compute the min by running the algorithm on
a coarse grid from 4 to 20 in increments of 1 to match the bounds in
Figure 19.6(a), and then on a finer grid in increments of .2 in a
neighborhood of radius 1 around the minimum found on the coarse grid to
get the final value.

``` r
h_grid_coarse <- seq(4, 20, by = 1)

get_ll_beta <- function(val,x,h){
  Z <- reg_data[-val] %>% 
    select(perc) %>% 
    mutate(cons = 1,
           perc = perc - x) %>% 
    select(cons, perc) %>% 
    as.matrix()
  
  K <- diag(dnorm(Z[,2] / h))
  
  beta <- solve(t(Z) %*% K %*% Z) %*% t(Z) %*% K %*% reg_data[-val]$totalscore
  return(t(beta))
}

get_loo_error <- function(val, h){
  
  x <- reg_data[val]$perc
  y <- reg_data[val]$totalscore
  
  beta_vals <- get_ll_beta(val, x, h)
  
  error <- y - beta_vals[, 'cons']
  
  return(error)
}

coarse_grid_results <- purrr::map_dfr(h_grid_coarse, function(h_val){
  errors <- purrr::map_dbl(1:nrow(reg_data), function(val){
    return(get_loo_error(val, h_val))
  })
  return(c('h_val' = h_val, 'loss' = mean(errors^2)))
})

min_h_coarse <- coarse_grid_results %>% 
  filter(loss == min(loss)) %>% 
  pull(h_val)

h_grid_fine <- seq(min_h_coarse - 1, min_h_coarse + 1, by = .2)

fine_grid_results <- purrr::map_dfr(h_grid_fine, function(h_val){
  errors <- purrr::map_dbl(1:nrow(reg_data), function(val){
    return(get_loo_error(val, h_val))
  })
  return(c('h_val' = h_val, 'loss' = mean(errors^2)))
})

min_h_fine <- fine_grid_results %>% 
  filter(loss == min(loss)) %>% 
  pull(h_val)

cat(str_interp('We get a CV bandwidth of `h_cv = ${min_h_fine}`.'))
```

We get a CV bandwidth of `h_cv = 9.4`.
