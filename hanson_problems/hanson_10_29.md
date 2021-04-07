Hanson Problem 10.29
================

## Clear Environment and Import Packages

``` r
rm(list = ls())

gc()
gc()

library(data.table) # probably not important for this
library(broom) # create regression tables
library(knitr) # create kables for latex
library(sandwich) # make nice HC covariance matrices
library(lmtest) # for robust coefficient testing
library(MASS) # some statistics package
library(tidyverse) # important for data manipulation, includes dplyr
```

## Hanson Problem 9.27

### Replicate 8.12

``` r
set.seed(2021)

# read in data

mrw <- fread('https://www.ssc.wisc.edu/~bhansen/econometrics/MRW1992.txt')

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
        format = 'html',
        digits = 4,
        align = 'lrr',
        caption = 'Estimates of Solow Growth Model')
```

<table>

<caption>

Estimates of Solow Growth Model

</caption>

<thead>

<tr>

<th style="text-align:left;">

term

</th>

<th style="text-align:right;">

estimate

</th>

<th style="text-align:right;">

std.error

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

(Intercept)

</td>

<td style="text-align:right;">

3.0215

</td>

<td style="text-align:right;">

0.7373

</td>

</tr>

<tr>

<td style="text-align:left;">

log\_gdp

</td>

<td style="text-align:right;">

\-0.2884

</td>

<td style="text-align:right;">

0.0543

</td>

</tr>

<tr>

<td style="text-align:left;">

log\_i

</td>

<td style="text-align:right;">

0.5237

</td>

<td style="text-align:right;">

0.1073

</td>

</tr>

<tr>

<td style="text-align:left;">

log\_vars

</td>

<td style="text-align:right;">

\-0.5057

</td>

<td style="text-align:right;">

0.2360

</td>

</tr>

<tr>

<td style="text-align:left;">

log\_school

</td>

<td style="text-align:right;">

0.2311

</td>

<td style="text-align:right;">

0.0664

</td>

</tr>

</tbody>

</table>

### Run Wald Test

``` r
# define parameters for test

beta_ols <- mrw_basic_reg_summary$estimate
R <- c(0,0,1,1,1)
c <- 0
X <- mrw_reg %>% 
  mutate(cons = 1) %>% 
  select(cons, log_gdp, log_i, log_vars, log_school) %>% 
  as.matrix()
n <- dim(X)[1]
k <- dim(X)[2]
q <- length(c)
# run the Wald test

constraint <- t(R) %*% beta_ols - c
w_n <- t(constraint) %*% ginv(t(R) %*% V_mrw %*% R) %*% constraint %>% 
  as.vector()

p_wald <- round(1 - pchisq(w_n, df = q), 2)
wald_text <- 'The p-value of the Wald test is ${p_wald}.'
cat(str_interp(wald_text))
```

The p-value of the Wald test is 0.36.

## Hanson Problem 10.29

### Get Jackknife Estimates

``` r
# get beta estimators from jackknife

data_id <- mrw_reg %>% 
  mutate(id = row_number())

beta_jk <- map_dfr(c(1:nrow(data_id)), function(x){
  reg_jk <- lm(log_growth ~ log_gdp + log_i + log_vars + log_school,
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
```

### Get Bootstrap Estimates

``` r
# construct bootstrap estimator

beta_boot <- map_dfr(c(1:nrow(data_id)), function(x){
  reg_boot <- lm(log_growth ~ log_gdp + log_i + log_vars + log_school,
               data = data_id %>% 
                 dplyr::sample_n(size = nrow(data_id),
                                 replace = TRUE))
  return(reg_boot$coefficients)
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

var_boot <- (1/(n-1)) * Reduce('+', mats)
se_boot <- sqrt(diag(var_boot))
```

### Return Estimates in Table Form

``` r
data.table('Term' = mrw_basic_reg_summary$term,
           'Estimate' = mrw_basic_reg_summary$estimate,
           'SE Asymptotic' = se_mrw,
           'SE Jackknife' = se_jk,
           'SE Bootstrap' = se_boot) %>% 
  kable(.,
        format = 'html',
        digits = 4,
        align = 'lrrrr',
        caption = 'Estimates of Solow Growth Model')
```

<table>

<caption>

Estimates of Solow Growth Model

</caption>

<thead>

<tr>

<th style="text-align:left;">

Term

</th>

<th style="text-align:right;">

Estimate

</th>

<th style="text-align:right;">

SE Asymptotic

</th>

<th style="text-align:right;">

SE Jackknife

</th>

<th style="text-align:right;">

SE Bootstrap

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

(Intercept)

</td>

<td style="text-align:right;">

3.0215

</td>

<td style="text-align:right;">

0.7373

</td>

<td style="text-align:right;">

0.7563

</td>

<td style="text-align:right;">

0.7141

</td>

</tr>

<tr>

<td style="text-align:left;">

log\_gdp

</td>

<td style="text-align:right;">

\-0.2884

</td>

<td style="text-align:right;">

0.0543

</td>

<td style="text-align:right;">

0.0569

</td>

<td style="text-align:right;">

0.0492

</td>

</tr>

<tr>

<td style="text-align:left;">

log\_i

</td>

<td style="text-align:right;">

0.5237

</td>

<td style="text-align:right;">

0.1073

</td>

<td style="text-align:right;">

0.1116

</td>

<td style="text-align:right;">

0.1063

</td>

</tr>

<tr>

<td style="text-align:left;">

log\_vars

</td>

<td style="text-align:right;">

\-0.5057

</td>

<td style="text-align:right;">

0.2360

</td>

<td style="text-align:right;">

0.2447

</td>

<td style="text-align:right;">

0.2202

</td>

</tr>

<tr>

<td style="text-align:left;">

log\_school

</td>

<td style="text-align:right;">

0.2311

</td>

<td style="text-align:right;">

0.0664

</td>

<td style="text-align:right;">

0.0690

</td>

<td style="text-align:right;">

0.0677

</td>

</tr>

</tbody>

</table>

## Estimate Theta and SEs

``` r
theta <- mrw_basic_reg_summary %>% 
  filter(term %in% c('log_i', 'log_vars', 'log_school')) %>% 
  pull(estimate) %>% 
  sum()

# variance is given by the variance-covariance matrix 

get_theta_var <- function(vcov_mat){
  return(sqrt(sum(vcov_mat[3:5,3:5])))
}

data.table('Theta' = theta,
           'SE Asymptotic' = get_theta_var(V_mrw),
           'SE Jackknife' = get_theta_var(var_jk),
           'SE Bootstrap' = get_theta_var(var_boot)) %>% 
  kable(.,
        format = 'html',
        digits = 4,
        align = 'lrrr',
        caption = 'Estimate of Theta in Solow Growth Model')
```

<table>

<caption>

Estimate of Theta in Solow Growth Model

</caption>

<thead>

<tr>

<th style="text-align:left;">

Theta

</th>

<th style="text-align:right;">

SE Asymptotic

</th>

<th style="text-align:right;">

SE Jackknife

</th>

<th style="text-align:right;">

SE Bootstrap

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

0.2492

</td>

<td style="text-align:right;">

0.2725

</td>

<td style="text-align:right;">

0.2809

</td>

<td style="text-align:right;">

0.2556

</td>

</tr>

</tbody>

</table>

## Get Bootstrapped Confidence Intervals

### Percentile Confidence Interval

``` r
theta_boot <- beta_boot %>% 
  mutate(theta = log_i + log_vars + log_school) %>% 
  pull()

quantile(theta_boot, c(.025, .975)) %>% 
  t() %>% 
  kable(.,
        format = 'html',
        digits = 4,
        align = 'rr',
        caption = 'Bootstrapped CIs Using Percentile Method')
```

<table>

<caption>

Bootstrapped CIs Using Percentile Method

</caption>

<thead>

<tr>

<th style="text-align:right;">

2.5%

</th>

<th style="text-align:right;">

97.5%

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:right;">

\-0.2304

</td>

<td style="text-align:right;">

0.6604

</td>

</tr>

</tbody>

</table>

### BC Percentile Confidence Interval

``` r
# to correct with bias, we need the median bias

p_star <- sum(theta_boot >= theta) / length(theta_boot)
z_0 <- qnorm(p_star)

x_alpha <- function(alpha){
  return(pnorm(qnorm(alpha) + 2*z_0))
}

quantile(theta_boot, c(x_alpha(.025), 
                       x_alpha(.975))) %>% 
  t() %>% 
  kable(.,
        col.names = c('2.5%', '97.5%'),
        format = 'html',
        digits = 4,
        align = 'rr',
        caption = 'Bootstrapped CIs Using BC Percentile Method')
```

<table>

<caption>

Bootstrapped CIs Using BC Percentile Method

</caption>

<thead>

<tr>

<th style="text-align:right;">

2.5%

</th>

<th style="text-align:right;">

97.5%

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:right;">

\-0.1835

</td>

<td style="text-align:right;">

0.6741

</td>

</tr>

</tbody>

</table>
