Hanson Problem 10.28
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

## Hanson Problem 9.26

### Get OLS Estimates

``` r
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

# get robust SEs

V_HC1 <- vcovHC(basic_reg, 
                type = "HC1")

# output results

basic_reg_results <- basic_reg %>% 
  coeftest(., vcov = V_HC1) %>% 
  tidy()

basic_reg_results %>%
  kable(., 
        format = 'html',
        digits = 4,
        align = 'lrrrr',
        caption = 'Ordinary Least Squares')
```

<table>

<caption>

Ordinary Least Squares

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

<th style="text-align:right;">

statistic

</th>

<th style="text-align:right;">

p.value

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

(Intercept)

</td>

<td style="text-align:right;">

\-3.5265

</td>

<td style="text-align:right;">

1.7186

</td>

<td style="text-align:right;">

\-2.0520

</td>

<td style="text-align:right;">

0.0420

</td>

</tr>

<tr>

<td style="text-align:left;">

output\_log

</td>

<td style="text-align:right;">

0.7204

</td>

<td style="text-align:right;">

0.0326

</td>

<td style="text-align:right;">

22.0997

</td>

<td style="text-align:right;">

0.0000

</td>

</tr>

<tr>

<td style="text-align:left;">

plabor\_log

</td>

<td style="text-align:right;">

0.4363

</td>

<td style="text-align:right;">

0.2456

</td>

<td style="text-align:right;">

1.7764

</td>

<td style="text-align:right;">

0.0778

</td>

</tr>

<tr>

<td style="text-align:left;">

pcapital\_log

</td>

<td style="text-align:right;">

\-0.2199

</td>

<td style="text-align:right;">

0.3238

</td>

<td style="text-align:right;">

\-0.6791

</td>

<td style="text-align:right;">

0.4982

</td>

</tr>

<tr>

<td style="text-align:left;">

pfuel\_log

</td>

<td style="text-align:right;">

0.4265

</td>

<td style="text-align:right;">

0.0755

</td>

<td style="text-align:right;">

5.6505

</td>

<td style="text-align:right;">

0.0000

</td>

</tr>

</tbody>

</table>

### Get CLS Estimates

``` r
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
        format = 'html',
        digits = 4,
        align = 'lrr',
        caption = 'Constrained Least Squares')
```

<table>

<caption>

Constrained Least Squares

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

\-4.6908

</td>

<td style="text-align:right;">

0.8149

</td>

</tr>

<tr>

<td style="text-align:left;">

output\_log

</td>

<td style="text-align:right;">

0.7207

</td>

<td style="text-align:right;">

0.0325

</td>

</tr>

<tr>

<td style="text-align:left;">

plabor\_log

</td>

<td style="text-align:right;">

0.5929

</td>

<td style="text-align:right;">

0.1691

</td>

</tr>

<tr>

<td style="text-align:left;">

pcapital\_log

</td>

<td style="text-align:right;">

\-0.0074

</td>

<td style="text-align:right;">

0.1558

</td>

</tr>

<tr>

<td style="text-align:left;">

pfuel\_log

</td>

<td style="text-align:right;">

0.4145

</td>

<td style="text-align:right;">

0.0729

</td>

</tr>

</tbody>

</table>

### Get Efficient Minimum Distance Estimates

``` r
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
        format = 'html',
        digits = 4,
        align = 'lrr',
        caption = 'Minimum Distance Estimation')
```

<table>

<caption>

Minimum Distance Estimation

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

\-4.7446

</td>

<td style="text-align:right;">

0.8154

</td>

</tr>

<tr>

<td style="text-align:left;">

output\_log

</td>

<td style="text-align:right;">

0.7202

</td>

<td style="text-align:right;">

0.0323

</td>

</tr>

<tr>

<td style="text-align:left;">

plabor\_log

</td>

<td style="text-align:right;">

0.5805

</td>

<td style="text-align:right;">

0.1695

</td>

</tr>

<tr>

<td style="text-align:left;">

pcapital\_log

</td>

<td style="text-align:right;">

0.0092

</td>

<td style="text-align:right;">

0.1552

</td>

</tr>

<tr>

<td style="text-align:left;">

pfuel\_log

</td>

<td style="text-align:right;">

0.4103

</td>

<td style="text-align:right;">

0.0724

</td>

</tr>

</tbody>

</table>

### Compute Wald Test

``` r
# wald test

w <- t(constraint) %*% ginv(t(R) %*% V_HC1 %*% R) %*% constraint %>% 
  as.vector()

p_wald <- round(1 - pchisq(w, df = q), 2)
wald_text <- 'The p-value of the Wald test is ${p_wald}'
cat(str_interp(wald_text))
```

The p-value of the Wald test is 0.42

### Compute EMD Test

``` r
# emd test

j <- t(beta_ols - beta_emd) %*% ginv(V_HC1) %*% (beta_ols - beta_emd)

p_emd <- round(1 - pchisq(j, df = q), 2)
emd_text <- 'The p-value of the EMD test is ${p_emd}'
cat(str_interp(emd_text))
```

The p-value of the EMD test is 0.42

## Hanson Problem 10.28

### Create Jackknife Estimator for OLS

``` r
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
        format = 'html',
        digits = 4,
        align = 'lrr',
        caption = 'OLS with Jackknife SEs')
```

<table>

<caption>

OLS with Jackknife SEs

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

\-3.5265

</td>

<td style="text-align:right;">

1.7880

</td>

</tr>

<tr>

<td style="text-align:left;">

output\_log

</td>

<td style="text-align:right;">

0.7204

</td>

<td style="text-align:right;">

0.0339

</td>

</tr>

<tr>

<td style="text-align:left;">

plabor\_log

</td>

<td style="text-align:right;">

0.4363

</td>

<td style="text-align:right;">

0.2532

</td>

</tr>

<tr>

<td style="text-align:left;">

pcapital\_log

</td>

<td style="text-align:right;">

\-0.2199

</td>

<td style="text-align:right;">

0.3363

</td>

</tr>

<tr>

<td style="text-align:left;">

pfuel\_log

</td>

<td style="text-align:right;">

0.4265

</td>

<td style="text-align:right;">

0.0778

</td>

</tr>

</tbody>

</table>

### Construct Bootstrap Estimator for OLS

``` r
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
        format = 'html',
        digits = 4,
        align = 'lrr',
        caption = 'OLS with Bootstrap SEs')
```

<table>

<caption>

OLS with Bootstrap SEs

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

<th style="text-align:left;">

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

\-3.5265

</td>

<td style="text-align:right;">

1.7186

</td>

<td style="text-align:left;">

1.7880

</td>

<td style="text-align:right;">

1.7477

</td>

</tr>

<tr>

<td style="text-align:left;">

output\_log

</td>

<td style="text-align:right;">

0.7204

</td>

<td style="text-align:right;">

0.0326

</td>

<td style="text-align:left;">

0.0339

</td>

<td style="text-align:right;">

0.0346

</td>

</tr>

<tr>

<td style="text-align:left;">

plabor\_log

</td>

<td style="text-align:right;">

0.4363

</td>

<td style="text-align:right;">

0.2456

</td>

<td style="text-align:left;">

0.2532

</td>

<td style="text-align:right;">

0.2503

</td>

</tr>

<tr>

<td style="text-align:left;">

pcapital\_log

</td>

<td style="text-align:right;">

\-0.2199

</td>

<td style="text-align:right;">

0.3238

</td>

<td style="text-align:left;">

0.3363

</td>

<td style="text-align:right;">

0.3327

</td>

</tr>

<tr>

<td style="text-align:left;">

pfuel\_log

</td>

<td style="text-align:right;">

0.4265

</td>

<td style="text-align:right;">

0.0755

</td>

<td style="text-align:left;">

0.0778

</td>

<td style="text-align:right;">

0.0775

</td>

</tr>

</tbody>

</table>

### Estimate Theta and Standard Errors

``` r
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
        format = 'html',
        digits = 4,
        align = 'lrrr',
        caption = 'Estimate of Theta in MRW Model')
```

<table>

<caption>

Estimate of Theta in MRW Model

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

0.643

</td>

<td style="text-align:right;">

0.4444

</td>

<td style="text-align:right;">

0.4627

</td>

<td style="text-align:right;">

0.4453

</td>

</tr>

</tbody>

</table>

### Bootstrapped Confidence Intervals using Percentile Method

``` r
theta_boot <- beta_boot %>% 
  mutate(theta = plabor_log + pcapital_log + pfuel_log) %>% 
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

\-0.1886

</td>

<td style="text-align:right;">

1.5472

</td>

</tr>

</tbody>

</table>

### Bootstrapped Confidence Intervals using BC Alpha

``` r
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
        format = 'html',
        digits = 4,
        align = 'rr',
        caption = 'Bootstrapped CIs Using BC Alpha Method')
```

<table>

<caption>

Bootstrapped CIs Using BC Alpha Method

</caption>

<thead>

<tr>

<th style="text-align:right;">

2.095656%

</th>

<th style="text-align:right;">

97.08089%

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:right;">

\-0.2156

</td>

<td style="text-align:right;">

1.5383

</td>

</tr>

</tbody>

</table>
