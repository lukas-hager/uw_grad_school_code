Hanson Problem 10.30
================

## Clear Environment and Import Packages

``` r
rm(list = ls())

gc()
gc()

library(data.table) # probably not important for this
library(broom) # create regression tables
library(knitr) # create kables for html
library(sandwich) # make nice HC covariance matrices
library(lmtest) # for robust coefficient testing
library(MASS) # some statistics package
library(openxlsx) # read excel sheets
library(tidyverse) # important for data manipulation, includes dplyr
```

## Hanson Problem 7.28

### Produce Regression Output with Robust SEs

``` r
# read in data

data_cps <- 'https://www.ssc.wisc.edu/~bhansen/econometrics/cps09mar.xlsx' %>% 
  read.xlsx()

data_reg <- data_cps %>% 
  filter(female == 0, race == 1, hisp == 1) %>% 
  mutate(log_wage = log(earnings),
         exp = age - education - 6,
         exp_2 = exp^2 / 100)

# run regression with robust SEs
basic_reg <- lm(log_wage ~ education + exp + exp_2,
                data = data_reg)

basic_reg_summary <- basic_reg %>% 
  tidy() %>% 
  data.table()

V_HC1 <- vcovHC(basic_reg, 
                type = 'HC1')

coeftest(basic_reg, 
         vcov = V_HC1) %>% 
  tidy() %>% 
  kable(.,
        format = 'html',
        digits = 2,
        align = 'lrrrr',
        caption = 'CPS Regression with Robust SEs')
```

<table>

<caption>

CPS Regression with Robust SEs

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

8.81

</td>

<td style="text-align:right;">

0.05

</td>

<td style="text-align:right;">

184.42

</td>

<td style="text-align:right;">

0

</td>

</tr>

<tr>

<td style="text-align:left;">

education

</td>

<td style="text-align:right;">

0.10

</td>

<td style="text-align:right;">

0.00

</td>

<td style="text-align:right;">

31.60

</td>

<td style="text-align:right;">

0

</td>

</tr>

<tr>

<td style="text-align:left;">

exp

</td>

<td style="text-align:right;">

0.04

</td>

<td style="text-align:right;">

0.00

</td>

<td style="text-align:right;">

13.87

</td>

<td style="text-align:right;">

0

</td>

</tr>

<tr>

<td style="text-align:left;">

exp\_2

</td>

<td style="text-align:right;">

\-0.05

</td>

<td style="text-align:right;">

0.01

</td>

<td style="text-align:right;">

\-8.96

</td>

<td style="text-align:right;">

0

</td>

</tr>

</tbody>

</table>

### Calculate Theta

``` r
# return to educ = exp(beta_1)
# return to exp at ten years = exp(beta_2 + beta_3/5)
# theta is exp(beta_1 - beta_2 - beta_3/5)

b1 <- basic_reg_summary[term == 'education']$estimate
b2 <- basic_reg_summary[term == 'exp']$estimate
b3 <- basic_reg_summary[term == 'exp_2']$estimate

theta_hat <- b1 / (b2 + b3 / 5)

# define the R vector for the nonlinear transformation

R <- c(0, 
       1/ (b2 + b3 / 5), 
       -b1 / (b2 + b3 /5)^2, 
       -(1/5) * b1 / (b2 + b3 / 5)^2)

V_theta <- t(R) %*% V_HC1 %*% R %>% 
  as.vector()

se_theta <- sqrt(V_theta)

# constructing 90% interval uses that sqrt(n) (theta_hat - theta)
# is distributed as N(0, se_theta)

ci_low <- qnorm(.05, mean = theta_hat, sd = se_theta)
ci_high <- qnorm(.95, mean = theta_hat, sd = se_theta)

# get the fitted value for educ = 12 and exp = 20

educ_ex <- 12
exp_ex <- 20

reg_vector <- c(1, educ_ex, exp_ex, exp_ex^2/100)

reg_value <- t(basic_reg$coefficients) %*% reg_vector %>% 
  as.vector()

# get SEs

reg_se <- sqrt(t(reg_vector) %*% V_HC1 %*% reg_vector) %>% 
  as.vector()

# create 95% CI

c('2.5%' = reg_value - qnorm(.975) * reg_se, 
  '97.5%' = reg_value + qnorm(.975) * reg_se) %>% 
  t() %>% 
  kable(.,
        format = 'html',
        digits = 2,
        align = 'rr',
        caption = '95% Confidence Interval')
```

<table>

<caption>

95% Confidence Interval

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

10.48

</td>

<td style="text-align:right;">

10.53

</td>

</tr>

</tbody>

</table>

### Create Forecast Interval

``` r
# forecast interval requires sigma^2

educ_f <- 16
exp_f <- 5

f_vector <- c(1, educ_f, exp_f, exp_f^2/100)

f_val <- t(basic_reg$coefficients) %*% f_vector %>% 
  as.vector()

s_2 <- mean(basic_reg$residuals^2)
s_2x <- sqrt(s_2 + t(f_vector) %*% V_HC1 %*% f_vector) %>% 
  as.vector()

c('2.5%' = f_val - qnorm(.9) * s_2x, 
  '97.5%' = f_val + qnorm(.9) * s_2x) %>% 
  t() %>% 
  kable(.,
        format = 'html',
        digits = 2,
        align = 'rr',
        caption = '80% Forecast Interval for Log(Wage)')
```

<table>

<caption>

80% Forecast Interval for Log(Wage)

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

9.76

</td>

<td style="text-align:right;">

11.27

</td>

</tr>

</tbody>

</table>

``` r
c('2.5%' = exp(f_val - qnorm(.9) * s_2x), 
  '97.5%' = exp(f_val + qnorm(.9) * s_2x)) %>% 
  t() %>% 
  kable(.,
        format = 'html',
        digits = 2,
        align = 'rr',
        caption = '80% Forecast Interval for Wage')
```

<table>

<caption>

80% Forecast Interval for Wage

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

17300.83

</td>

<td style="text-align:right;">

78303.46

</td>

</tr>

</tbody>

</table>

## Hanson Problem 10.30

### Get Jackknife and Bootstrap Estimates

``` r
data_reg_2 <- data_reg %>% 
  filter(region == 2, 
         marital == 7) %>% 
  mutate(id = row_number())

n <- nrow(data_reg_2)

theta_jk <- map_dbl(c(1:n), function(x){
  reg_jk <- lm(log_wage ~ education + exp + exp_2,
               data = data_reg_2 %>% 
                 filter(id != x))
  
  b1 <- reg_jk$coefficients['education']
  b2 <- reg_jk$coefficients['exp']
  b3 <- reg_jk$coefficients['exp_2']
  
  theta_hat <- b1 / (b2 + b3 / 5)
  
  return(theta_hat)
})

var_jk <- ((n-1)/n) * sum((theta_jk - mean(theta_jk))^2)
se_jk <- sqrt(var_jk)

B <- 1000

theta_boot <- map_dbl(c(1:B), function(x){
  reg_boot <- lm(log_wage ~ education + exp + exp_2,
                 data = data_reg_2 %>% 
                   dplyr::sample_n(size = nrow(data_reg_2),
                                   replace = TRUE))
  
  b1 <- reg_boot$coefficients['education']
  b2 <- reg_boot$coefficients['exp']
  b3 <- reg_boot$coefficients['exp_2']
  
  theta_hat <- b1 / (b2 + b3 / 5)
  
  return(theta_hat)
})

var_boot <- (1/(B-1)) * sum((theta_boot - mean(theta_boot))^2)
se_boot <- sqrt(var_boot)

data.table('Theta' = theta_hat,
           'SE Asymptotic' = se_theta,
           'SE Jackknife' = se_jk,
           'SE Bootstrap' = se_boot) %>% 
  kable(.,
        format = 'html',
        digits = 2,
        align = 'lrrr',
        caption = 'Theta and SEs')
```

<table>

<caption>

Theta and SEs

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

3.51

</td>

<td style="text-align:right;">

0.23

</td>

<td style="text-align:right;">

0.89

</td>

<td style="text-align:right;">

1.5

</td>

</tr>

</tbody>

</table>

### CI using BC Percentile method

``` r
p_star <- mean(theta_boot >= mean(theta_boot))
z_0 <- qnorm(p_star)

x_alpha <- function(alpha){
  return(pnorm(qnorm(alpha) + 2*z_0))
}

quantile(theta_boot, c(x_alpha(.025), 
                       x_alpha(.975))) %>% 
  t() %>% 
  kable(.,
        format = 'html',
        digits = 2,
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

0.6918195%

</th>

<th style="text-align:right;">

92.7641%

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:right;">

1.74

</td>

<td style="text-align:right;">

5

</td>

</tr>

</tbody>

</table>