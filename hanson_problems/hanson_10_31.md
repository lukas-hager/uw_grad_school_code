Hanson Problem 10.31
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
library(openxlsx) # get excel files
library(scales) # for percentage formatting
library(MASS) # some statistics package
library(tidyverse) # important for data manipulation, includes dplyr
```

## Hanson Problem 4.28

### Get OLS Estimates with Robust and Clustered SEs

``` r
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
        format = 'html',
        digits = 4,
        align = 'lrrr',
        caption = 'Regression with Robust and Clustered SEs')
```

<table>

<caption>

Regression with Robust and Clustered SEs

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

SE Robust

</th>

<th style="text-align:right;">

SE Clustered

</th>

<th style="text-align:left;">

SE Difference

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

(Intercept)

</td>

<td style="text-align:right;">

\-0.7291

</td>

<td style="text-align:right;">

0.0810

</td>

<td style="text-align:right;">

0.1297

</td>

<td style="text-align:left;">

\-0.0488

</td>

</tr>

<tr>

<td style="text-align:left;">

tracking

</td>

<td style="text-align:right;">

0.1725

</td>

<td style="text-align:right;">

0.0240

</td>

<td style="text-align:right;">

0.0762

</td>

<td style="text-align:left;">

\-0.0522

</td>

</tr>

<tr>

<td style="text-align:left;">

agetest

</td>

<td style="text-align:right;">

\-0.0408

</td>

<td style="text-align:right;">

0.0085

</td>

<td style="text-align:right;">

0.0133

</td>

<td style="text-align:left;">

\-0.0048

</td>

</tr>

<tr>

<td style="text-align:left;">

girl

</td>

<td style="text-align:right;">

0.0812

</td>

<td style="text-align:right;">

0.0241

</td>

<td style="text-align:right;">

0.0285

</td>

<td style="text-align:left;">

\-0.0044

</td>

</tr>

<tr>

<td style="text-align:left;">

etpteacher

</td>

<td style="text-align:right;">

0.1799

</td>

<td style="text-align:right;">

0.0237

</td>

<td style="text-align:right;">

0.0375

</td>

<td style="text-align:left;">

\-0.0138

</td>

</tr>

<tr>

<td style="text-align:left;">

percentile

</td>

<td style="text-align:right;">

0.0173

</td>

<td style="text-align:right;">

0.0004

</td>

<td style="text-align:right;">

0.0007

</td>

<td style="text-align:left;">

\-0.0003

</td>

</tr>

</tbody>

</table>

## Hanson Problem 10.31

### Get Bootstraps

``` r
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
        format = 'html',
        digits = 4,
        align = 'lrrr',
        caption = 'Regression with Robust, Clustered, and Bootstrapped SEs')
```

<table>

<caption>

Regression with Robust, Clustered, and Bootstrapped SEs

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

SE Robust

</th>

<th style="text-align:right;">

SE Clustered

</th>

<th style="text-align:left;">

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

\-0.7291

</td>

<td style="text-align:right;">

0.0810

</td>

<td style="text-align:right;">

0.1297

</td>

<td style="text-align:left;">

0.1274

</td>

</tr>

<tr>

<td style="text-align:left;">

tracking

</td>

<td style="text-align:right;">

0.1725

</td>

<td style="text-align:right;">

0.0240

</td>

<td style="text-align:right;">

0.0762

</td>

<td style="text-align:left;">

0.0730

</td>

</tr>

<tr>

<td style="text-align:left;">

agetest

</td>

<td style="text-align:right;">

\-0.0408

</td>

<td style="text-align:right;">

0.0085

</td>

<td style="text-align:right;">

0.0133

</td>

<td style="text-align:left;">

0.0133

</td>

</tr>

<tr>

<td style="text-align:left;">

girl

</td>

<td style="text-align:right;">

0.0812

</td>

<td style="text-align:right;">

0.0241

</td>

<td style="text-align:right;">

0.0285

</td>

<td style="text-align:left;">

0.0284

</td>

</tr>

<tr>

<td style="text-align:left;">

etpteacher

</td>

<td style="text-align:right;">

0.1799

</td>

<td style="text-align:right;">

0.0237

</td>

<td style="text-align:right;">

0.0375

</td>

<td style="text-align:left;">

0.0369

</td>

</tr>

<tr>

<td style="text-align:left;">

percentile

</td>

<td style="text-align:right;">

0.0173

</td>

<td style="text-align:right;">

0.0004

</td>

<td style="text-align:right;">

0.0007

</td>

<td style="text-align:left;">

0.0007

</td>

</tr>

</tbody>

</table>

### Estimate BC Alpha Percentiles for Coefficients

``` r
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
        format = 'html',
        digits = 4,
        align = 'lrrrr',
        caption = 'Bootstrapped CIs Using BC Alpha Method')
```

<table>

<caption>

Bootstrapped CIs Using BC Alpha Method

</caption>

<thead>

<tr>

<th style="text-align:left;">

Term

</th>

<th style="text-align:right;">

Low

</th>

<th style="text-align:right;">

High

</th>

<th style="text-align:right;">

Lower Bound

</th>

<th style="text-align:right;">

Upper Bound

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

(Intercept)

</td>

<td style="text-align:right;">

2.3618%

</td>

<td style="text-align:right;">

97.3549%

</td>

<td style="text-align:right;">

\-0.9952

</td>

<td style="text-align:right;">

\-0.4790

</td>

</tr>

<tr>

<td style="text-align:left;">

tracking

</td>

<td style="text-align:right;">

2.4385%

</td>

<td style="text-align:right;">

97.4372%

</td>

<td style="text-align:right;">

0.0256

</td>

<td style="text-align:right;">

0.3152

</td>

</tr>

<tr>

<td style="text-align:left;">

agetest

</td>

<td style="text-align:right;">

2.7569%

</td>

<td style="text-align:right;">

97.7370%

</td>

<td style="text-align:right;">

\-0.0661

</td>

<td style="text-align:right;">

\-0.0132

</td>

</tr>

<tr>

<td style="text-align:left;">

girl

</td>

<td style="text-align:right;">

3.0629%

</td>

<td style="text-align:right;">

97.9726%

</td>

<td style="text-align:right;">

0.0291

</td>

<td style="text-align:right;">

0.1412

</td>

</tr>

<tr>

<td style="text-align:left;">

etpteacher

</td>

<td style="text-align:right;">

1.9096%

</td>

<td style="text-align:right;">

96.7616%

</td>

<td style="text-align:right;">

0.1058

</td>

<td style="text-align:right;">

0.2487

</td>

</tr>

<tr>

<td style="text-align:left;">

percentile

</td>

<td style="text-align:right;">

2.4958%

</td>

<td style="text-align:right;">

97.4958%

</td>

<td style="text-align:right;">

0.0159

</td>

<td style="text-align:right;">

0.0186

</td>

</tr>

</tbody>

</table>
