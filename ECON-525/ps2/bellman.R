# first value function
alpha = .3
beta = .6
A = 20
min = 0.05
max = 12
step_len = .05
delta = .5
k_grid = seq(min, max, by = step_len)
v = data.table('k_t' = k_grid, 'v' = rep(0, length(k_grid)))
v_old = data.table('k_t' = k_grid, 'v' = rep(0, length(k_grid)))
converge = 10^-5
converged <- FALSE

while(converged == FALSE){
  consump_df <- CJ('k_t' = k_grid, 'k_t1' = k_grid)
  consump_df[, c := A * k_t ^ alpha - k_t1 + k_t * (1-delta)]
  consump_df <- consump_df[c >= 0]
  consump_df[, u := log(c)]
  consump_df <- consump_df %>% 
    left_join(v, by = c('k_t1'= 'k_t')) %>% 
    data.table()
  
  value_fun <- copy(consump_df)
  value_fun <- value_fun[, .(v = max(u + beta * v)), by = k_t]
  
  v_old <<- v
  v <<- value_fun
  
  converge_dist <- max(abs(v_old$v - v$v))
  print(str_interp('Convergence distance: ${converge_dist}'))
  
  if(converge_dist < converge){
    converged <<- TRUE
  }
}

ggplot(data = v) + 
  geom_point(aes(x = k_t, y =v))

################

alpha = .3
beta = .6
A = 1
min = .05
max = 12
step_len = .05
k_grid = seq(min, max, by = step_len)
v = data.table('k_t' = k_grid, 'v' = rep(0, length(k_grid)))
v_old = data.table('k_t' = k_grid, 'v' = rep(0, length(k_grid)))
converge = 10^-5
converged <- FALSE

while(converged == FALSE){
  consump_df <- CJ('k_t' = k_grid, 'k_t1' = k_grid)
  consump_df[, c := A * k_t ^ alpha - k_t1]
  consump_df <- consump_df[c >= 0]
  consump_df[, u := log(c)]
  consump_df <- consump_df %>% 
    left_join(v, by = c('k_t1'= 'k_t')) %>% 
    data.table()
  
  value_fun <- copy(consump_df)
  value_fun <- value_fun[, .(v = max(u + beta * v)), by = k_t]
  
  v_old <<- v
  v <<- value_fun
  
  converge_dist <- max(abs(v_old$v - v$v))
  print(str_interp('Convergence distance: ${converge_dist}'))
  
  if(converge_dist < converge){
    converged <<- TRUE
  }
}

true_value_fun <- function(x){
  a <- alpha/((1-alpha * beta))
  b <- (1/(1-beta)) * (log(1-alpha *beta) + 
                         (alpha * beta / (1- alpha * beta)) * log(alpha *beta) + 
                         (1 / (1-alpha * beta)) * log(A))
  return(a * log(x) + b)
}

ggplot(data = v) + 
  geom_point(aes(x = k_t, y =v)) + 
  stat_function(fun = true_value_fun, color = 'red')

######## 

while(converged == FALSE){
  x_df <- CJ('x_t' = c(0:10), 'i_t' = c(0,1))
  consump_df <- CJ('k_t' = k_grid, 'k_t1' = k_grid)
  consump_df[, c := A * k_t ^ alpha - k_t1 + k_t * (1-delta)]
  consump_df <- consump_df[c >= 0]
  consump_df[, u := log(c)]
  consump_df <- consump_df %>% 
    left_join(v, by = c('k_t1'= 'k_t')) %>% 
    data.table()
  
  value_fun <- copy(consump_df)
  value_fun <- value_fun[, .(v = max(u + beta * v)), by = k_t]
  
  v_old <<- v
  v <<- value_fun
  
  converge_dist <- max(abs(v_old$v - v$v))
  print(str_interp('Convergence distance: ${converge_dist}'))
  
  if(converge_dist < converge){
    converged <<- TRUE
}
  
  
beta = .95
theta1 = .3
theta2 = 0
theta3 = 4
probs <- evd::qgumbel(seq(.05,.95, by = .1))

x_df <- CJ('x_t' = c(0:10), 'i_t' = c(0,1)) %>% 
  mutate(lambda = rbinom(22, 1, .8))



v_new <- CJ('x_t' = c(0:10), 'ep_0' = probs) %>% 
  left_join(CJ('x_t' = c(0:10), 'ep_1' = probs),
            by = 'x_t') %>% 
  left_join(CJ('x_t' = c(0:10), 'i_t' = c(0,1)),
               by = 'x_t') %>% 
  left_join(v_old_join,
            by = c('x_t', 'i_t')) %>% 
  mutate(v = ifelse(i_t == 0, 
                    -theta1 * x_t - theta2 * x_t^2 + ep_0,
                    -theta3 + ep_1),
         v = v + .95 * e_v) %>% 
  group_by(x_t, ep_0, ep_1) %>% 
  filter(v == max(v)) %>% 
  ungroup() %>% 
  group_by(x_t) %>% 
  summarise(v = mean(v),
            replacement_prob = mean(i_t)) %>% 
  ungroup()




converged <- FALSE
converge = 10^-5

v_old <- data.table('x_t' = c(0:10), 'v' = rep(0,11))
v_old_join <- CJ('x_t' = c(0:10), 'i_t' = c(0,1)) %>% 
  mutate(e_v = 0)

while(converged == FALSE){
  v_new <- CJ('x_t' = c(0:10), 'ep_0' = probs) %>% 
    left_join(CJ('x_t' = c(0:10), 'ep_1' = probs),
              by = 'x_t') %>% 
    left_join(CJ('x_t' = c(0:10), 'i_t' = c(0,1)),
              by = 'x_t') %>% 
    left_join(v_old_join,
              by = c('x_t', 'i_t')) %>% 
    mutate(v = ifelse(i_t == 0, 
                      -theta1 * x_t - theta2 * x_t^2 + ep_0,
                      -theta3 + ep_1),
           v = v + .95 * e_v) %>% 
    group_by(x_t, ep_0, ep_1) %>% 
    filter(v == max(v)) %>% 
    ungroup() %>% 
    group_by(x_t) %>% 
    summarise(v = mean(v),
              replacement_prob = mean(i_t)) %>% 
    ungroup()
  
  converge_dist <- v_new %>% 
    left_join(v_old %>% 
                rename(v_old = v),
              by = 'x_t') %>% 
    mutate(diff = abs(v-v_old)) %>% 
    pull(diff) %>% 
    max()
  
  v_old <<- v_new
  
  v_old_join <- CJ('x_t' = c(0:10), 'i_t' = c(0,1)) %>% 
    mutate(poss = pmin(x_t + i_t, 10),
           lambda = rep(c(.2,.8),11)) %>% 
    select(-i_t) %>% 
    left_join(v_new,
              by = c('poss' = 'x_t')) %>% 
    group_by(x_t) %>% 
    summarise(e_v = sum(lambda * v)) %>% 
    ungroup() %>% 
    mutate(i_t = 0) %>%
    bind_rows(
      CJ('x_t' = c(0:10), 'poss' = c(0,1)) %>% 
        left_join(v_new,
                  by = c('poss' = 'x_t')) %>% 
        mutate(lambda = rep(c(.2,.8),11)) %>% 
        group_by(x_t) %>% 
        summarise(e_v = sum(lambda * v)) %>% 
        ungroup() %>% 
        mutate(i_t = 1)
    ) %>% 
    arrange(x_t, i_t)

  print(str_interp('Convergence distance: ${converge_dist}'))
  
  if(converge_dist < converge){
    converged <<- TRUE
  }
}

converged <- FALSE
converge = 10^-5

v_old <- data.table('x_t' = c(0:10),
                    'v_0' = rep(0,11),
                    'v_1' = rep(0,11))

while(converged == FALSE){
  v_new_0 <- CJ('x_t' = c(0:10), 'x_t1' = c(0,1)) %>% 
    mutate(lambda = ifelse(x_t1 == 1, .8, .2),
           u = -theta1 * x_t - theta2 * x_t^2) %>% 
    left_join(v_old,
              by = c('x_t1' = 'x_t')) %>% 
    left_join(CJ('x_t1' = c(0:10), 'ep_0' = probs),
              by = 'x_t1') %>% 
    left_join(CJ('x_t1' = c(0:10), 'ep_1' = probs),
              by = 'x_t1') %>% 
    mutate(v0_t1 = v_0 + ep_0,
           v1_t1 = v_1 + ep_1,
           max_v_t1 = pmax(v0_t1,v1_t1)) %>% 
    group_by(x_t, u) %>%
    summarise(e_v = mean(lambda * max_v_t1)) %>% 
    ungroup() %>% 
    mutate(v_0 = u + beta * e_v)
  
  v_new_1 <- CJ('x_t' = c(0:10), 'x_t1' = c(0,1)) %>% 
    mutate(lambda = ifelse(x_t1 == 1, .8, .2),
           x_t1 = pmin(x_t1 + x_t, 10),
           u = -theta3) %>% 
    left_join(v_old,
              by = c('x_t1' = 'x_t')) %>% 
    left_join(CJ('x_t1' = c(0:10), 'ep_0' = probs),
              by = 'x_t1') %>% 
    left_join(CJ('x_t1' = c(0:10), 'ep_1' = probs),
              by = 'x_t1') %>% 
    mutate(v0_t1 = v_0 + ep_0,
           v1_t1 = v_1 + ep_1,
           max_v_t1 = pmax(v0_t1,v1_t1)) %>% 
    group_by(x_t, x_t1) %>%
    summarise(e_v = lambda * mean(max_v_t1)) %>% 
    ungroup() %>% 
    mutate(u = -theta3) %>% 
    group_by(x_t) %>% 
    summarise(v_1 = u + beta * e_v)
  
  v_new <- v_new_0 %>% 
    select(x_t, v_0) %>% 
    left_join(v_new_1 %>% 
                select(x_t, v_1),
              by = 'x_t')
  
  converge_dist <- v_new %>% 
    left_join(v_old,
              by = 'x_t') %>% 
    mutate(diff = pmax(abs(v_1.x-v_1.y), abs(v_0.x-v_0.y))) %>% 
    pull(diff) %>% 
    max()
  
  v_old <<- v_new
  
  print(str_interp('Convergence distance: ${converge_dist}'))
  
  if(converge_dist < converge){
    converged <<- TRUE
  }
}

beta = .95
theta1 = .3
theta2 = 0
theta3 = 4
options(dplyr.summarise.inform = FALSE)

v_old <- data.table('x_t' = c(0:10),
                    'v_0' = rep(0,11),
                    'v_1' = rep(0,11))

mu = .5772

converged <- FALSE
converge = 10^-5

while(converged == FALSE){
  v_new_0 <- CJ('x_t' = c(0:10), 'x_t1' = c(0,1)) %>% 
    mutate(lambda = ifelse(x_t1 == 1, .8, .2),
           x_t1 = pmin(x_t1 + x_t, 10),
           u = -theta1 * x_t - theta2 * x_t^2) %>% 
    left_join(v_old,
              by = c('x_t1' = 'x_t')) %>% 
    mutate(ev_term = lambda * (mu + log(exp(v_0) + exp(v_1)))) %>% 
    group_by(x_t, u) %>% 
    summarise(ev_term = sum(ev_term)) %>% 
    ungroup() %>% 
    mutate(v_0 = u + beta * ev_term) %>% 
    select(x_t, v_0)
  
  v_new_1 <- CJ('x_t' = c(0:10), 'x_t1' = c(0,1)) %>% 
    mutate(lambda = ifelse(x_t1 == 1, .8, .2),
           u = -theta3) %>% 
    left_join(v_old,
              by = c('x_t1' = 'x_t')) %>% 
    mutate(ev_term = lambda * (mu + log(exp(v_0) + exp(v_1)))) %>% 
    group_by(x_t, u) %>% 
    summarise(ev_term = sum(ev_term)) %>% 
    ungroup() %>% 
    mutate(v_1 = u + beta * ev_term) %>% 
    select(x_t, v_1)
  
  v_new <- v_new_0 %>% 
    left_join(v_new_1,
              by = 'x_t')
  
  converge_dist <- v_new %>% 
    left_join(v_old,
              by = 'x_t') %>% 
    mutate(diff = pmax(abs(v_1.x-v_1.y), abs(v_0.x-v_0.y))) %>% 
    pull(diff) %>% 
    max()
  
  v_old <<- v_new
  
  print(str_interp('Convergence distance: ${converge_dist}'))
  
  if(converge_dist < converge){
    converged <<- TRUE
    v_final <- v_new %>% 
      mutate(p = exp(v_1) / (exp(v_0) + exp(v_1)))
  }
}


mu = .5772

converged <- FALSE
converge = 10^-5

v_old <- data.table('x_t' = c(0:10),
                    'v_0' = rep(0,11),
                    'v_1' = rep(0,11))
while(converged == FALSE){
  
  P_xt <- v_old %>% 
    mutate(p = 1 / (exp(v_0-v_1) + 1)) %>% 
    pull(p)
  
  e1 <- mu - log(P_xt)
  e0 <- mu - log(1-P_xt)
  
  Pi <- P_xt * (-theta3 + e1) + (1-P_xt) * (-theta1 * c(0:10) + e0)
  
  G_0 <- matrix(sapply(c(0:10), 
                       function(x){if(x != 10){
                         c(rep(0,x), .2, .8, rep(0, max(0,9-x)))
                       }else{
                         c(rep(0,10), 1)
                       }}),
                nrow = 11,
                ncol = 11,
                byrow = TRUE)
  
  G_1 <- matrix(rep(c(.2, .8, rep(0, 9)), 11), 
                nrow = 11, 
                ncol = 11, 
                byrow = TRUE)
  
  G <- diag(P_xt) %*% G_1 + diag(1-P_xt) %*% G_0
  
  V <- solve(diag(11) - beta * G) %*% Pi
  
  V_0 <- -theta1 * c(0:10) + beta * G_0 %*% V
  V_1 <- -theta3 + beta * G_1 %*% V
  
  v_new <- data.table('x_t' = c(0:10),
                      'v_0' = as.numeric(V_0),
                      'v_1' = as.numeric(V_1))
  
  converge_dist <- v_new %>% 
    left_join(v_old,
              by = 'x_t') %>% 
    mutate(diff = pmax(abs(v_1.x-v_1.y), abs(v_0.x-v_0.y))) %>% 
    pull(diff) %>% 
    max()
  
  v_old <<- v_new
  
  print(str_interp('Convergence distance: ${converge_dist}'))
  
  if(converge_dist < converge){
    converged <<- TRUE
    v_final <- v_new %>% 
      mutate(p = exp(v_1) / (exp(v_0) + exp(v_1)))
  }
}

P_xt <- v_old %>% 
  mutate(p = exp(v_1) / (exp(v_0) + exp(v_1))) %>% 
  pull(p)

P_xt <- v_final %>% 
  mutate(p = exp(v_1) / (exp(v_0) + exp(v_1))) %>% 
  pull(p)


e1 <- mu - log(P_xt)

e0 <- mu - log(1-P_xt)

Pi <- P_xt * (-theta3 + e1) + (1-P_xt) * (-theta1 * c(0:10) + e0)

G_0 <- matrix(sapply(c(0:10), 
                     function(x){if(x != 10){
                       c(rep(0,x), .2, .8, rep(0, max(0,9-x)))
                     }else{
                       c(rep(0,10), 1)
                     }}),
              nrow = 11,
              ncol = 11,
              byrow = TRUE)

G_1 <- matrix(rep(c(.2, .8, rep(0, 9)), 11), 
              nrow = 11, 
              ncol = 11, 
              byrow = TRUE)

G <- diag(P_xt) %*% G_1 + diag(1-P_xt) %*% G_0

V <- solve(diag(11) - beta * G) %*% Pi

V_0 <- -theta1 * c(0:10) + beta * G_0 %*% V
V_1 <- -theta3 + beta * G_1 %*% V

v_new <- data.table('x_t' = c(0:10),
                    'v_0' = as.numeric(V_0),
                    'v_1' = as.numeric(V_1))
v_old <- v_new



















  
v_new_0 <- CJ('x_t' = c(0:10), 'x_t1' = c(0,1)) %>% 
  mutate(lambda = ifelse(x_t1 == 1, .8, .2),
         u = -theta1 * x_t - theta2 * x_t^2) %>% 
  left_join(v_old,
            by = c('x_t1' = 'x_t')) %>% 
  left_join(CJ('x_t1' = c(0:10), 'ep_0' = probs),
            by = 'x_t1') %>% 
  left_join(CJ('x_t1' = c(0:10), 'ep_1' = probs),
            by = 'x_t1') %>% 
  mutate(v0_t1 = v_0 + ep_0,
         v1_t1 = v_1 + ep_1,
         max_v_t1 = pmax(v0_t1,v1_t1)) %>% 
  group_by(x_t, u) %>%
  summarise(e_v = mean(lambda * max_v_t1)) %>% 
  ungroup() %>% 
  mutate(v_0 = u + beta * e_v)

v_new_1 <- CJ('x_t' = c(0:10), 'x_t1' = c(0,1)) %>% 
  mutate(lambda = ifelse(x_t1 == 1, .8, .2),
         x_t1 = pmin(x_t1 + x_t, 10),
         u = -theta3) %>% 
  left_join(v_old,
            by = c('x_t1' = 'x_t')) %>% 
  left_join(CJ('x_t1' = c(0:10), 'ep_0' = probs),
            by = 'x_t1') %>% 
  left_join(CJ('x_t1' = c(0:10), 'ep_1' = probs),
            by = 'x_t1') %>% 
  mutate(v0_t1 = v_0 + ep_0,
         v1_t1 = v_1 + ep_1,
         max_v_t1 = pmax(v0_t1,v1_t1)) %>% 
  group_by(x_t, u) %>%
  summarise(e_v = mean(lambda * max_v_t1)) %>% 
  ungroup() %>% 
  mutate(v_1 = u + beta * e_v)

v_new <- v_new_0 %>% 
  select(x_t, v_0) %>% 
  left_join(v_new_1 %>% 
              select(x_t, v_1),
            by = 'x_t')

converge_dist <- v_new %>% 
  left_join(v_old,
            by = 'x_t') %>% 
  mutate(diff = pmax(abs(v_1.x-v_1.y), abs(v_0.x-v_0.y))) %>% 
  pull(diff) %>% 
  max()




  
  group_by(x_t, )
  mutate(v = u + .95 * e_v) %>% 
  group_by(x_t, ep_0, ep_1) %>% 
  filter(v == max(v)) %>% 
  ungroup() %>% 
  group_by(x_t) %>% 
  summarise(v = mean(v),
            replacement_prob = mean(i_t)) %>% 
  ungroup()
  
v_old_join <- CJ('x_t' = c(0:10), 'i_t' = c(0,1)) %>% 
  mutate(v = c(1:22))




v_final <- v_final %>% mutate(v = v_1 * p + (1-p) * v_0)

p <- v_final$p[11]

v0 <- v_final$v[1]
v1 <- v_final$v[2]
v10 <- v_final$v[11]

p * (-theta3 + mu - log(p) + beta * (.2 * v0 + .8 * v1)) + (1-p) * (mu - log(1-p) + beta * v10)

yuya_rvs <- fread('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/ECON 525/Problem Sets/Assignment 2/draw.csv') %>% 
  mutate(lambda = as.numeric(V2 >= .2))

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

lambda_hat <- sims %>% 
  filter(x_t != 10) %>% 
  mutate(val = (1-lag(i_t)) * (x_t - lag(x_t)) + lag(i_t) * x_t) %>% 
  summarise(lambda_hat = mean(val, na.rm = TRUE)) %>% 
  pull(lambda_hat)

solve_dp <- function(t1, t2, t3){
  mu = .5772
  
  converged <- FALSE
  converge = 10^-5
  
  v_old <- data.table('x_t' = c(0:10),
                      'v_0' = rep(0,11),
                      'v_1' = rep(0,11))
  
  while(converged == FALSE){
    
    P_xt <- v_old %>% 
      mutate(p = 1 / (exp(v_0-v_1) + 1)) %>% 
      pull(p)
    
    e1 <- mu - log(P_xt)
    e0 <- mu - log(1-P_xt)
    
    Pi <- P_xt * (-t3 + e1) + (1-P_xt) * (-t1 * c(0:10) -t2 * c(0:10)^2 + e0)
    
    G_0 <- matrix(sapply(c(0:10), 
                         function(x){if(x != 10){
                           c(rep(0,x), .2, .8, rep(0, max(0,9-x)))
                         }else{
                           c(rep(0,10), 1)
                         }}),
                  nrow = 11,
                  ncol = 11,
                  byrow = TRUE)
    
    G_1 <- matrix(rep(c(.2, .8, rep(0, 9)), 11), 
                  nrow = 11, 
                  ncol = 11, 
                  byrow = TRUE)
    
    G <- diag(P_xt) %*% G_1 + diag(1-P_xt) %*% G_0
    
    V <- solve(diag(11) - beta * G) %*% Pi
    
    V_0 <- -t1 * c(0:10) -t2 * c(0:10)^2 + beta * G_0 %*% V
    V_1 <- -t3 + beta * G_1 %*% V
    
    v_new <- data.table('x_t' = c(0:10),
                        'v_0' = as.numeric(V_0),
                        'v_1' = as.numeric(V_1))
    
    converge_dist <- v_new %>% 
      left_join(v_old,
                by = 'x_t') %>% 
      mutate(diff = pmax(abs(v_1.x-v_1.y), abs(v_0.x-v_0.y))) %>% 
      pull(diff) %>% 
      max()
    
    v_old <- v_new
    
    if(converge_dist < converge){
      converged <- TRUE
      v_final <- v_new %>% 
        mutate(p = exp(v_1) / (exp(v_0) + exp(v_1)))
    }
  }
  
  return(v_final)
}

v_final <- solve_dp(.3, 0, 4)

write_csv(v_final,
          '/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/ECON 525/Problem Sets/Assignment 2-3/v_final.csv')

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
  
    probs <- solve_dp(t1, t2, t3)
    
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

get_g <- function(t1, t2, t3){
  P_xt <- solve_dp(t1, t2, t3) %>% 
    pull(p)
  
  e1 <- mu - log(P_xt)
  e0 <- mu - log(1-P_xt)
  
  Pi <- P_xt * (-t3 + e1) + (1-P_xt) * (-t1 * c(0:10) -t2 * c(0:10)^2 + e0)
  
  G_0 <- matrix(sapply(c(0:10), 
                       function(x){if(x != 10){
                         c(rep(0,x), .2, .8, rep(0, max(0,9-x)))
                       }else{
                         c(rep(0,10), 1)
                       }}),
                nrow = 11,
                ncol = 11,
                byrow = TRUE)
  
  G_1 <- matrix(rep(c(.2, .8, rep(0, 9)), 11), 
                nrow = 11, 
                ncol = 11, 
                byrow = TRUE)
  
  G <- diag(P_xt) %*% G_1 + diag(1-P_xt) %*% G_0
  return(G)
}

x <- 0
v_final <- solve_dp(theta1, theta2, theta3)
lambda_hat <- sims %>% 
  filter(!(x_t == 10 & i_t == 0)) %>% 
  mutate(val = (1-lag(i_t)) * (x_t - lag(x_t)) + lag(i_t) * x_t) %>% 
  summarise(lambda_hat = mean(val, na.rm = TRUE)) %>% 
  pull(lambda_hat)

new_sims <- bind_rows(
  lapply(c(1:5000),function(i){
    if(i == 1){
      x_old <<- x
      return(c('i_t' = 0, 'x_t' = x))
    }else{
      lambda <- rbinom(1,1,lambda_hat)
      
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


G <- get_g(theta1, theta2, theta3)
P_xt <- solve_dp(theta1, theta2, theta3) %>% 
  pull(p)
ss_probs <- as.numeric(
  t(G) %*% eigen(t(G))$vectors[,1] / sum(eigen(t(G))$vectors[,1])
)

lr_replacement_sim <- mean(new_sims %>% pull(i_t))
lr_replacement_ss <- sum(ss_probs * P_xt)

G_new <- get_g(theta1, theta2, .9*theta3)
P_xt <- solve_dp(theta1, theta2, .9*theta3) %>% 
  pull(p)
ss_probs <- as.numeric(
  t(G_new) %*% eigen(t(G_new))$vectors[,1] / sum(eigen(t(G_new))$vectors[,1])
)

lr_replacement_ss <- sum(ss_probs * P_xt)



converged <- FALSE
converge = 10^-5

v_old <- data.table('x_t' = c(0:10),
                    'v_0' = rep(0,11),
                    'v_1' = rep(0,11))

while(converged == FALSE){
  
  P_xt <- v_old %>% 
    mutate(p = 1 / (exp(v_0-v_1) + 1)) %>% 
    pull(p)
  
  e1 <- mu - log(P_xt)
  e0 <- mu - log(1-P_xt)
  
  Pi <- P_xt * (-theta3 + e1) + (1-P_xt) * (-theta1 * c(0:10) + e0)
  
  G_0 <- matrix(sapply(c(0:10), 
                       function(x){if(x != 10){
                         c(rep(0,x), .2, .8, rep(0, max(0,9-x)))
                       }else{
                         c(rep(0,10), 1)
                       }}),
                nrow = 11,
                ncol = 11,
                byrow = TRUE)
  
  G_1 <- matrix(rep(c(.2, .8, rep(0, 9)), 11), 
                nrow = 11, 
                ncol = 11, 
                byrow = TRUE)
  
  G <- diag(P_xt) %*% G_1 + diag(1-P_xt) %*% G_0
  
  V <- solve(diag(11) - beta * G) %*% Pi
  
  V_0 <- -theta1 * c(0:10) + beta * G_0 %*% V
  V_1 <- -theta3 + beta * G_1 %*% V
  
  v_new <- data.table('x_t' = c(0:10),
                      'v_0' = as.numeric(V_0),
                      'v_1' = as.numeric(V_1))
  
  converge_dist <- v_new %>% 
    left_join(v_old,
              by = 'x_t') %>% 
    mutate(diff = pmax(abs(v_1.x-v_1.y), abs(v_0.x-v_0.y))) %>% 
    pull(diff) %>% 
    max()
  
  v_old <<- v_new
  
  if(converge_dist < converge){
    converged <<- TRUE
    v_final <- v_new %>% 
      mutate(p = exp(v_1) / (exp(v_0) + exp(v_1)))
  }
}

print(v_final)

