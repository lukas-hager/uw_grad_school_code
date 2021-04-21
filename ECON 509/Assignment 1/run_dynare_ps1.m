clearvars;  

clc;

% path to dynare
addpath('/Applications/Dynare/4.6.4/matlab')

% pick the path
cd '/Users/hlukas/Google Drive/Grad School/Spring/ECON 509/Assignments/Assignment 1'

% run dynare
dynare problem_2_1.mod;

% get index of pi and x variables
pi_index = find(strcmp(oo_.var_list, 'pi'));
x_index = find(strcmp(oo_.var_list, 'x'));

% get variance of pi and x
v_pi = oo_.var(pi_index,pi_index);
v_x = oo_.var(x_index,x_index);

% compute welfare
welfare_2_1 = -(v_pi + .1 * v_x);

% print welfare
fprintf('Welfare for Problem 2.1 is %f', welfare_2_1);

clearvars;

% run dynare
dynare problem_2_2.mod;

% get index of pi and x variables
pi_index = find(strcmp(oo_.var_list, 'pi'));
x_index = find(strcmp(oo_.var_list, 'x'));

% get variance of pi and x
v_pi = oo_.var(pi_index,pi_index);
v_x = oo_.var(x_index,x_index);

% compute welfare
welfare_2_2 = -(v_pi + .1 * v_x);

% print welfare
fprintf('Welfare for Problem 2.2 is %f', welfare_2_2);

clearvars;

% run dynare
dynare problem_2_3a.mod;

% get index of pi and x variables
pi_index = find(strcmp(oo_.var_list, 'pi'));
x_index = find(strcmp(oo_.var_list, 'x'));

% get variance of pi and x
v_pi = oo_.var(pi_index,pi_index);
v_x = oo_.var(x_index,x_index);

% compute welfare
welfare_2_3a = -(v_pi + .1 * v_x);

% print welfare
fprintf('Welfare for Problem 2.3a is %f', welfare_2_3a);

clearvars;

% run dynare
dynare problem_2_3b.mod;

% get index of pi and x variables
pi_index = find(strcmp(oo_.var_list, 'pi'));
x_index = find(strcmp(oo_.var_list, 'x'));

% get variance of pi and x
v_pi = oo_.var(pi_index,pi_index);
v_x = oo_.var(x_index,x_index);

% compute welfare
welfare_2_3b = -(v_pi + .1 * v_x);

% print welfare
fprintf('Welfare for Problem 2.3b is %f', welfare_2_3b);

clearvars;

% run dynare
dynare problem_2_3c.mod;

% get index of pi and x variables
pi_index = find(strcmp(oo_.var_list, 'pi'));
x_index = find(strcmp(oo_.var_list, 'x'));

% get variance of pi and x
v_pi = oo_.var(pi_index,pi_index);
v_x = oo_.var(x_index,x_index);

% compute welfare
welfare_2_3c = -(v_pi + .1 * v_x);

% print welfare
fprintf('Welfare for Problem 2.3c is %f', welfare_2_3c);
