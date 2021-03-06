%
% Status : main Dynare file
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

if isoctave || matlab_ver_less_than('8.6')
    clear all
else
    clearvars -global
    clear_persistent_variables(fileparts(which('dynare')), false)
end
tic0 = tic;
% Define global variables.
global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info estimation_info ys0_ ex0_
options_ = [];
M_.fname = 'problem_1';
M_.dynare_version = '4.6.4';
oo_.dynare_version = '4.6.4';
options_.dynare_version = '4.6.4';
%
% Some global variables initialization
%
global_initialization;
diary off;
diary('problem_1.log');
M_.exo_names = cell(1,1);
M_.exo_names_tex = cell(1,1);
M_.exo_names_long = cell(1,1);
M_.exo_names(1) = {'epx'};
M_.exo_names_tex(1) = {'epx'};
M_.exo_names_long(1) = {'epx'};
M_.endo_names = cell(6,1);
M_.endo_names_tex = cell(6,1);
M_.endo_names_long = cell(6,1);
M_.endo_names(1) = {'c'};
M_.endo_names_tex(1) = {'c'};
M_.endo_names_long(1) = {'c'};
M_.endo_names(2) = {'k'};
M_.endo_names_tex(2) = {'k'};
M_.endo_names_long(2) = {'k'};
M_.endo_names(3) = {'l'};
M_.endo_names_tex(3) = {'l'};
M_.endo_names_long(3) = {'l'};
M_.endo_names(4) = {'z'};
M_.endo_names_tex(4) = {'z'};
M_.endo_names_long(4) = {'z'};
M_.endo_names(5) = {'w'};
M_.endo_names_tex(5) = {'w'};
M_.endo_names_long(5) = {'w'};
M_.endo_names(6) = {'r'};
M_.endo_names_tex(6) = {'r'};
M_.endo_names_long(6) = {'r'};
M_.endo_partitions = struct();
M_.param_names = cell(8,1);
M_.param_names_tex = cell(8,1);
M_.param_names_long = cell(8,1);
M_.param_names(1) = {'beta'};
M_.param_names_tex(1) = {'beta'};
M_.param_names_long(1) = {'beta'};
M_.param_names(2) = {'gamma'};
M_.param_names_tex(2) = {'gamma'};
M_.param_names_long(2) = {'gamma'};
M_.param_names(3) = {'phi'};
M_.param_names_tex(3) = {'phi'};
M_.param_names_long(3) = {'phi'};
M_.param_names(4) = {'eta'};
M_.param_names_tex(4) = {'eta'};
M_.param_names_long(4) = {'eta'};
M_.param_names(5) = {'delta'};
M_.param_names_tex(5) = {'delta'};
M_.param_names_long(5) = {'delta'};
M_.param_names(6) = {'alpha'};
M_.param_names_tex(6) = {'alpha'};
M_.param_names_long(6) = {'alpha'};
M_.param_names(7) = {'rho'};
M_.param_names_tex(7) = {'rho'};
M_.param_names_long(7) = {'rho'};
M_.param_names(8) = {'sigma'};
M_.param_names_tex(8) = {'sigma'};
M_.param_names_long(8) = {'sigma'};
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 1;
M_.endo_nbr = 6;
M_.param_nbr = 8;
M_.orig_endo_nbr = 6;
M_.aux_vars = [];
M_.Sigma_e = zeros(1, 1);
M_.Correlation_matrix = eye(1, 1);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = true;
M_.det_shocks = [];
options_.linear = false;
options_.block = false;
options_.bytecode = false;
options_.use_dll = false;
options_.linear_decomposition = false;
M_.orig_eq_nbr = 6;
M_.eq_nbr = 6;
M_.ramsey_eq_nbr = 0;
M_.set_auxiliary_variables = exist(['./+' M_.fname '/set_auxiliary_variables.m'], 'file') == 2;
M_.epilogue_names = {};
M_.epilogue_var_list_ = {};
M_.orig_maximum_endo_lag = 1;
M_.orig_maximum_endo_lead = 1;
M_.orig_maximum_exo_lag = 0;
M_.orig_maximum_exo_lead = 0;
M_.orig_maximum_exo_det_lag = 0;
M_.orig_maximum_exo_det_lead = 0;
M_.orig_maximum_lag = 1;
M_.orig_maximum_lead = 1;
M_.orig_maximum_lag_with_diffs_expanded = 1;
M_.lead_lag_incidence = [
 0 2 8;
 0 3 9;
 0 4 0;
 1 5 0;
 0 6 0;
 0 7 10;]';
M_.nstatic = 2;
M_.nfwrd   = 3;
M_.npred   = 1;
M_.nboth   = 0;
M_.nsfwrd   = 3;
M_.nspred   = 1;
M_.ndynamic   = 4;
M_.dynamic_tmp_nbr = [9; 1; 0; 0; ];
M_.model_local_variables_dynamic_tt_idxs = {
};
M_.equations_tags = {
  1 , 'name' , '1' ;
  2 , 'name' , '2' ;
  3 , 'name' , 'k' ;
  4 , 'name' , 'r' ;
  5 , 'name' , 'w' ;
  6 , 'name' , '6' ;
};
M_.mapping.c.eqidx = [1 2 3 ];
M_.mapping.k.eqidx = [3 4 5 ];
M_.mapping.l.eqidx = [1 3 4 5 ];
M_.mapping.z.eqidx = [4 5 6 ];
M_.mapping.w.eqidx = [1 3 5 ];
M_.mapping.r.eqidx = [2 3 4 ];
M_.mapping.epx.eqidx = [6 ];
M_.static_and_dynamic_models_differ = false;
M_.has_external_function = false;
M_.state_var = [4 ];
M_.exo_names_orig_ord = [1:1];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(6, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(1, 1);
M_.params = NaN(8, 1);
M_.endo_trends = struct('deflator', cell(6, 1), 'log_deflator', cell(6, 1), 'growth_factor', cell(6, 1), 'log_growth_factor', cell(6, 1));
M_.NNZDerivatives = [23; -1; -1; ];
M_.static_tmp_nbr = [8; 1; 0; 0; ];
M_.model_local_variables_static_tt_idxs = {
};
M_.params(1) = .98;
beta = M_.params(1);
M_.params(2) = 2.5;
gamma = M_.params(2);
M_.params(3) = .4;
phi = M_.params(3);
M_.params(4) = 2;
eta = M_.params(4);
M_.params(5) = .1;
delta = M_.params(5);
M_.params(6) = .35;
alpha = M_.params(6);
M_.params(7) = .95;
rho = M_.params(7);
M_.params(8) = .01;
sigma = M_.params(8);
%
% INITVAL instructions
%
options_.initval_file = false;
oo_.steady_state(3) = 1;
oo_.steady_state(2) = 1;
oo_.steady_state(1) = 1;
oo_.steady_state(5) = 1;
oo_.steady_state(6) = 1;
oo_.steady_state(4) = 1;
if M_.exo_nbr > 0
	oo_.exo_simul = ones(M_.maximum_lag,1)*oo_.exo_steady_state';
end
if M_.exo_det_nbr > 0
	oo_.exo_det_simul = ones(M_.maximum_lag,1)*oo_.exo_det_steady_state';
end
%
% SHOCKS instructions
%
M_.exo_det_length = 0;
M_.Sigma_e(1, 1) = M_.params(8)^2;
steady;
options_.order = 1;
var_list_ = {};
[info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, var_list_);
save('problem_1_results.mat', 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save('problem_1_results.mat', 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save('problem_1_results.mat', 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save('problem_1_results.mat', 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save('problem_1_results.mat', 'estimation_info', '-append');
end
if exist('dataset_info', 'var') == 1
  save('problem_1_results.mat', 'dataset_info', '-append');
end
if exist('oo_recursive_', 'var') == 1
  save('problem_1_results.mat', 'oo_recursive_', '-append');
end


disp(['Total computing time : ' dynsec2hms(toc(tic0)) ]);
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
diary off
