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
M_.fname = 'problem_2_3b';
M_.dynare_version = '4.6.4';
oo_.dynare_version = '4.6.4';
options_.dynare_version = '4.6.4';
%
% Some global variables initialization
%
global_initialization;
diary off;
diary('problem_2_3b.log');
M_.exo_names = cell(2,1);
M_.exo_names_tex = cell(2,1);
M_.exo_names_long = cell(2,1);
M_.exo_names(1) = {'etap'};
M_.exo_names_tex(1) = {'etap'};
M_.exo_names_long(1) = {'etap'};
M_.exo_names(2) = {'etax'};
M_.exo_names_tex(2) = {'etax'};
M_.exo_names_long(2) = {'etax'};
M_.endo_names = cell(5,1);
M_.endo_names_tex = cell(5,1);
M_.endo_names_long = cell(5,1);
M_.endo_names(1) = {'pi'};
M_.endo_names_tex(1) = {'pi'};
M_.endo_names_long(1) = {'pi'};
M_.endo_names(2) = {'i'};
M_.endo_names_tex(2) = {'i'};
M_.endo_names_long(2) = {'i'};
M_.endo_names(3) = {'x'};
M_.endo_names_tex(3) = {'x'};
M_.endo_names_long(3) = {'x'};
M_.endo_names(4) = {'ep'};
M_.endo_names_tex(4) = {'ep'};
M_.endo_names_long(4) = {'ep'};
M_.endo_names(5) = {'ex'};
M_.endo_names_tex(5) = {'ex'};
M_.endo_names_long(5) = {'ex'};
M_.endo_partitions = struct();
M_.param_names = cell(10,1);
M_.param_names_tex = cell(10,1);
M_.param_names_long = cell(10,1);
M_.param_names(1) = {'beta'};
M_.param_names_tex(1) = {'beta'};
M_.param_names_long(1) = {'beta'};
M_.param_names(2) = {'gam'};
M_.param_names_tex(2) = {'gam'};
M_.param_names_long(2) = {'gam'};
M_.param_names(3) = {'lambda'};
M_.param_names_tex(3) = {'lambda'};
M_.param_names_long(3) = {'lambda'};
M_.param_names(4) = {'sigma'};
M_.param_names_tex(4) = {'sigma'};
M_.param_names_long(4) = {'sigma'};
M_.param_names(5) = {'rhop'};
M_.param_names_tex(5) = {'rhop'};
M_.param_names_long(5) = {'rhop'};
M_.param_names(6) = {'rhox'};
M_.param_names_tex(6) = {'rhox'};
M_.param_names_long(6) = {'rhox'};
M_.param_names(7) = {'sigmap'};
M_.param_names_tex(7) = {'sigmap'};
M_.param_names_long(7) = {'sigmap'};
M_.param_names(8) = {'sigmax'};
M_.param_names_tex(8) = {'sigmax'};
M_.param_names_long(8) = {'sigmax'};
M_.param_names(9) = {'phip'};
M_.param_names_tex(9) = {'phip'};
M_.param_names_long(9) = {'phip'};
M_.param_names(10) = {'phix'};
M_.param_names_tex(10) = {'phix'};
M_.param_names_long(10) = {'phix'};
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 2;
M_.endo_nbr = 5;
M_.param_nbr = 10;
M_.orig_endo_nbr = 5;
M_.aux_vars = [];
M_.Sigma_e = zeros(2, 2);
M_.Correlation_matrix = eye(2, 2);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = true;
M_.det_shocks = [];
options_.linear = false;
options_.block = false;
options_.bytecode = false;
options_.use_dll = false;
options_.linear_decomposition = false;
M_.orig_eq_nbr = 5;
M_.eq_nbr = 5;
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
 0 3 8;
 0 4 0;
 0 5 0;
 1 6 0;
 2 7 0;]';
M_.nstatic = 2;
M_.nfwrd   = 1;
M_.npred   = 2;
M_.nboth   = 0;
M_.nsfwrd   = 1;
M_.nspred   = 2;
M_.ndynamic   = 3;
M_.dynamic_tmp_nbr = [0; 0; 0; 0; ];
M_.model_local_variables_dynamic_tt_idxs = {
};
M_.equations_tags = {
  1 , 'name' , 'x' ;
  2 , 'name' , 'pi' ;
  3 , 'name' , '3' ;
  4 , 'name' , 'ep' ;
  5 , 'name' , 'ex' ;
};
M_.mapping.pi.eqidx = [2 3 ];
M_.mapping.i.eqidx = [3 ];
M_.mapping.x.eqidx = [1 ];
M_.mapping.ep.eqidx = [2 4 ];
M_.mapping.ex.eqidx = [3 5 ];
M_.mapping.etap.eqidx = [4 ];
M_.mapping.etax.eqidx = [5 ];
M_.static_and_dynamic_models_differ = false;
M_.has_external_function = false;
M_.state_var = [4 5 ];
M_.exo_names_orig_ord = [1:2];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(5, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(2, 1);
M_.params = NaN(10, 1);
M_.endo_trends = struct('deflator', cell(5, 1), 'log_deflator', cell(5, 1), 'growth_factor', cell(5, 1), 'log_growth_factor', cell(5, 1));
M_.NNZDerivatives = [13; -1; -1; ];
M_.static_tmp_nbr = [0; 0; 0; 0; ];
M_.model_local_variables_static_tt_idxs = {
};
M_.params(1) = 0.96;
beta = M_.params(1);
M_.params(2) = .1;
gam = M_.params(2);
M_.params(3) = 0.25;
lambda = M_.params(3);
M_.params(4) = 2;
sigma = M_.params(4);
M_.params(5) = 0.95;
rhop = M_.params(5);
M_.params(6) = 0.95;
rhox = M_.params(6);
M_.params(7) = .01;
sigmap = M_.params(7);
M_.params(8) = .005;
sigmax = M_.params(8);
M_.params(9) = 1.5;
phip = M_.params(9);
M_.params(10) = .5;
phix = M_.params(10);
%
% INITVAL instructions
%
options_.initval_file = false;
oo_.steady_state(4) = 0;
oo_.steady_state(5) = 0;
oo_.steady_state(2) = .5;
oo_.steady_state(3) = 0;
oo_.steady_state(1) = .5;
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
M_.Sigma_e(1, 1) = M_.params(7)^2;
M_.Sigma_e(2, 2) = M_.params(8)^2;
steady;
options_.order = 1;
var_list_ = {};
[info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, var_list_);
save('problem_2_3b_results.mat', 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save('problem_2_3b_results.mat', 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save('problem_2_3b_results.mat', 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save('problem_2_3b_results.mat', 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save('problem_2_3b_results.mat', 'estimation_info', '-append');
end
if exist('dataset_info', 'var') == 1
  save('problem_2_3b_results.mat', 'dataset_info', '-append');
end
if exist('oo_recursive_', 'var') == 1
  save('problem_2_3b_results.mat', 'oo_recursive_', '-append');
end


disp(['Total computing time : ' dynsec2hms(toc(tic0)) ]);
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
diary off
