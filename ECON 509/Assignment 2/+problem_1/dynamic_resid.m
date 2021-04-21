function residual = dynamic_resid(T, y, x, params, steady_state, it_, T_flag)
% function residual = dynamic_resid(T, y, x, params, steady_state, it_, T_flag)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T             [#temp variables by 1]     double   vector of temporary terms to be filled by function
%   y             [#dynamic variables by 1]  double   vector of endogenous variables in the order stored
%                                                     in M_.lead_lag_incidence; see the Manual
%   x             [nperiods by M_.exo_nbr]   double   matrix of exogenous variables (in declaration order)
%                                                     for all simulation periods
%   steady_state  [M_.endo_nbr by 1]         double   vector of steady state values
%   params        [M_.param_nbr by 1]        double   vector of parameter values in declaration order
%   it_           scalar                     double   time period for exogenous variables for which
%                                                     to evaluate the model
%   T_flag        boolean                    boolean  flag saying whether or not to calculate temporary terms
%
% Output:
%   residual
%

if T_flag
    T = problem_1.dynamic_resid_tt(T, y, x, params, steady_state, it_);
end
residual = zeros(6, 1);
lhs = params(3)/y(6)*T(1);
rhs = T(2);
residual(1) = lhs - rhs;
lhs = T(2);
rhs = params(1)*(1+y(10)-params(5))*T(3);
residual(2) = lhs - rhs;
lhs = y(9);
rhs = y(6)*y(4)+(1+y(7)-params(5))*y(3)-y(2);
residual(3) = lhs - rhs;
lhs = y(7);
rhs = T(5)*T(6);
residual(4) = lhs - rhs;
lhs = y(6);
rhs = T(8)*T(9);
residual(5) = lhs - rhs;
lhs = log(y(5));
rhs = params(7)*log(y(1))+x(it_, 1);
residual(6) = lhs - rhs;

end
