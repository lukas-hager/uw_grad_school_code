function [residual, g1] = dynamic_resid_g1(T, y, x, params, steady_state, it_, T_flag)
% function [residual, g1] = dynamic_resid_g1(T, y, x, params, steady_state, it_, T_flag)
%
% Wrapper function automatically created by Dynare
%

    if T_flag
        T = problem_2_3c.dynamic_g1_tt(T, y, x, params, steady_state, it_);
    end
    residual = problem_2_3c.dynamic_resid(T, y, x, params, steady_state, it_, false);
    g1       = problem_2_3c.dynamic_g1(T, y, x, params, steady_state, it_, false);

end
