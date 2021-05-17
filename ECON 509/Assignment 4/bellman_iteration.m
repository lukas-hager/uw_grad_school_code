clearvars;

clc;

% define parameters

r = .01;
beta = .97;
gamma = 1.5;
epsilon = 10^(-4);

transition_mat = [.75, .25; .2, .8];

% define grid

a_grid = linspace(0, 50, 250);
z_grid = [.5, 1];

[a,z] = meshgrid(a_grid,z_grid);

% define empty matrices for policy and value functions
V_old = zeros(2,250);
V_new = zeros(2,250);
policy = zeros(2,250);

% iterate over grid

max_diff = 1;
while max_diff > epsilon
    for i = 1:500
    
        % calculate c
        a_i = a(i);
        z_j = z(i);
        c = (1+r)*a_i + z_j - a_grid;

        % get the indices of positive c values
        pos_c = find(c >= 0);

        % calculate u
        u = c(pos_c).^(1-gamma)/(1-gamma);

        % get the expectations
        z_loc = find(ismember(z_grid, z_j));
        E = transition_mat(z_loc, :)*V_old(:, pos_c);

        all_V = u + beta .* E;
        max_V = max(all_V);

        % update the value function
        V_new(i) = max_V;

        % update the policy function
        policy(i) = a_grid(pos_c(all_V == max_V));

    end

    max_diff = max(abs(V_new - V_old), [], 'all');
    fprintf('Maximum Difference: %i\n', max_diff);
    V_old = V_new;
    
end

% plot the policy and value functions

tiledlayout(2,1)
nexttile
plot(a_grid, V_new(1,:),a_grid,V_new(2,:))
legend({'z=.5', 'z=1'}, 'Location', 'northwest')
xlabel('a')
ylabel('V(a,z)')
title('Value Function')
nexttile
plot(a_grid, policy(1,:),a_grid,policy(2,:),a_grid,a_grid, '--')
legend({'z=.5', 'z=1', '45 Degrees'}, 'Location', 'northwest')
xlabel('a')
ylabel('a(a,z)')
title('Policy Function');
saveas(gcf,'policy_and_value_functions.png');

% write functions

value_df = array2table(V_new','RowNames',string(a_grid),'VariableNames',string(z_grid));
writetable(value_df, 'value_df.xlsx')

policy_df = array2table(policy','RowNames',string(a_grid),'VariableNames',string(z_grid));
writetable(policy_df, 'policy_df.xlsx')