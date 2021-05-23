clearvars;
clc;

% define parameters

beta = .96;
epsilon = 10^-4;
gamma = 1.5;
rho = .9;
sigma = .2;
alpha = .36;
delta = .1;
N = 7;
m = 2.5;

% use tauchen's method to discretize

[l_log, l_prob] = tauchen(N, 0, rho, sigma, m);

% translate nodes from log

l_grid = exp(l_log);

% get the invariant distribution of labor

l_dist = mc_invdist(l_prob);

% get the expectation of labor

l_mean = dot(l_grid,l_dist);
fprintf('Q1.1: E[Labor] = %f\n', l_mean);

% we can use the euler equation to pin down r

r = 1 / beta + delta - 1;

% we can use r to pin down K using r = MPK

k_mean = (r / (l_mean^(1-alpha) * alpha))^(1/(alpha - 1));
fprintf('Q1.2: E[Capital] = %f\n', k_mean);

% then we can pin down w using w = MPL

w = (1-alpha)*l_mean^(-alpha)*k_mean^(alpha);

% define grid

a_grid = linspace(0, 50, 150);

[l,a] = meshgrid(l_grid,a_grid);

s_dim = max(size(l_grid))*max(size(a_grid));

s = [reshape(a, [s_dim, 1]), reshape(l, [s_dim, 1])];

% iterate to find convergence

k = k_mean;
crit = 1;
while crit > .001
    % define empty matrices for policy and value functions
    V_old = zeros(max(size(a_grid)),max(size(l_grid)));
    V_new = zeros(max(size(a_grid)),max(size(l_grid)));
    policy = zeros(max(size(a_grid)),max(size(l_grid)));

    % iterate over grid

    max_diff = 1;
    while max_diff > (1-beta)*epsilon
        for i = 1:s_dim

            % calculate c
            a_i = a(i);
            l_j = l(i);
            c = (1+r)*a_i + w * l_j - a_grid;

            % get the indices of positive c values
            pos_c = find(c >= 0);

            % calculate u
            u = c(pos_c).^(1-gamma)/(1-gamma);

            % get the expectations
            l_loc = find(l_j == l_grid);
            E = l_prob(l_loc, :)*V_old(pos_c, :)';

            all_V = u + beta .* E;
            max_V = max(all_V);

            % update the value function
            V_new(i) = max_V;

            % update the policy function
            policy(i) = a_grid(pos_c(all_V == max_V));

        end

        max_diff = max(abs(V_new - V_old), [], 'all');
        V_old = V_new;

    end
    
    % can we create a clever transition matrix as such: create
    % two values, one which has values 1 and 0 for policy
    % function matching, and one which has probabilities of transitioning,
    % and then do elementwise products to get the full transition
    % matrix

    transition_mat = zeros(s_dim, s_dim);
    for i = 1:s_dim
       for j = 1:s_dim 
           % get the transition probability
           i_dim = find(s(i,2) == l_grid);
           j_dim = find(s(j,2) == l_grid);
           s_prob = l_prob(i_dim, j_dim);

           % get the policy "probability" for i
           chosen_level = policy(mod(i-1,150) + 1, floor((i-1)/150) + 1);
           actual_level = s(j,1);

           % update transition matrix
           transition_mat(i,j) = s_prob * (chosen_level == actual_level);
       end
    end

    % get aggregate savings
    stat_dist = mc_invdist(transition_mat);
    savings = sum(policy .* reshape(stat_dist', 150, 7), 'all');

    % compute criterion and update k,w,r
    crit = abs((k - savings) / k);
    k = k + .1*(savings - k);
    w = (1-alpha)*l_mean^(-alpha)*k^(alpha);
    r = alpha*l_mean^(1-alpha)*k^(alpha-1);
    fprintf('Capital Guess: %i\n', k);
end

fprintf('Q1.3: Aiyagari Capital = %f\n', k);
fprintf('Q1.3: Aiyagari Wage = %f\n', w);
fprintf('Q1.3: Aiyagari Interest Rate = %f\n', r);

% plot the value functions
figure;
hold on
for val = 1:7
    plot(a_grid, V_new(:, val));
end
xlabel('a')
ylabel('V(a,z)')
title('Value Function')
hold off

% plot the policy functions
figure;
hold on
for val = 1:7
    plot(a_grid, policy(:, val));
end
xlabel('a')
ylabel('a''(a,z)')
title('Policy Function')
hold off

% plot density as heatmap
heatmap(reshape(stat_dist', 150, 7))