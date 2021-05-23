% P = mc_invdist(PI). Computes the invariant distribution P of a Markov 
% chain with transition matrix PI

function P = mc_invdist(PI)

[V, D] = eig(PI');                          

ii = find(abs(diag(D) - 1) < 1E-8, 1);      % Find first unit eigenvalue               

P = V(:,ii) / sum(V(:,ii));                 % Normalize unit eigenvector

assert(max(abs(P' - P'*PI)) < 1E-12)        % Verify dist. is stationary

if sum(abs(diag(D) - 1) < 1E-8) > 1         % Check for uniqueness
   warning('P not unique')
end

end

