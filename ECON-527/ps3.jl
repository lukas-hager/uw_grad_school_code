# to run OOB for Dr. Greaney, stealing this code from Ralf_Dietrich
# https://discourse.julialang.org/t/how-to-use-pkg-dependencies-instead-of-pkg-installed/36416/19

# Make sure all needed Pkg's are ready to go
neededPackages = [
    :QuantEcon, 
    :StatsBase,
    :Statistics,
    :BasicInterpolators,
    :LinearAlgebra,
    :Random,
    :Plots,
    :Printf,
    :Optim,
    :ForwardDiff,
    :LaTeXStrings
] 

using Pkg;
for neededpackage in neededPackages
    (String(neededpackage) in keys(Pkg.project().dependencies)) || Pkg.add(String(neededpackage))
    @eval using $neededpackage
end

# set the seed
Random.seed!(42069)

# set the output path
path = "/Users/hlukas/Google Drive/Grad School/2022-2023/Spring/ECON 527/Assignments/Assignment 3/"

############################################
# Speeding Up VFI
############################################

println("-------------------------------------------")
println("Speeding Up VFI")

# define parameters
a_min = 0.
a_max = 50.
r = .02
gamma = 1.5
beta = .96
rho = .9
sigma = .2
mu = 0.
n_states = 5
n_gridpoints = 100
c = 2.
epsilon = 10e-4

# discretize log income, get state values and transition matrix
z_discrete = QuantEcon.rouwenhorst(n_states,rho,sigma,mu)
Pi = z_discrete.p
z_vals = z_discrete.state_values

# discretize wealth
kappa = a_min .+ (a_max - a_min)*range(0,1,length=n_gridpoints).^c

# define the value function
function V(a_t_1, a_t, z_t, interp_obj_array) 
    # evaluate the value functions at the future value of wealth
    V_t_1 = [x(a_t_1) for x in interp_obj_array]
    flow_utility = ((1. + r)*a_t + exp(z_t) - a_t_1)^(1. - gamma) / (1. - gamma)
    continuation_utility = dot(V_t_1, Pi[findall(x->x==z_t, z_vals)[1], :])
    return flow_utility + beta * continuation_utility
end

function V_vec(policy, interp_obj_array)
    # evaluate the future value using the policy function
    nest = [interp_obj_array[i].(vec(policy)) for i in 1:5]
    V_t_1 = reshape(reduce(vcat,nest),(size(nest[1])[1],size(nest)[1]))
    # calculate flow utility given policy function
    flow_utility = ((1. + r)*kappa .+ exp.(transpose(z_vals)) .- policy).^(1. - gamma) ./ (1. - gamma)
    # calculate all combinations of the continuation Utility
    all_combos = [(V_t_1 * transpose(Pi))[(i-1)*n_gridpoints+1:i*n_gridpoints, i] for i in 1:5]
    continuation_utility = reshape(reduce(vcat,all_combos), (n_gridpoints,n_states))
    return flow_utility + beta .* continuation_utility
end

function V_vec_endog(inv_policy, V_old) 
    # evaluate the future value using the policy function
    flow_utility = ((1. + r)*inv_policy .+ exp.(transpose(z_vals)) .- kappa).^(1. - gamma) ./ (1. - gamma)
    # calculate all combinations of the continuation Utility
    continuation_utility = V_old * transpose(Pi)
    return flow_utility .+ beta .* continuation_utility
end

function bellman_iteration(V_old)
    global V_interp_array = [CubicSplineInterpolator(kappa, V_old[:,i]) for i in 1:5]
    global V_new = zeros(n_gridpoints, n_states)
    global policy = zeros(n_gridpoints, n_states)

    for z_ind in 1:n_states
        for a_ind in 1:n_gridpoints
            res = optimize(
                a_t_1 -> -V(a_t_1, kappa[a_ind], z_vals[z_ind], V_interp_array), 
                a_min, 
                minimum([a_max, (1. + r) * kappa[a_ind] + exp(z_vals[z_ind]) - .001])
            )
            V_new[a_ind, z_ind] = -res.minimum
            policy[a_ind, z_ind] = res.minimizer
        end
    end

    return V_new, policy, V_interp_array
end

function convergence(V_new, V_old)
    diff = maximum(abs.(V_new - V_old)) 
    if diff < epsilon*(1-beta)
        println("Converged in $(counter) iterations")
        return true
    else
        return false
    end
end

# value function interation
function howard(m)
    global converged = false
    global counter = 0

    @time while converged == false
        # update iteration counter
        global counter += 1

        # initialize to zero if it's the first iteration
        if counter == 1
            global V_old = zeros(n_gridpoints, n_states)
        end

        # run one bellman iteration
        global V_new, policy, V_interp_array = bellman_iteration(V_old)
        global converged = convergence(V_new, V_old)
        global V_old = V_new

        # run m policy iterations
        for l in 1:m
            global V_interp_array = [CubicSplineInterpolator(kappa, V_old[:,i]) for i in 1:5]
            global V_new = V_vec(policy, V_interp_array)
            global V_old = V_new
        end
    end
end

println("Howard Policy Iteration; m = 5")
howard(5)
println("Howard Policy Iteration; m = 10")
howard(10)
println("Howard Policy Iteration; m = 20")
howard(20)

function d_v(spline, x)
    i = findcell(x, spline)
    ξ = x - spline.r.x[i]
    return spline.b[i] + 2 * spline.c[i]*ξ + 3 * spline.d[i]*ξ^2
end

function endog_grid(spline_arr)
    nest = [[d_v(spline_arr[i], x) for x in kappa] for i in 1:5]
    e_d_v = reshape(reduce(vcat,nest),(size(nest[1])[1],size(nest)[1])) * transpose(Pi)
    return (kappa .- exp.(transpose(z_vals)) + (beta * e_d_v).^(-1. /gamma)) ./ (1. + r)
end

function egm()
    global converged = false
    global counter = 0

    @time while converged == false
        # update iteration counter
        global counter += 1

        # initialize to zero if it's the first iteration
        if counter == 1
            global V_old = ((1+r).*hcat(kappa, kappa, kappa, kappa, kappa).+exp.(transpose(z_vals))).^(1-gamma)./((1-gamma) * (1-beta))
        end

        # compute endogeneous gridpoints
        global V_interp_array = [CubicSplineInterpolator(kappa, V_old[:,i]) for i in 1:5]
        global eg = endog_grid(V_interp_array)
        global V_new_endog = V_vec_endog(eg, V_old)
        global V_interp_array_endog = [CubicSplineInterpolator(eg[:,i][sortperm(eg[:,i])], V_new_endog[:,i][sortperm(eg[:,i])]) for i in 1:5]
        global nest = [V_interp_array_endog[i].(kappa[findall(>=(minimum(eg[:, i])), kappa)]) for i in 1:5]
        for i in 1:5
            global idx = findall(<(minimum(eg[:, i])), kappa)
            if length(idx) > 0
                global nest[i] = vcat(((1+r) * kappa[idx] .+ exp(z_vals[i])).^(1-gamma)./(1-gamma) .+ beta * dot(V_old[1,:], Pi[i, :]), nest[i])
            end
        end
        global V_new = reshape(reduce(vcat,nest),(size(nest[1])[1],size(nest)[1]))
        global converged = convergence(V_new, V_old)
        global V_old = V_new
    end
end

println("EGM")
egm()

############################################
# Stationary Distribution of State Variables
############################################

println("-------------------------------------------")
println("Stationary Distribution of State Variables")

n_gridpoints = 1000
kappa = range(a_min,a_max,length=n_gridpoints)

howard(20)

function s(i,j)
    return (j.-1)*n_gridpoints .+ i
end

function i(s)
    return mod.(s.-1, n_gridpoints) .+ 1
end

function j(s)
    return floor.(Int, (s.-1)/n_gridpoints) .+ 1
end

function a_prime(s)
    return policy[i(s), j(s)]
end

function g(s)
    return minimum([maximum(findall(<=(a_prime(s)), kappa)), n_gridpoints - 1])
end

function omega(s)
    return (a_prime(s) - kappa[g(s)]) / (kappa[g(s) + 1] - kappa[g(s)])
end

# create gamma
Gamma = zeros(n_gridpoints*n_states, n_gridpoints*n_states)

for s in 1:n_gridpoints*n_states
    for s_prime in 1:n_gridpoints*n_states
        if i(s_prime) == g(s)
            Gamma[s, s_prime] = (1-omega(s)) * Pi[j(s), j(s_prime)]
        elseif i(s_prime) == g(s) + 1
            Gamma[s, s_prime] = omega(s) * Pi[j(s), j(s_prime)]
        end
    end
end

# use invariant distribution to compute stationary distribution
function compute_invar_eigen(Gamma)
    decomp = eigen(transpose(Gamma))
    idx = findall(==(1), round.(real(decomp.values), digits = 2))[1]
    return real.(decomp.vectors[:, idx]) / sum(real.(decomp.vectors[:, idx]))
end

@time stat_dist = compute_invar_eigen(Gamma)

sparse_Gamma = sparse(Gamma)
t_sparse_Gamma = transpose(sparse_Gamma)

a_vec = vcat(kappa, kappa, kappa, kappa, kappa)

vals = [cumsum(stat_dist[Integer.(s(1:1000, ones(1000)* i))], dims = 1) for i in 1:5]
p_2_1 = plot(
    kappa, 
    vals, 
    xlab=L"a",
    ylab=L"\Phi(a,z)",
    label=["z=$(round(x;digits=2))" for x in transpose(z_vals)],
    title = "CDF of Stationary Distribution by State"
)

savefig(p_2_1, "$(path)stat_dist_cdf.png")

net_assets = sum(stat_dist .* a_vec)
println("Net Assets via Eigenvector Method: $(round(net_assets, digits = 5))")

# compute invariate distribution using simulation

n_sims = 1000
z_invariant_dist = QuantEcon.stationary_distributions(z_discrete)[1]

function compute_invar_sim(t_sparse_Gamma)
    start_vals = sample(1:n_states, Weights(z_invariant_dist), n_sims)
    net_demand_sims = zeros(1000*1000)

    for history in 1:1000
        start_dist = zeros(n_gridpoints*n_states)
        start_dist[s(1, start_vals[history])] = 1
        global period_dist = start_dist
        for period in 1:2000
            if period <= 1000
                global period_dist = t_sparse_Gamma * period_dist
            else
                global period_dist = t_sparse_Gamma * period_dist
                net_demand_sims[(history-1)*1000 + period - 1000] = sum(period_dist .* a_vec)
            end
        end
    end
    return net_demand_sims
end

@time net_demand_sims = compute_invar_sim(t_sparse_Gamma)

println("Average Net Assets Across Simulations: $(round(mean(net_demand_sims), digits = 5))")