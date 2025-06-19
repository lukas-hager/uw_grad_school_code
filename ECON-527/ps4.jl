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
# Wealth Distribution in Standard IM Models
############################################

println("-------------------------------------------")
println("Wealth Distribution in Standard IM Models")

function d_v(spline, x)
    """
    This function calculates the derivative of a cubic spline interpolation.
    """
    i = findcell(x, spline)
    ξ = x - spline.r.x[i]
    return spline.b[i] + 2 * spline.c[i]*ξ + 3 * spline.d[i]*ξ^2
end

function V_vec_endog(inv_policy, V_old, u_func, r, a_i, mc, w) 
    """
    This function returns the value function on the endogeneous grid.
    """
    # evaluate the future value using the policy function
    flow_utility = u_func.((1. + r)*inv_policy .+ w .* exp.(transpose(mc.state_values)) .- a_i)
    # calculate all combinations of the continuation Utility
    continuation_utility = V_old * transpose(mc.p)
    return flow_utility .+ beta .* continuation_utility
end

function endog_grid(spline_arr, u_inv_func, r, a_i, mc, w)
    """
    This function produces the endogeneous grid values.
    """
    nest = [[d_v(spline_arr[i], x) for x in a_i] for i in 1:n_states]
    e_d_v = reshape(reduce(vcat,nest),(size(nest[1])[1],size(nest)[1])) * transpose(mc.p)
    return (a_i .- w .* exp.(transpose(mc.state_values)) + u_inv_func.(beta * e_d_v)) ./ (1. + r)
end

function convergence(V_new, V_old)
    """
    This function tests for convergence in value functions.
    """
    diff = maximum(abs.(V_new - V_old)) 
    if diff < epsilon*(1-beta)
        return true
    else
        return false
    end
end

function interpolate_policy(eg, a_i, a_i_fine)
    policy = [CubicSplineInterpolator(eg[:,i], a_i, NoBoundaries()) for i in 1:n_states]
    policy_nest = [policy[i].(a_i_fine[findall(>=(minimum(eg[:, i])), a_i_fine)]) for i in 1:n_states]
    policy_nest = [vcat(ones(length(a_i_fine) - length(policy_nest[i])) * a_min, policy_nest[i]) for i in 1:n_states]
    policy_fine = reshape(reduce(vcat,policy_nest),(size(policy_nest[1])[1],size(policy_nest)[1]))
    return policy_fine
end

function egm(r, mc, a_i, u, u_prime_inv, beta, w)
    global converged = false
    global counter = 0

    while converged == false
        # update iteration counter
        global counter += 1
        # initialize to zero if it's the first iteration
        if counter == 1
            global V_old = log.(repeat(a_i, 1, n_states) .- a_min .+ .01) #u.((1+r).*repeat(a_i, 1, n_states) .+ w .* exp.(transpose(mc.state_values))) ./ (1-beta)
        end

        # compute endogeneous gridpoints
        global V_interp_array = [CubicSplineInterpolator(a_i, V_old[:,i], NoBoundaries()) for i in 1:n_states]
        global eg = endog_grid(V_interp_array, u_prime_inv, r, a_i, mc, w)
        global V_new_endog = V_vec_endog(eg, V_old, u, r, a_i, mc, w)
        global V_interp_array_endog = [CubicSplineInterpolator(eg[:,i], V_new_endog[:,i], NoBoundaries()) for i in 1:n_states]
        global nest = [V_interp_array_endog[i].(a_i[findall(>=(minimum(eg[:, i])), a_i)]) for i in 1:n_states]
        for i in 1:n_states
            global idx = findall(<(minimum(eg[:, i])), a_i)
            if length(idx) > 0
                global nest[i] = vcat(u.((1+r) * a_i[idx] .+ w .* exp(mc.state_values[i]) .- a_min) .+ beta * dot(V_old[1,:], mc.p[i, :]), nest[i])
            end
        end
        global V_new = reshape(reduce(vcat,nest),(size(nest[1])[1],size(nest)[1]))
        global converged = convergence(V_new, V_old)
        global V_old = V_new
    end
    return eg, V_new
end

function i(s, a_i_fine)
    return mod.(s.-1, length(a_i_fine)) .+ 1
end

function a_prime(s, policy)
    return policy[s]
end

function g(s, a_i_fine, policy)
    return vec(clamp.(sum(a_i_fine .<= transpose(a_prime(s, policy)), dims = 1), 0, length(a_i_fine) - 1))
end

function omega(s, a_i_fine, policy)
    g_val = g(s, a_i_fine, policy)
    return (a_prime(s, policy) .- a_i_fine[g_val]) ./ (a_i_fine[g_val .+ 1] .- a_i_fine[g_val])
end

function compute_Gamma(policy, mc, a_i_fine)
    pol_dim = length(policy)
    n_gridpoints = size(policy)[1]
    # repeat vector elements
    big_Pi = repeat(mc.p, inner=(n_gridpoints,n_gridpoints))
    big_g = g(1:pol_dim, a_i_fine, policy)
    big_i = i(transpose(1:pol_dim), a_i_fine)
    big_omega = omega(1:pol_dim, a_i_fine, policy)
    # add two matrices
    mat_1 = (big_g .== big_i) .* (1 .-big_omega) .* big_Pi
    mat_2 = ((big_g .+ 1) .== big_i) .* big_omega .* big_Pi
    return mat_1 .+ mat_2
end

# use invariant distribution to compute stationary distribution
function compute_invar_eigen(Gamma)
    decomp = eigen(transpose(Gamma))
    idx = findall(==(1), round.(real(decomp.values), digits = 2))[1]
    return real.(decomp.vectors[:, idx]) / sum(real.(decomp.vectors[:, idx]))
end

function f_spe(
    r,
    markov_chain,
    u,
    u_prime_inv,
    beta,
    a_i,
    a_i_fine,
    w
)
    eg, V_new = egm(r, markov_chain, a_i, u, u_prime_inv, beta, w)
    policy = interpolate_policy(eg, a_i, a_i_fine)
    Gamma = compute_Gamma(policy, markov_chain, a_i_fine)
    phi = compute_invar_eigen(Gamma)
    nad = sum(phi .* repeat(a_i_fine, n_states))
    return V_new, policy, phi, nad
end

function f_nad(
    r,
    markov_chain,
    u,
    u_prime_inv,
    beta,
    a_i,
    a_i_fine,
    w
)
    eg, V_new = egm(r, markov_chain, a_i, u, u_prime_inv, beta, w)
    policy = interpolate_policy(eg, a_i, a_i_fine)
    Gamma = compute_Gamma(policy, markov_chain, a_i_fine)
    phi = compute_invar_eigen(Gamma)
    nad = sum(phi .* repeat(a_i_fine, n_states))
    return nad
end


############################################
# Huggett
############################################

# define parameters
#r = .02
gamma = 2
beta = .96
rho = .9
sigma = .2
mu = 0.
n_states = 7
n_gridpoints = 150
c = 2.
epsilon = 10e-4

# discretize log income, get state values and transition matrix
z_discrete = QuantEcon.rouwenhorst(n_states,rho,sigma,mu)
stat_dist = QuantEcon.stationary_distributions(z_discrete)[1,1]
# discretize wealth
avg_inc = dot(exp.(z_discrete.state_values), stat_dist)
a_min = - avg_inc
a_max = avg_inc * 50

a_i = a_min .+ (a_max - a_min)*range(0,1,length=n_gridpoints).^c

function crra(x)
    return x^(1 - gamma) / (1 - gamma)
end

function crra_prime_inv(x)
    return x^(-1 / gamma)
end

res = optimize(
    x -> f_nad(first(x), z_discrete, crra, crra_prime_inv, beta, a_i, a_i, 1)^2,
    -.2,
    0
)

V_new_hug, policy_hug, phi_hug, nad_hug = f_spe(res.minimizer, z_discrete, crra, crra_prime_inv, beta, a_i, a_i, 1)

############################################
# Aiyagari
############################################

alpha = .36
delta = .1
Delta = .1
L = dot(exp.(z_discrete.state_values), stat_dist)
a_min = 0
a_max = 240
n_gridpoints = 600
a_i = a_min .+ (a_max - a_min)*range(0,1,length=n_gridpoints).^c


function r(K)
    return alpha * (L / K)^(1-alpha) - delta
end

function w(K)
    return (1-alpha) * (K / L)^alpha
end

outer_counter = 0
converged = false
while converged == false
    outer_counter += 1
    if outer_counter == 1
        global K_val = 6
    end
    
    global S_val = f_nad(r(K_val), z_discrete, crra, crra_prime_inv, beta, a_i, a_i, w(K_val))
    
    global converged = abs(S_val - K_val) < epsilon

    println(r(K_val))

    global K_val += Delta * (S_val - K_val)

end

V_new_aiy, policy_aiy, phi_aiy, nad_aiy = f_spe(r(K_val), z_discrete, crra, crra_prime_inv, beta, a_i, a_i, w(K_val))







# discretize wealth
kappa = a_min .+ (a_max - a_min)*range(0,1,length=n_gridpoints).^c

egm(r, Pi, kappa, crra, beta)

a_i_fine = range(a_min, a_max,length=1000)

my_pol = interpolate_policy(eg, kappa, a_i_fine)

Gamma = compute_Gamma(my_pol)