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
path = "/Users/hlukas/Google Drive/Grad School/2022-2023/Spring/ECON 527/Assignments/Assignment_2/"

# Neoclassical Growth Model

println("NEOCLASSICAL GROWTH MODEL")
println("__________________________")

# define utility parameters
beta = 0.96
A = 1.
delta = 1.
alpha = .35
epsilon = 10e-4

# define grid parameters
a = .01
b = 10.
c = 2.
n_gridpoints = 100

# get a polynomial grid
kappa = a .+ (b - a)*range(0,1,length=n_gridpoints).^c

function V(k_t_1, k_t, interp_obj) 
    log(A * k_t ^ alpha + (1. - delta) * k_t - k_t_1) + beta * interp_obj(k_t_1)
end

# value function interation
function vfi_neoclassical_growth_model()
    converged = false
    counter = 0

    @time while converged == false
        # update iteration counter
        counter += 1

        # initialize to zero if it's the first iteration
        if counter == 1
            V_old = zeros(n_gridpoints)
        end

        global V_interp = CubicInterpolator(kappa, V_old)
        global V_new = zeros(n_gridpoints)
        global policy = zeros(n_gridpoints)

        for i in 1:100
            res = optimize(
                k_t_1 -> -V(k_t_1, kappa[i], V_interp), 
                a, 
                minimum([b, A * kappa[i] ^ alpha - .001])
            )
            V_new[i] = -res.minimum
            policy[i] = res.minimizer
        end

        if maximum(abs.(V_new - V_old)) < epsilon*(1-beta)
            converged = true
            println("Converged in $(counter) iterations")
        else
            global V_old = V_new
        end

    end
end

function true_value(k) 
    a = (1. / (1. - beta)) * (log(A * (1. - alpha * beta)) + (alpha * beta) * log(A * alpha * beta)/(1. - alpha*beta))
    b = alpha / (1. - alpha * beta)
    return a .+ b * log.(k)
end

function true_policy(k)
    alpha * beta * A * k.^ alpha 
end

vfi_neoclassical_growth_model()

p_2_a1 = plot(
    kappa, 
    [V_interp.(kappa) true_value(kappa)], 
    xlab=L"k",
    ylab=L"V(k)",
    label=["Estimated" "Actual"],
    linestyle=[:solid :dash],
    color=[:blue :red],
    title="Value Function"
)

p_2_a2 = plot(
    kappa, 
    [policy true_policy(kappa)], 
    xlab=L"k",
    ylab=L"k'",
    label=["Estimated" "Actual"],
    linestyle=[:solid :dash],
    color=[:blue :red],
    title="Policy Function"
)

println("Decumulation Point: k=$(round(minimum(kappa[kappa .> policy]); digits=2))")
println("Maximum Difference: $(100. * round(maximum(abs.((V_new - true_value(kappa))/true_value(kappa)));digits=6))%")

# Income Fluctuation Problem

println("INCOME FLUCTUATION PROBLEM")
println("__________________________")

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

# value function interation
function vfi_income_fluctuation_problem()
    converged = false
    counter = 0

    @time while converged == false
        # update iteration counter
        counter += 1

        # initialize to zero if it's the first iteration
        if counter == 1
            V_old = zeros(n_gridpoints, n_states)
        end

        global V_interp_array = [CubicInterpolator(kappa, V_old[:,i]) for i in 1:5]
        global V_new = zeros(n_gridpoints, n_states)
        global policy = zeros(n_gridpoints, n_states)

        for z_ind in 1:5
            for a_ind in 1:100
                res = optimize(
                    a_t_1 -> -V(a_t_1, kappa[a_ind], z_vals[z_ind], V_interp_array), 
                    a_min, 
                    minimum([a_max, (1. + r) * kappa[a_ind] + exp(z_vals[z_ind]) - .001])
                )
                V_new[a_ind, z_ind] = -res.minimum
                policy[a_ind, z_ind] = res.minimizer
            end
        end

        if maximum(abs.(V_new - V_old)) < epsilon*(1-beta)
            converged = true
            println("Converged in $(counter) iterations")
        else
            global V_old = V_new
        end

    end
end

vfi_income_fluctuation_problem()

# calculate the optimal consumption
opt_c = (1. + r) * kappa .+ exp.(transpose(z_vals)) .- policy

p_2_1 = plot(
    kappa, 
    V_new, 
    xlab=L"a",
    ylab=L"V(a,z)",
    label=["z=$(round(x;digits=2))" for x in transpose(z_vals)],
    title="Value Function"
)

p_2_2 = plot(
    kappa, 
    policy, 
    xlab=L"a",
    ylab=L"a'",
    label=["z=$(round(x;digits=2))" for x in transpose(z_vals)],
    title="Policy Function"
)

p_3 = plot(
    kappa, 
    opt_c, 
    xlab=L"a",
    ylab=L"c'",
    label=["z=$(round(x;digits=2))" for x in transpose(z_vals)],
    title="Optimal Consumption Function"
)

savefig(p_2_a1, "$(path)ngm_vf.png")
savefig(p_2_a2, "$(path)ngm_pf.png")
savefig(p_2_1, "$(path)ifp_vf.png")
savefig(p_2_2, "$(path)ifp_pf.png")
savefig(p_3, "$(path)ifp_c.png")
;