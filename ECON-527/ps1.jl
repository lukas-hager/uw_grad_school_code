using QuantEcon
using StatsBase
using Statistics
using BasicInterpolators
using Random
using Plots
using Printf

# set the seed
Random.seed!(42069)

# set the output path
path = "/Users/hlukas/Google Drive/Grad School/2022-2023/Spring/ECON 527/Assignments/Assignment_1/"

# functions that are used throughtout the script (mostly formatting results)

# compare the moments of the approximation to the theoretical moments
function appx_stats(appx)

    # first calculate the invariant distribution for seeding the simulations
    invariant_dist = QuantEcon.stationary_distributions(appx)[1,1]

    # initialize an empty matrix
    simulations = Array{Float64, 2}(undef, n_periods, n_sims)

    # create the simulation seeds based on the invariant distribution
    start_vals = sample(1:9, Weights(invariant_dist), n_sims)

    sim_results = QuantEcon.simulate!(simulations, appx; init=start_vals)

    # calculate autocorrelation
    sims_autocor = mean(autocor(sim_results, [1]), dims=2)[1]

    # calculate variance
    sims_var = mean(var(sim_results, dims=1), dims = 2)[1]

    # calculate autocovariance
    sims_autocov = mean(autocov(sim_results, [1]), dims=2)[1]

    # calculate conditional variance
    sims_cond_var_mat = sim_results[2:end, 1:end] - rho * sim_results[1:(end-1), 1:end]
    sims_cond_var = mean(var(sims_cond_var_mat, dims=1), dims = 2)[1]

    @printf "Autocorrelation ratio: %.4f \n" sims_autocor / rho
    @printf "Variance ratio: %.4f\n" sims_var / (sigma^2 / (1 - rho^2))
    @printf "Autocovariance ratio: %.4f\n" sims_autocov / (rho * sigma^2 / (1 - rho^2))
    @printf "Conditional variance ratio: %.4f\n\n" sims_cond_var / sigma^2

end

# the CRRA utility function
function u(c)
    c.^(1-gamma) / (1-gamma)
end

# calculating the percentage of an array of true/false that is true
function perc_true(arr)
    sum(arr) * 100. / length(arr)
end

# test the relevant properties for an interpolated function
function check_properties(x, y)
    y_diff = y[2:end] - y[1:(end-1)]
    x_diff = x[2:end] - x[1:(end-1)]
    diff_ratio = y_diff ./ x_diff

    @printf "Monotonicity: %.1f%%\n" perc_true(y_diff .> 0)
    @printf "Concavity: %.1f%%\n" perc_true(diff_ratio[1:(end-1)] .> diff_ratio[2:end])
    @printf "Negativity: %.1f%%\n" perc_true(y .< 0)

end

################################################################################
# Discretizing an AR(1)
################################################################################

# define the parameters of the AR process
rho = .99
sigma = .2
mu = 0

# define the tauchen parameters
gridpoints = 9
n_std = 3

# define the simulation parameters

n_sims = 1000
n_periods = 1000

# get the approximations
tauchen_appx_3 = QuantEcon.tauchen(gridpoints,rho,sigma,mu,n_std)
rouwenhorst_appx = QuantEcon.rouwenhorst(gridpoints,rho,sigma,mu)

# add the extra approximation
tauchen_appx_2_5 = QuantEcon.tauchen(gridpoints,rho,sigma,mu,2.5)

# calculate and print the results
println("2. Tauchen Approximation (3 SD)")
appx_stats(tauchen_appx_3)

println("3. Rouwenhorst Approximation")
appx_stats(rouwenhorst_appx)

println("4. Tauchen Approximation (2.5 SD)")
appx_stats(tauchen_appx_2_5)

################################################################################
# Interpolation
################################################################################ 

# define the parameters
gamma = 4.
a = .1
b = 30.
c = 3.

# get a grid for c, use it and u(c) to interpolate
c_i = range(a, b, length=100)
itp = CubicInterpolator(c_i, u(c_i))

# get a finer grid for testing the interpolation
x_j = range(a, b, length=1000)

println("2 (a)")
@printf "Maximum error is %.4f\n" maximum(abs.(u(x_j) - itp.(x_j)))

p_2a = plot(
    x_j, 
    [u(x_j), itp.(x_j)], 
    label=["u(c)" "Cubic Spline"],
    title="Utility Function Interpolation Comparison"
)

println("2 (b)")
check_properties(x_j, itp.(x_j))

# get a polynomial grid
z_j = a .+ (b - a)*range(0,1,length=100).^c

# use this to interpolate, specify no boundaries because we need to extrapolate
itp_poly = CubicInterpolator(z_j, u(z_j), NoBoundaries())

println("3 (a)")
@printf "Maximum error is %.4f\n" maximum(abs.(u(x_j) - itp_poly.(x_j)))

p_3a = plot(
    x_j, 
    [u(x_j), itp_poly.(x_j)], 
    label=["u(c)" "Cubic Spline"],
    title="Utility Function Interpolation Comparison"
)

println("3 (b)")
check_properties(x_j, itp_poly.(x_j))

# get the right extrapolation range
x_30_40 = range(30, 40, length = 100)

p_4a = plot(
    x_30_40, 
    [u(x_30_40), itp_poly.(x_30_40)], 
    label=["u(c)" "Cubic Spline (Polynomial Grid)"],
    title="Utility Function Interpolation Comparison"
)

println("4 (a)")
@printf "Maximum error is %.6f\n" maximum(abs.(u(x_30_40) - itp_poly.(x_30_40)))
check_properties(x_30_40, itp_poly.(x_30_40))

# get the left extrapolation range
x_001_01 = range(.01, .1, length = 100)

p_4b = plot(
    x_001_01, 
    [u(x_001_01), itp_poly.(x_001_01)], 
    label=["u(c)" "Cubic Spline (Polynomial Grid)"],
    title="Utility Function Interpolation Comparison"
)

println("4 (b)")
@printf "Maximum error is %.4f\n" maximum(abs.(u(x_001_01) - itp_poly.(x_001_01)))
check_properties(x_001_01, itp_poly.(x_001_01))

# save files
savefig(p_2a, path * "p_2a.png")
savefig(p_3a, path * "p_3a.png")
savefig(p_4a, path * "p_4a.png")
savefig(p_4b, path * "p_4b.png")
;