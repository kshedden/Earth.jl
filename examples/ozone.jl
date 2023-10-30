using RCall
using Earth
using DataFrames
using Statistics
using Plots
using Printf

R"
install.packages('earth', repos='https://cloud.r-project.org')
library(earth)
D = ozone1
"

ozone = @rget D;

# The response variable is a vector of Float64 type.

y = Vector(D[:, :O3])
X = select(D, Not(:O3));

# Create a named tuple of the variables.

na = tuple(Symbol.(names(X))...)
X = NamedTuple{na}(eachcol(X))

# Fit an additive model with degree 1

cfg = EarthConfig(; maxit=30, maxdegree=1, maxorder=1)
m1 = fit(EarthModel, X, y; config=cfg, verbosity=1)

# Fit a model with degree 1 that allows pairwise interactions

cfg = EarthConfig(; maxit=30, maxdegree=1, maxorder=2)
m2 = fit(EarthModel, X, y; config=cfg, verbosity=1)

# Get the generalized R2 statistics for each model

r2_1 = gr2(m1)
r2_2 = gr2(m2);

p = plot(eachindex(r2_1), r2_1, xlabel="Number of terms", ylabel="R2", label="maxorder=1")
plot!(p, eachindex(r2_2), r2_2, label="maxorder=2")
Plots.savefig(p, "./assets/ozone1.svg");

# Estimate the conditional mean function with respect to one variable `xvar`, fixing
# another variable `zvar` at three of its quantiles (0.2, 0.5, 0.8), and holding all
# other variables fixed at their mean values.

function fit_cmean(md, xvar, zvar)
    @assert xvar != zvar
    vars = [x for x in names(D) if x != "O3"]
    dp = D[1:150, vars]
    for v in vars
        dp[:, v] .= mean(D[:, v])
    end
    qq = quantile(D[:, zvar], [0.2, 0.5, 0.8])
    for j in 1:3
        i = 50*(j-1)
        dp[(i+1):(i+50), xvar] = range(extrema(D[:, xvar])..., 50)
        dp[(i+1):(i+50), zvar] .= qq[j]
    end
    yh = predict(md, dp)
    return dp[:, xvar], yh, qq
end;

# Plot the fitted values from `fit_cmean`.

function plot_cmean(m, xvar, zvar, ifig)
    x, y, qq = fit_cmean(m, xvar, zvar)
    p = plot(x[1:50], y[1:50], xlabel=string(xvar), ylabel="O3", label=@sprintf("%s=%.2f", zvar, qq[1]))
    plot!(p, x[51:100], y[51:100], label=@sprintf("%s=%.2f", zvar, qq[2]))
    plot!(p, x[101:150], y[101:150], label=@sprintf("%s=%.2f", zvar, qq[3]))
    Plots.savefig(p, "./assets/ozone$(ifig).svg");
end

# Plot the estimated mean ozone with respect to day of year (doy), for three
# fixed values of the visibilty variable.

plot_cmean(m1, :doy, :vis, 2)
plot_cmean(m2, :doy, :vis, 3)
