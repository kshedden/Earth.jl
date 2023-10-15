using RCall
using Earth
using DataFrames

R"
install.packages('earth')
library(earth)
D = ozone1
"

ozone = @rget D

# The response variable is a vector of Float64 type.

y = Vector(D[:, :O3])
X = select(D, Not(:O3))

# Create a named tuple of the variables.

na = tuple(Symbol.(names(X))...)
X = NamedTuple{na}(eachcol(X))

m = fit(EarthModel, X, y; maxit=30, knots=20, verbose=true)

r2 = gr2(m)
p = plot(1:length(r2), r2, xlabel="Number of terms", ylabel="R2")
plot!(p, 1:m, r2_2, label="2")
Plots.savefig(p, "./assets/ozone1.svg");

