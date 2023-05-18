using RCall
using Earth
using DataFrames

R"
install.packages('earth')
library(earth)
D = ozone1
"

ozone = @rget D

y = Vector(D[:, :O3])
X = Matrix(select(D, Not(:O3)))

m = fit(EarthModel, X, y; maxit=30, knots=20, knot_penalty=3, maxorder=2, verbose=true)

