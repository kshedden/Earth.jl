using RCall
using Earth
using DataFrames

R"
install.packages('AmesHousing')
library(AmesHousing)
D = make_ames()
"

ames = @rget D

y = Float64.(D[:, :Sale_Price])
X = select(D, Not(:Sale_Price))

m = fit(EarthModel, X, y; maxit=20, knots=20, knot_penalty=3, maxorder=2, verbose=true)













