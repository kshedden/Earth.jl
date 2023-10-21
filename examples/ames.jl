# # Ames housing data

using RCall
using DataFrames
using Earth

ENV["GKSwstype"] = "nul" #hide
using Plots

R"
install.packages('AmesHousing', repos='https://cloud.r-project.org')
library(AmesHousing)
D = make_ames()
"

ames = @rget D;

# The response variable is the sale price of each property.

y = Float64.(D[:, :Sale_Price]) / 10000;

# Everything else in the dataframe is a covariate

X = select(D, Not(:Sale_Price));
names(X)

# Fit a model and inspect its structure

cfg = EarthConfig(; maxit=40, maxorder=1)
m = fit(EarthModel, X, y; config=cfg, verbose=true)

# Next we generate a plot showing the generalized R2 as we increase the number of terms

r2 = gr2(m)
p = plot(1:length(r2), r2, xlabel="Number of terms", ylabel="R2")
plot!(p, 1:length(r2), r2, label="2")
Plots.savefig(p, "./assets/ames1.svg");

# ![R-squares](assets/ames1.svg)
