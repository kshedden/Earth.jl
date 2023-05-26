# Earth/MARS

This is a Julia implementation of a regression modeling procedure that
is similar to Jerome Friedman's 1991 Multivariate Adaptive Regression
Splines (MARS), which is also known as Earth for trademark reasons.

The original MARS used backward selection for model pruning, but this
implementation uses the Lasso, which was not invented yet at the time
that MARS was conceived.

```julia
using Earth, UnicodePlots, StableRNGs

rng = StableRNG(123)
n = 500
X = randn(rng, n, 2)
y = X[:, 1].^2 - X[:, 2] + randn(rng, n)

m = fit(EarthModel, X, y; maxorder=1)

# Estimate E[y | x1, x2=0]
x = -2:0.2:2
X1 = [x zeros(length(x))]
y1 = predict(m, X1)
p1 = lineplot(x, y1)
println(p1)

# Estimate E[y | x1=0, x2]
x = -2:0.2:2
X2 = [zeros(length(x)) x]
y2 = predict(m, X2)
p2 = lineplot(x, y2)
println(p2)
```

## References

[1] Multivariate Adaptive Regression Splines, Jerome H. Friedman.
The Annals of Statistics, Vol. 19, No. 1. (Mar., 1991), pp. 1-67.
