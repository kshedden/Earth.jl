```@meta
EditURL = "<unknown>/README.jl"
```

# Earth/MARS

This is a Julia implementation of a regression modeling procedure that
is similar to Jerome Friedman's 1991 Multivariate Adaptive Regression
Splines (MARS), which is also known as Earth for trademark reasons.

The original MARS used backward selection for model pruning, but this
implementation uses the Lasso, which was not invented yet at the time
that MARS was conceived.

````@example README
using Earth, UnicodePlots, StableRNGs

rng = StableRNG(123)
n = 500
X = randn(rng, n, 2)
y = X[:, 1].^2 - X[:, 2] + randn(rng, n)

m = fit(EarthModel, X, y; maxorder=1)
````

Estimate E[y | x1, x2=0]

````@example README
x = -2:0.2:2
X1 = [x zeros(length(x))]
y1 = predict(m, X1)
p1 = lineplot(x, y1)
savefig("./assets/readme1.svg", p1)
````

![Example plot 1](assets/readme1.svg)

Estimate E[y | x1=0, x2]

````@example README
x = -2:0.2:2
X2 = [zeros(length(x)) x]
y2 = predict(m, X2)
p2 = lineplot(x, y2)
savefig("./assets/readme2.svg", p2)
````

![Example plot 2](assets/readme2.svg)

## References

[1] Multivariate Adaptive Regression Splines, Jerome H. Friedman.
The Annals of Statistics, Vol. 19, No. 1. (Mar., 1991), pp. 1-67.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

