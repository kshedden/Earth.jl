using Earth
using Statistics

n = 1000

x1 = randn(n)
x2 = randn(n)

y = x1 .* clamp.(x1 .- 1, 0, Inf) + x2 .* clamp.(1 .- x2, -Inf, 0) + randn(n)

X = (x1=x1, x2=x2)

m = fit(EarthModel, X, y; maxorder=1)

z = range(-3, 3, 100)
y1 = predict(m, (x1=z, x2=zeros(100)))
y1x = z .* clamp.(z .- 1, 0, Inf)


y2 = predict(m, (x1=zeros(100), x2=z))
y2x = z .* clamp.(1 .- z, -Inf, 0)
