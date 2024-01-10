using CategoricalArrays
using DataFrames
using Earth
using LinearAlgebra
using StableRNGs
using Statistics
using Test

@testset "Hinges" begin

    n = 3
    X = [1. 1 -1; 2 2 -2; 3 5 -3]
    y = [1., 2, 3]

    h1 = Earth.Hinge(1, 0, true)
    z = Float64[1, 1, 1]
    Earth.update!(h1, X, z)
    @test isapprox(z, [1., 2, 3])
    Earth.update!(h1, X, z)
    @test isapprox(z, [1., 4, 9])

    h2 = Earth.Hinge(2, 1, true)
    Earth.update!(h2, X, z)
    @test isapprox(z, [0., 4, 36])

    h3 = Earth.Hinge(3, -2, false)
    Earth.update!(h3, X, z)
    @test isapprox(z, [0., 0, 36])
    Earth.update!(h3, X, z)
    @test isapprox(z, [0., 0, 36])

    io = IOBuffer()
    print(io, h1)
    print(io, h2)
    print(io, h3)
    s = String(take!(io))
    @test s == "h(v1 - 0.000)h(v2 - 1.000)h(-2.000 - v3)"
end

@testset "Basic no prune" begin

    rng = StableRNG(123)

    n = 1000
    X = randn(rng, n, 3)
    y = X[:, 2] + X[:, 2] .* X[:, 3] + randn(rng, n)

    cfg = EarthConfig(; maxit=3, prune=false)
    m = fit(EarthModel, X, y; config=cfg)

    @test isapprox(mean(residuals(m).^2), 1, atol=0.01, rtol=0.02)

    # Test that the columns of U are orthogonal
    U = hcat(m.U...)
    @test isapprox(U' * U, I(size(U, 2)))

    # Check that D and U span the same subspace
    D = hcat(m.D...)
    D1, _, _ = svd(D)
    U1, _, _ = svd(U)

    @test length(m.U) == length(m.D)
    @test isapprox(sum(svd(D1' * U1).S), length(m.D))

    # Check that D spans the intended subspace
    XX = [ones(n) X[:, 2] (X[:, 2] .* X[:, 3])]
    XX, _, _ = svd(XX)
    @test isapprox(sum(svd(XX'*D1).S), 3, rtol=0.01, atol=0.01)

    @test Earth.checkstate(m)

    io = IOBuffer()
    println(io, m)
end

@testset "Basic prune" begin

    rng = StableRNG(123)

    n = 1000
    X = randn(rng, n, 3)
    y = X[:, 2] + X[:, 2] .* X[:, 3] + randn(rng, n)

    cfg = EarthConfig(; maxit=3, refit=:ols)
    m = fit(EarthModel, X, y; config=cfg, verbosity=0)
    @test isapprox(mean(residuals(m).^2), 1, atol=0.01, rtol=0.02)

    # Check that D spans the intended subspace
    D = hcat(m.D...)
    D1, _, _ = svd(D)
    XX = [ones(n) X[:, 2] (X[:, 2] .* X[:, 3])]
    XX, _, _ = svd(XX)
    @test isapprox(sum(svd(XX'*D1).S), 3, rtol=0.01, atol=0.01)

    # Check the residuals (under OLS refit)
    resid = y - D1 * (D1' * y)
    @test isapprox(resid, residuals(m))
end

@testset "Basic constraints" begin

    rng = StableRNG(123)

    n = 1000
    X = randn(rng, n, 3)
    y = X[:, 2] + X[:, 2] .* X[:, 3] + randn(rng, n)

    constraints = Set([[0, 1, 0], [0, 0, 1], [0, 1, 1]])

    cfg = EarthConfig(; maxit=3, num_knots=100, constraints=constraints)
    m = fit(EarthModel, X, y; config=cfg)
    @test isapprox(std(residuals(m)), 1, atol=0.01, rtol=0.02)

    # Check that D spans the intended subspace
    D = hcat(m.D...)
    D1, _, _ = svd(D)
    XX = [ones(n) X[:, 2] (X[:, 2] .* X[:, 3])]
    XX, _, _ = svd(XX)
    @test isapprox(sum(svd(XX'*D1).S), 3, rtol=0.01, atol=0.01)
end

@testset "Check maxorder" begin

    rng = StableRNG(123)

    n = 1000
    X = randn(rng, n, 3)
    y = X[:, 2] + X[:, 2] .* X[:, 3] + randn(rng, n)

    config = EarthConfig(; maxorder=1)
    m1 = fit(EarthModel, X, y; config=config)
    config = EarthConfig(; maxorder=2)
    m2 = fit(EarthModel, X, y; config=config)
    @test maximum(Earth.order(m1)) == 1
    @test maximum(Earth.order(m2)) == 2
end

@testset "Check maxdegree" begin

    rng = StableRNG(123)

    n = 1000
    X = randn(rng, n, 3)
    y = X[:, 2] + X[:, 2] .* X[:, 3] + randn(rng, n)

    cfg = EarthConfig(; maxdegree=1)
    m1 = fit(EarthModel, X, y; config=cfg)
    cfg = EarthConfig(; maxdegree=2)
    m2 = fit(EarthModel, X, y; config=cfg)
    @test maximum(Earth.degree(m1)) == 1
    @test maximum(Earth.degree(m2)) == 2
end

@testset "Predict" begin

    rng = StableRNG(123)

    n = 1000
    X = randn(rng, n, 3)
    y = X[:, 2] + X[:, 2] .* X[:, 3] + randn(rng, n)

    for prune in [false, true]
        cfg = EarthConfig(; maxit=4, num_knots=100, prune=prune)
        m = fit(EarthModel, X, y; config=cfg, verbosity=0)
        yhat1 = predict(m)
        yhat2 = predict(m, X)
        @test isapprox(yhat1, yhat2)
    end
end

@testset "Test additive" begin

    rng = StableRNG(123)

    n = 1000
    x1 = randn(rng, n)
    x2 = randn(rng, n)

    y = x1 .* clamp.(x1 .- 1, 0, Inf) + x2 .* clamp.(1 .- x2, -Inf, 0) + 0.25*randn(rng, n)
    X = (x1=x1, x2=x2)

    cfg = EarthConfig(; maxorder=1)
    m = fit(EarthModel, X, y; config=cfg)

    z = range(-3, 3, 100)
    y1 = predict(m, (x1=z, x2=zeros(100)))
    y1x = z .* clamp.(z .- 1, 0, Inf)
    @test mean(abs.(y1 - y1x)) < 0.05

    y2 = predict(m, (x1=zeros(100), x2=z))
    y2x = z .* clamp.(1 .- z, -Inf, 0)
    @test mean(abs.(y2 - y2x)) < 0.05
end

@testset "Categorical" begin

    rng = StableRNG(123)

    n = 5000
    X = [randn(rng, n), CategoricalArray(rand(rng, ["a", "b"], n)),
         CategoricalArray(rand(rng, [1, 2, 3], n))]
    y = X[1] + X[1] .* (X[2] .== "b") + (X[2] .== "a") .* (X[3] .== 3) + randn(rng, n)

    Xvec = X
    Xtup = tuple(X...)
    Xnt = (x1=X[1], x2=X[2], x3=X[3])
    Xdf = DataFrame(X, :auto)

    for X in [Xvec, Xtup, Xnt, Xdf]
        cfg = EarthConfig(; maxit=5)
        m = fit(EarthModel, X, y; config=cfg, verbosity=0)
        io = IOBuffer()
        println(io, m)
        @test isapprox(mean(residuals(m).^2), 1, atol=0.01, rtol=0.1)
        predict(m)
        predict(m, X)

        # Check that an error is thrown if the data for prediction
        # contains levels of a categorical variable that are not
        # in the training data.
        let err = nothing
            X = deepcopy(X)
            if typeof(X) <: AbstractDataFrame
                X[1, 2] = "z"
            else
                X[2][1] = "z"
            end
            try
                predict(m, X)
            catch err
            end
            @test err isa Exception
        end
    end
end

@testset "Test weights" begin

    rng = StableRNG(123)

    n = 1000
    x1 = randn(rng, n)
    x2 = randn(rng, n)

    y = x1 .* clamp.(x1 .- 1, 0, Inf) + x2 .* clamp.(1 .- x2, -Inf, 0) + 0.25*randn(rng, n)
    X = (x1=x1, x2=x2)

    # The first 500 observations, no weights
    y0 = y[1:500]
    X0 = (x1=x1[1:500], x2=x2[1:500])
    cfg = EarthConfig()
    m0 = fit(EarthModel, X0, y0; config=cfg, verbosity=0)
    y0_pred = predict(m0)

    # All observations, but place zero weight on the second half of data
    w = ones(length(y))
    w[501:end] .= 0
    mw = fit(EarthModel, X, y; weights=w, config=cfg, verbosity=0)
    y_pred = predict(mw)

    @assert isapprox(m0.meanx, mw.meanx)
    @assert isapprox(m0.sdx, mw.sdx)
    @assert isapprox(m0.meany, mw.meany)
    @assert isapprox(m0.sdy, mw.sdy)
    @assert isapprox(y0_pred, y_pred[1:500])
end
