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
    @test s == "v1 > 0.000v2 > 1.000v3 < -2.000"
end

@testset "Basic no prune" begin

    rng = StableRNG(123)

    n = 1000
    X = randn(rng, n, 3)
    y = X[:, 2] + X[:, 2] .* X[:, 3] + randn(rng, n)

    m = fit(EarthModel, X, y; maxit=3, knots=20, prune=false)

    @test isapprox(mean(m.resid.^2), 1, atol=0.01, rtol=0.02)

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

    m = fit(EarthModel, X, y; maxit=3, knots=20, prune=true)
    @test isapprox(mean(m.resid.^2), 1, atol=0.01, rtol=0.02)

    # Check that D spans the intended subspace
    D = hcat(m.D...)
    D1, _, _ = svd(D)
    XX = [ones(n) X[:, 2] (X[:, 2] .* X[:, 3])]
    XX, _, _ = svd(XX)
    @test isapprox(sum(svd(XX'*D1).S), 3, rtol=0.01, atol=0.01)

    # Check the residuals
    resid = y - D1 * (D1' * y)
    @test isapprox(resid, m.resid)
end

@testset "Basic constraints" begin

    rng = StableRNG(123)

    n = 1000
    X = randn(rng, n, 3)
    y = X[:, 2] + X[:, 2] .* X[:, 3] + randn(rng, n)

    constraints = Set([[0, 1, 0], [0, 0, 1], [0, 1, 1]])

    m = fit(EarthModel, X, y; maxit=3, knots=100, constraints=constraints, prune=true)
    @test isapprox(std(m.resid), 1, atol=0.01, rtol=0.02)

    # Check that D spans the intended subspace
    D = hcat(m.D...)
    D1, _, _ = svd(D)
    XX = [ones(n) X[:, 2] (X[:, 2] .* X[:, 3])]
    XX, _, _ = svd(XX)
    @test isapprox(sum(svd(XX'*D1).S), 3, rtol=0.01, atol=0.01)
end

@testset "Predict" begin

    rng = StableRNG(123)

    n = 1000
    X = randn(rng, n, 3)
    y = X[:, 2] + X[:, 2] .* X[:, 3] + randn(rng, n)

    for prune in [false]
        m = fit(EarthModel, X, y; maxit=3, knots=100, prune=prune, verbose=false)
        yhat1 = predict(m)
        yhat2 = predict(m, X)
        @test isapprox(yhat1, yhat2)
    end
end

@testset "Categorical" begin

    rng = StableRNG(123)

    n = 1000
    X = [randn(rng, n), CategoricalArray(rand(rng, ["a", "b"], n)),
         CategoricalArray(rand(rng, [1, 2, 3], n))]
    y = X[1] + X[1] .* (X[2] .== "b") + (X[2] .== "a") .* (X[3] .== 3) + randn(rng, n)

    Xvec = X
    Xtup = tuple(X...)
    Xnt = (x1=X[1], x2=X[2], x3=X[3])
    Xdf = DataFrame(X, :auto)

    for X in [Xvec, Xtup, Xnt, Xdf]
        m = fit(EarthModel, X, y; maxit=5)
        io = IOBuffer()
        println(io, m)
        @test isapprox(mean(residuals(m).^2), 1, atol=0.01, rtol=0.1)
    end
end
