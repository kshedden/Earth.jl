using Statistics
using LinearAlgebra
using Lasso

# A hinge function for the variable in position 'var' of the parent
# data matrix.  The hinge is located at 'cut'.  If 'dir' == true, the
# hinge is max(0, x - cut), otherwise it is max(0, cut - x).
struct Hinge
    var::Int
    cut::Float64
    dir::Bool
end

# A MARSTerm is a product of hinge functions.
struct MARSTerm

    # The hinge functions whose product forms this MARS term.
    hinges::Vector{Hinge}

    # Indicators of which variables in the parent data matrix are active
    # in this term.
    vmask::Vector{Bool}
end

mutable struct EarthModel

    # All terms in the model
    Terms::Vector{MARSTerm}

    # The basis vectors, D[1] is always the intercept
    D::Vector{Vector{Float64}}

    # Orthogonalized basis vectors, removed after fitting.
    U::Vector{Vector{Float64}}

    # The residuals
    resid::Vector{Float64}

    # A set of variable masks.  If not empty, only terms
    # whose variables match one of the masks are allowed.
    # If empty, no constraints are imposed.
    constraints::Set{Vector{Bool}}

    # Knots for each variable
    K::Vector{Vector{Float64}}

    # The covariates, rows are observations and columns are variables.
    X::Matrix{Float64}

    # The response
    y::Vector{Float64}
end

function nobs(E::Earth)
    return length(E.y)
end

function modelmatrix(E::Earth)
    return E.X
end

function get_knots(x::AbstractVector, knots::Int)
    n = length(x)
    pp = range(0, 1, knots)
    return [quantile(x, p) for p in pp]
end

# Constructor
function EarthModel(X::AbstractMatrix, y::AbstractVector, knots; constraints=Set{Vector{Bool}}())
    n, p = size(X)
    if length(y) != n
        throw(ArgumentError("The length of y must match the leading dimension of X."))
    end

    # Always start with an intercept
    icept = Hinge(-1, 0, true)
    vmask = zeros(Bool, p)
    term = MARSTerm(Hinge[icept,], vmask)

    # Fitted values based on the intercept-only model
    yhat = mean(y) * ones(n)
    resid = y - yhat

    K = [get_knots(x, knot) for (x, knot) in zip(eachcol(X), knots)]

    # Initial basis with the intercept-only model
    z = ones(n)
    D = [z]
    U = [z / norm(z)]

    return EarthModel(MARSTerm[term], D, U, resid, constraints, K, X, y)
end

"""
    fit(EarthModel, X, y; knots=20, maxit=10, constraints=Set{Vector{Bool}}(), prune=true, verbose=true)

Fit a regression model using an approach similar to Friedman's 1991
MARS procedure (also known as Earth for trademark reasons).
Earth/MARS involves two steps: a greedy basis construction approach
followed by a pruning approach that eliminates irrelevant terms.  The
basis functions are products of hinge functions.  This implementation
uses the Lasso instead of back-selection to prune the model.

References:

Friedman (1991) "Multivariate Adaptive Regression Splines".
Ann. Statist. 19(1): 1-67 (March, 1991). DOI: 10.1214/aos/1176347963
https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full
"""
function fit(::Type{EarthModel}, X, y; knots=20, maxit=10, constraints=Set{Vector{Bool}}(),
             prune=true, verbose=false)

    if typeof(knots) <: Number
        knots = knots * ones(Int, size(X, 2))
    end

    E = EarthModel(X, y, knots; constraints=constraints)
    fit!(E; maxit=maxit, prune=prune, verbose=verbose)
    return E
end

function fit!(E::EarthModel; maxit=10, prune=true, verbose=verbose)
    for k in 1:maxit
        verbose && println("$(k)...")
        nextterm!(E)
    end
    if prune
        verbose && println("Pruning...")
        prune!(E)
    end
end

# Replace z with its projection onto the orthogonal complement of the
# subspace spanned by the columns in U.  Assumes that the columns in
# U are orthonormal.
function residualize!(U::Vector{Vector{Float64}}, z::Vector{Float64})
    for u in U
        z .-= dot(u, z) * u
    end
end

# Expand the model by adding a new basis function.  The new basis function
# is constructed by multiplying the current term in position 'iterm' by
# the hinge function 'h'.  The vector 'z' contains the current values
# of the basis function in position 'iterm'.
function addterm!(E::EarthModel, iterm::Int, h::Hinge, z::Vector{Float64})
    z = copy(z)
    term = E.Terms[iterm]
    newterm = deepcopy(term)
    push!(newterm.hinges, h)
    newterm.vmask[h.var] = 1
    push!(E.Terms, newterm)
    push!(E.D, copy(z))
    residualize!(E.U, z)
    z ./= norm(z)
    push!(E.U, z)
end

# Check the consistency of the internal state of E (used for debugging).
function checkstate(E::EarthModel)

    (; y, X, D) = E

    DD = hcat(D...)
    yhat = DD * (qr(DD) \ y)
    resid = y - yhat
    c = norm(resid - E.resid)
    if c > 1e-10
        error("residuals do not agree: ", c)
    end

    if length(E.D) != length(E.U)
        error("lengths of D and U do not agree")
    end

    UU = hcat(E.U...)
    UU, _, _ = svd(UU)
    DD, _, _ = svd(DD)
    s = sum(svd(UU' * DD).S)
    if !isapprox(s, size(UU, 2))
        error("U and S spans do not agree: ", s, " ", size(UU, 2))
    end

    if !isapprox(UU' * UU, I(size(UU, 2)))
        error("U is not orthogonal")
    end
end

function addterm!(E::EarthModel, iterm::Int, ivar::Int, icut::Int)

    h1, z1, h2, z2 = mirrorbasis(E, iterm, ivar, icut)

    ya, t1, t2 = fitreg(copy(z1), copy(z2), E.resid, E.U)
    E.resid .-= ya

    t1 && addterm!(E, iterm, h1, z1)
    t2 && addterm!(E, iterm, h2, z2)
end

# Construct a pair of mirror image hinge functions.
function mirrorbasis(E::EarthModel, iterm::Int, ivar::Int, icut::Int)
    (; Terms, K, D, X) = E
    t = Terms[iterm]
    z = D[iterm]
    h1 = Hinge(ivar, K[ivar][icut], true)
    z1 = copy(z)
    update!(h1, X, z1)

    h2 = Hinge(ivar, K[ivar][icut], false)
    z2 = copy(z)
    update!(h2, X, z2)

    return h1, z1, h2, z2
end

function fitreg(z1, z2, resid, U)

    n = length(z1)

    # Residualize against the existing columns
    residualize!(U, z1)
    residualize!(U, z2)

    fitval = zeros(n)
    nz1 = norm(z1)
    t1 = nz1 > 1e-10
    nz2 = norm(z2)
    t2 = nz2 > 1e-10

    if t1
        fitval .+= z1 * dot(z1, resid) / nz1^2
        z2 .-= z1 * dot(z1, z2) / nz1^2
    end

    if t2
        fitval .+= z2 * dot(z2, resid) / nz2^2
    end

    return fitval, t1, t2
end

# Evaluate the improvement in fit for adding the given term to the model.
function checkvar(E::EarthModel, iterm::Int, ivar::Int, icut::Int)

    (; D, U, y, X, K, resid) = E

    h1, z1, h2, z2 = mirrorbasis(E, iterm, ivar, icut)
    ya, _, _ = fitreg(z1, z2, resid, U)

    return sum(abs2, resid - ya)
end

# Check if the given term can be added to the model.
function isvalid(E::EarthModel, iterm::Int, ivar::Int)

    (; Terms, constraints) = E
    if length(constraints) == 0
        return true
    end

    t = Terms[iterm]
    s = t.vmask[ivar]
    t.vmask[ivar] = 1
    f = t.vmask in constraints
    t.vmask[ivar] = s
    return f
end

# Add the next best-fitting term to the model.
function nextterm!(E::EarthModel)

    (; K, X) = E
    n, p = size(X)

    ii, jj, kk = -1, -1, -1
    rr = Inf

    for i in eachindex(E.Terms)
        for j in 1:p
            for k in eachindex(K[j])
                if !isvalid(E, i, j)
                    continue
                end
                r = checkvar(E, i, j, k)
                if r < rr
                    ii, jj, kk = i, j, k
                    rr = r
                end
            end
        end
    end

    if ii != -1
        addterm!(E, ii, jj, kk)
    end
end

function prune!(E)

    (; y, D) = E

    # Design matrix excluding the intercept
    X = hcat(D[2:end]...)

    m = fit(LassoModel, X, y)
    c = Lasso.coef(m)

    ii = findall(c .!= 0)
    E.Terms = E.Terms[ii]
    E.D = E.D[ii]
    E.U = []

    D = hcat(E.D...)
    E.resid = y - D * (qr(D) \ y)
end

function update!(h::Hinge, X::AbstractMatrix, z::AbstractVector)
    if h.var < 0
        return
    end
    n = size(X, 1)
    for i in 1:n
        if h.dir
            z[i] *= max(0.0, X[i, h.var] - h.cut)
        else
            z[i] *= max(0.0, h.cut - X[i, h.var])
        end
    end
end
