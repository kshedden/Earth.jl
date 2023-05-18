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

# A MarsTerm is a product of hinge functions.
struct MarsTerm

    # The hinge functions whose product forms this MARS term.
    hinges::Vector{Hinge}

    # Indicators of which variables in the parent data matrix are active
    # in this term.
    vmask::Vector{Bool}
end

mutable struct EarthModel

    # All terms in the model
    Terms::Vector{MarsTerm}

    # The basis vectors, D[1] is always the intercept
    D::Vector{Vector{Float64}}

    # Orthogonalized basis vectors (removed after fitting).
    U::Vector{Vector{Float64}}

    # The residuals
    resid::Vector{Float64}

    # A set of variable masks.  If not empty, only terms
    # whose variables match one of the masks are allowed.
    # If empty, no constraints are imposed.
    constraints::Set{Vector{Bool}}

    # Knots for each variable
    K::Vector{Vector{Float64}}

    # The coefficients
    coef::Vector{Float64}

    # Maximum number of variables in any single term.
    maxorder::Int

    # The residual sum of squares at each forward iteration
    rss::Vector{Float64}

    # The effective degrees of freedom (dof) in each forward iteration
    edof::Vector{Float64}

    # Number of terms in each forward iteration
    nterms::Vector{Float64}

    # The penalty for each knot that is added to the model
    knot_penalty::Float64

    # The covariates, rows are observations and columns are variables.
    X::Matrix{Float64}

    # The response
    y::Vector{Float64}
end

function response(E::EarthModel)
    return E.y
end

function residuals(E::EarthModel)
    return E.resid
end

function predict(E::EarthModel)
    return response(E) - residuals(E)
end

function predict(E::EarthModel, X::AbstractMatrix)

    m, q = size(X)
    d = length(E.D)
    Z = ones(m, d)
    for (j,t) in enumerate(E.Terms)
        for h in t.hinges
            update!(h, X, @view(Z[:, j]))
        end
    end

    return Z * E.coef
end

function nobs(E::EarthModel)
    return length(E.y)
end

function modelmatrix(E::EarthModel)
    return E.X
end

"""
    gcv(E::EarthModel)

Returns the generalized cross validation (GCV) statistics at each
step of the forward pass.
"""
function gcv(E::EarthModel)
    n = nobs(E)
    return (E.rss ./ n) ./ (1 .- E.edof/n).^2
end

"""
     gr2(E::EarthModel)

Returns the generalized R-square (adjusted for model complexity).
"""
function gr2(E::EarthModel)
    g = gcv(E)
    return 1 .- g / g[1]
end

function get_knots(x::AbstractVector, knots::Int)
    n = length(x)
    pp = range(0, 1, knots)
    return [quantile(x, p) for p in pp]
end

# Constructor
function EarthModel(X::AbstractMatrix, y::AbstractVector, knots; constraints=Set{Vector{Bool}}(),
                    maxorder=2, knot_penalty=2)
    n, p = size(X)
    if length(y) != n
        throw(ArgumentError("The length of y must match the leading dimension of X."))
    end

    # Always start with an intercept
    icept = Hinge(-1, 0, true)
    vmask = zeros(Bool, p)
    term = MarsTerm(Hinge[icept,], vmask)

    # Fitted values based on the intercept-only model
    yhat = mean(y) * ones(n)
    resid = y - yhat

    K = [get_knots(x, knot) for (x, knot) in zip(eachcol(X), knots)]

    # Initial basis consisting only of the intercept
    z = ones(n)
    D = [z]
    U = [z / norm(z)]
    edof = Float64[1]
    rss = Float64[sum(abs2, resid)]
    nterms = Int[1]

    return EarthModel(MarsTerm[term], D, U, resid, constraints, K, [],
                      maxorder, rss, edof, nterms, knot_penalty, X, y)
end

"""
    fit(EarthModel, X, y; knots=20, maxit=10, constraints=Set{Vector{Bool}}(),
        prune=true, verbose=false, knot_penalty=2, maxorder=2)

Fit a regression model using an approach similar to Friedman's 1991
MARS procedure (also known as Earth for trademark reasons).  The
design matrix `X` is a design matrix of covariates (do not include
an intercept), and the vector `y` contains the response values.

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
             prune=true, verbose=false, maxorder=2, knot_penalty=ifelse(maxorder>1, 3, 2))

    if typeof(knots) <: Number
        knots = knots * ones(Int, size(X, 2))
    end

    E = EarthModel(X, y, knots; constraints=constraints, knot_penalty=knot_penalty, maxorder=2)
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
    else
        D = hcat(E.D...)
        E.coef = qr(D) \ E.y
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
    term = E.Terms[iterm]
    newterm = deepcopy(term)
    push!(newterm.hinges, h)
    newterm.vmask[h.var] = 1
    push!(E.Terms, newterm)
    push!(E.D, copy(z))
    z = copy(z)
    residualize!(E.U, z)
    z ./= norm(z)
    push!(E.U, z)
end

# Check the consistency of the internal state of E (used for debugging).
# This should run without error up to the pruning step, but not after
# the pruning step.
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
    if !isapprox(UU' * E.resid, zeros(size(UU, 2)), atol=1e-10)
        error("Residuals are not orthogonal to U")
    end

    DD, _, _ = svd(DD)
    s = sum(svd(UU' * DD).S)
    if !isapprox(s, size(UU, 2))
        error("U and D spans do not agree: ", s, " ", size(UU, 2))
    end

    if !isapprox(UU' * UU, I(size(UU, 2)))
        error("U is not orthogonal")
    end

    return true
end

function addterm!(E::EarthModel, iterm::Int, ivar::Int, icut::Int)

    n = nobs(E)
    z1 = zeros(n)
    z2 = zeros(n)
    h1, h2 = mirrorbasis(E, iterm, ivar, icut, z1, z2)

    ya, t1, t2 = fitreg(E, copy(z1), copy(z2))
    E.resid .-= ya

    t1 && addterm!(E, iterm, h1, z1)
    t2 && addterm!(E, iterm, h2, z2)

    push!(E.edof, last(E.edof) + t1 + t2 + (t1 || t2) * E.knot_penalty)
    push!(E.rss, sum(abs2, E.resid))
    push!(E.nterms, length(E.D))
end

# Construct a pair of mirror image hinge functions.
function mirrorbasis(E::EarthModel, iterm::Int, ivar::Int, icut::Int, z1::Vector{Float64}, z2::Vector{Float64})
    (; Terms, K, D, X) = E
    t = Terms[iterm]
    z = D[iterm]
    h1 = Hinge(ivar, K[ivar][icut], true)
    z1 .= z
    update!(h1, X, z1)

    h2 = Hinge(ivar, K[ivar][icut], false)
    z2 .= z
    update!(h2, X, z2)

    return h1, h2
end

function fitreg(E::EarthModel, z1, z2)

    (; U, resid) = E

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
        fitval .+= z2 * (dot(z2, resid) - dot(z2, fitval)) / dot(z2, z2)
    end

    return fitval, t1, t2
end

# Evaluate the improvement in fit for adding the given term to the model.
function checkvar(E::EarthModel, iterm::Int, ivar::Int, icut::Int, z1::Vector{Float64}, z2::Vector{Float64})

    (; D, U, y, X, K, resid) = E

    n = nobs(E)
    h1, h2 = mirrorbasis(E, iterm, ivar, icut, z1, z2)
    ya, _, _ = fitreg(E, z1, z2)

    return sum(abs2, resid - ya)
end

# Check if the given term can be added to the model.
function isvalid(E::EarthModel, iterm::Int, ivar::Int)
    (; Terms, constraints, maxorder) = E
    t = Terms[iterm]
    s = t.vmask[ivar]
    t.vmask[ivar] = 1
    f1 = length(constraints) > 0 ? t.vmask in constraints : true
    f2 = sum(t.vmask) <= maxorder
    t.vmask[ivar] = s
    return f1 && f2
end

# Add the next best-fitting basis term to the model.
function nextterm!(E::EarthModel)

    (; K, X) = E
    n, p = size(X)
    z1 = zeros(n)
    z2 = zeros(n)

    ii, jj, kk = -1, -1, -1
    rr = Inf

    for i in eachindex(E.Terms)
        for j in 1:p
            for k in eachindex(K[j])
                if !isvalid(E, i, j)
                    continue
                end
                r = checkvar(E, i, j, k, z1, z2)
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

# Use the Lasso to drop irrelevant terms from the model.
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
    E.coef = qr(D) \ y
    E.resid = y - D * E.coef
end

# Update the basis vector z in-place by multiplying it by the
# given hinge function.
function update!(h::Hinge, X::AbstractMatrix, z::AbstractVector)
    if h.var < 0
        # If var < 0, the hinge function is the intercept.
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
