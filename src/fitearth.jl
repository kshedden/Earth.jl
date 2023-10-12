using Statistics
using StatsBase
using CategoricalArrays
using LinearAlgebra
using Lasso
using DataFrames

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

    # Variable names
    vnames::Vector{String}
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

function coefnames(E::EarthModel)
    return E.vnames
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

# Return a vector containing `n_knots` knots for the values in `x`.
# If `x` has no more than `n_knots` distinct values, these distinct
# values are the knots.  Otherwise the knots lie on quantiles of `x`
# corresponding to `n_knots` equi-spaced probability points.
function get_knots(x::AbstractVector{<:Real}, n_knots::Int)

    u = unique(x)
    sort!(u)
    if length(u) <= n_knots
        return u
    end
    pp = range(0, 1, n_knots)
    return Float64[quantile(x, p) for p in pp]
end

# Constructor
function EarthModel(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, knots;
                    vnames::Vector{<:AbstractString}=[], constraints=Set{Vector{Bool}}(),
                    maxorder=2, knot_penalty=ifelse(maxorder>1, 3, 2))
    n, p = size(X)
    if length(y) != n
        throw(ArgumentError("The length of y must match the leading dimension of X."))
    end

    if length(vnames) == 0
        vnames = ["v$(j)" for j in 1:size(X,2)]
    end

    if length(vnames) != size(X, 2)
        error("Length of `vnames` is incompatible with size of `X`")
    end

    if length(knots) != size(X, 2)
        error("size of `knots` is incompatible with size of `X`")
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
    U = [z ./ sqrt(n)]
    edof = Float64[1]
    rss = Float64[sum(abs2, resid)]
    nterms = Int[1]

    return EarthModel(MarsTerm[term], D, U, resid, constraints, K, [],
                      maxorder, rss, edof, nterms, knot_penalty, X, y, vnames)
end

function _handle_covars(X)

    # Get variable names (creating generic names if needed) and
    # a sequence of data columns.
    nams, cols = if typeof(X) <: AbstractMatrix
        na = ["v$(j)" for j in 1:size(X, 2)]
        return na, Float64.(X)
    elseif typeof(X) <: DataFrame
        names(X), eachcol(X)
    elseif typeof(X) <: NamedTuple
        String.([keys(X)...]), values(X)
    elseif typeof(X) <: Tuple || typeof(X) <: AbstractVector
        ["v$(j)" for j in 1:length(X)], X
    else
        error("Invalid type $(typeof(X)) for covariates `X`")
    end

    if length(cols) == 0
        error("No covariates")
    end

    nams = Vector{Any}(nams)
    A = []
    n = length(first(cols))
    for (j,c) in enumerate(cols)
        if length(c) != n
            error("Variables 1 and $(j) have different lengths ($(n) and $(length(c))")
        end
        a = nams[j]
        if eltype(c) <: Real
            push!(A, Float64.(c))
        elseif eltype(c) <: AbstractString
            m = indicatormat(c)'
            levels = sort(unique(c))
            nams[j] = ["$(a)$(x)" for x in levels]
            push!(A, indicatormat(c)')
        elseif eltype(c) <: CategoricalArray
            levels = sort(unique(c))
            nams[j] = ["$(a)$(x)" for x in levels]
            push!(a, (levels .== c)')
        else
            error("Unknown type `$(eltype(c))` for covariate $(j)")
        end
    end

    # Flatten the names vector
    nams = reduce(vcat, nams)

    return nams, hcat(A...)
end

"""
    fit(EarthModel, X, y; knots=20, maxit=10, constraints=Set{Vector{Bool}}(),
        prune=true, verbose=false, knot_penalty=ifelse(maxorder>1, 3, 2), maxorder=2)

Fit a regression model using an approach similar to Friedman's 1991
MARS procedure (also known as Earth for trademark reasons).  The
covariates are in `X` and the vector `y` contains the response values.

Earth/MARS involves two steps: a greedy basis construction followed by
pruning step that aims to eliminate irrelevant terms.  The basis
functions are products of hinge functions.  This implementation uses
the Lasso instead of back-selection to prune the model.

The covariates `X` can be a numeric Matrix, a vector or tuple of vectors,
or a named tuple whose values are vectors.  In the latter two cases, each
covariate vector must be of numeric or string type, or be an instance of
CategoricalArray. The latter-two types are expanded into binary indicator
vectors.

# Keyword arguments

- `knots`: The number of hinge function knots for each variable.
- `maxit`: The number of basis construction iterations.
- `constraints`: A set of bit vectors that constrain the combinations of variables that can be used to produce a term.
- `prune`: If false, perform the basis construction step but do not perform the pruning step.
- `verbose`: Print some information as the fitting algorithm runs.
- `knot_penalty`: A parameter that controls how easily a new term can enter the model.
- `maxorder`: The maximum number of distinct variables that can be present in a single term.

References:

Friedman (1991) "Multivariate Adaptive Regression Splines".
Ann. Statist. 19(1): 1-67 (March, 1991). DOI: 10.1214/aos/1176347963
https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full
"""
function fit(::Type{EarthModel}, X, y; knots=20, maxit=10, constraints=Set{Vector{Bool}}(),
             prune=true, verbose=false, maxorder=2, knot_penalty=ifelse(maxorder>1, 3, 2))

    vnames, X = _handle_covars(X)

    if typeof(knots) <: Number
        knots = knots * ones(Int, size(X, 2))
    end

    E = EarthModel(X, y, knots; vnames=vnames, constraints=constraints,
                   knot_penalty=knot_penalty, maxorder=2)
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
# subspace spanned by the columns in U.  Assumes that the columns in U
# are orthonormal.
function residualize!(U::Vector{Vector{Float64}}, z::Vector{Float64})
    for u in U
        z .-= dot(u, z) * u
    end
end

# Check the consistency of the internal state of .  This function is
# used only for debugging.  This function should run without error
# after building the basis functions but before performing the pruning
# step.
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

# Expand the model by adding a new basis function.  The new basis
# function is constructed by multiplying the current term in position
# 'iterm' by the hinge function 'h'.  The vector 'z' contains the
# current values of the basis function in position 'iterm'.
function addterm!(E::EarthModel, iterm::Int, h::Hinge, z::Vector{Float64})
    term = E.Terms[iterm]

    # Specification of the term to be added
    newterm = deepcopy(term)
    push!(newterm.hinges, h)
    newterm.vmask[h.var] = 1
    push!(E.Terms, newterm)

    # Add the term data
    push!(E.D, copy(z))
    z = copy(z)

    # Add the unique component of the new term (the part
    # that is orthogonal to all other terms in the model)
    residualize!(E.U, z)
    z ./= norm(z)
    push!(E.U, z)
end

# Expand the model by adding a new basis function.  The new basis
# function is constructed by multiplying the current term in position
# 'iterm' by a hinge function derived from variable `ivar`, using the
# knot in position `icut` of the member `K` of the EarthModel.
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

# Construct a pair of mirror image hinge functions.  The term with
# index 'iterm' is multiplied by left and right hinge functions based
# on the variable with index 'ivar'.  The hinge functions are
# constructed using the knot at position 'icut'.  The newly
# constructed terms are written into the arrays 'z1' and 'z2'.
function mirrorbasis(E::EarthModel, iterm::Int, ivar::Int, icut::Int, z1::Vector{Float64}, z2::Vector{Float64})
    (; Terms, K, D, X) = E
    t = Terms[iterm]
    z = D[iterm]

    # Create the right hinge
    h1 = Hinge(ivar, K[ivar][icut], true)
    z1 .= z
    update!(h1, X, z1)

    # Create the left hinge
    h2 = Hinge(ivar, K[ivar][icut], false)
    z2 .= z
    update!(h2, X, z2)

    return h1, h2
end

# Calculate fitted values for a model that adds variables `z1` and
# `z2` to the current model.  These variables are only included if
# they have norm exceeding `qtol` when residualized against all terms
# that are already in the model.  This function returns updated fitted
# values, and indicators of whether each of the new terms was used to
# determine the fit.
function fitreg(E::EarthModel, z1, z2; qtol::Float64=1e-10)

    (; U, resid) = E

    n = length(z1)

    # Residualize against the existing columns
    residualize!(U, z1)
    residualize!(U, z2)

    fitval = zeros(n)

    # Determine which of the two new terms should be added to
    # the model.  Terms that are almost identically zero are
    # not added.
    nz1 = norm(z1)
    t1 = nz1 > qtol
    nz2 = norm(z2)
    t2 = nz2 > qtol

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

# Check if the given term can be added to the model.  A term cannot be
# added to the model if its order exceeds `maxorder` or if it contains
# a combination of terms that is not in `constraints`.  If
# `constraints` is empty then the latter condition is not tested.
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

    # The parameters of the best term yet seen
    ii, jj, kk = -1, -1, -1
    rr = Inf

    # Each term can be the parent of the new term.
    for i in eachindex(E.Terms)

        # Each variable can combine with the parent term
        # to produce the new term.
        for j in 1:p

            # The variable j is used to produce a hinge function,
            # which requires specification of a knot.
            for k in eachindex(K[j])

                # Skip terms that do not satisfy declared constraints
                if !isvalid(E, i, j)
                    continue
                end

                # Check if this is the best term yet seen
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

# Update the basis vector `z` in-place by multiplying it by the hinge
# function `h`.
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
