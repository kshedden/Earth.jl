using CategoricalArrays
using DataFrames
using Lasso
using LinearAlgebra
using Printf
using Statistics
using StatsBase

# A hinge function for the variable in position 'var' of the parent
# data matrix.  The hinge is located at 'cut'.  If 'dir' == true, the
# hinge is max(0, x - cut), otherwise it is max(0, cut - x).
struct Hinge
    var::Int
    cut::Float64
    dir::Bool
end

# A EarthTerm is a product of hinge functions.
struct EarthTerm

    # The hinge functions whose product forms this MARS term.
    hinges::Vector{Hinge}

    # Number of times that each variable in the parent data matrix are active
    # in this term.
    vmask::Vector{Int}

    # Indicators of which variables in the parent data matrix are active
    # in this term.
    bmask::Vector{Int}
end

# Returns the order of a term, which is the number of distinct variables
# in the term.
function order(T::EarthTerm)
    return sum(T.bmask)
end

# Returns the degree of a term, which is the maximum number of hinge
# functions for a single variable that occur in the term.
function degree(T::EarthTerm)
    return maximum(T.vmask)
end

mutable struct EarthConfig

    # Maximum number of iterations in the forward pass
    maxit::Int

    # Maximum number of variables in any single term.
    maxorder::Int

    # The maximum number of times that a variable can
    # be present in a single term
    maxdegree::Int

    # The total number of knots, or the number of knots per variable
    num_knots::Vector{Int}

    # The penalty for each knot that is added to the model
    knot_penalty::Float64

    # A set of variable masks.  If not empty, only terms
    # whose variables match one of the masks are allowed.
    # If empty, no constraints are imposed.
    constraints::Set{Vector{Bool}}

    # Stop adding new basis functions once the two members of
    # a mirror pair of hinge functions adds less than this
    # amount to the R^2.
    min_r2::Float64

    # Drop variables whose Lasso coefficient estimate is smaller
    # than this number in absolute value.
    min_coef::Float64

    # After pruning, how to refit the coefficients
    refit::Symbol
end


"""
TODO function signature

# Keyword arguments

- `num_knots`: The number of hinge function knots for each variable.
- `maxit`: The number of basis construction iterations.
- `constraints`: A set of bit vectors that constrain the combinations of variables that can be used to produce a term.
- `prune`: If false, perform the basis construction step but do not perform the pruning step.
- `verbose`: Print some information as the fitting algorithm runs.
- `maxorder`: The maximum number of distinct variables that can be present in a single term.
- `maxdegree`: The maximum number of hinges for a single variable that can occur in one term
- `knot_penalty`: A parameter that controls how easily a new term can enter the model.
- `min_r2`: Terminate the forward pass if the increase in R2 falls below this value.
- `min_coef`: Terms with standardized coefficient falling below this value are pruned.
- `refit`: After pruning, refit the model using either lasso (:lasso) or OLS (:ols).
"""
function EarthConfig(; constraints=Set{Vector{Bool}}(), num_knots=20, maxit=10, maxorder=2, maxdegree=2,
                       knot_penalty=ifelse(maxorder>1, 3, 2), min_r2=0.001, min_coef=0.01,
                       refit::Symbol=:lasso)

    # The num_knots argument can be either scalar or vector.  Here construct
    # num_knotsv that is always a vector.  Since we don't know the number
    # of variables yet, a scalar is stored as a vector of length 1.
    num_knotsv = if typeof(num_knots) <: Int
        Int[num_knots,]
    elseif typeof(num_knots) <: Vector{<:Int}
        num_knots
    else
        error("Invalid type for `num_knots` ($(typeof(num_knots))) in EarthConfig")
    end

    return EarthConfig(maxit, maxorder, maxdegree, num_knotsv, knot_penalty, constraints,
                       min_r2, min_coef, refit)
end

mutable struct EarthModel

    # Parameters that control how the model is fit
    config::EarthConfig

    # All terms in the model
    Terms::Vector{EarthTerm}

    # The basis vectors, D[1] is always the intercept
    D::Vector{Vector{Float64}}

    # Orthogonalized basis vectors (removed after fitting).
    U::Vector{Vector{Float64}}

    # The residuals
    resid::Vector{Float64}

    # Knots for each variable
    K::Vector{Vector{Float64}}

    # The coefficients
    coef::Vector{Float64}

    # The residual sum of squares at each forward iteration
    rss::Vector{Float64}

    # The effective degrees of freedom (dof) in each forward iteration
    edof::Vector{Float64}

    # Number of terms in each forward iteration
    nterms::Vector{Float64}

    # The covariates, rows are observations and columns are variables.
    X::Matrix{Float64}

    # The response
    y::Vector{Float64}

    # The mean of y
    meany::Float64

    # The standard deviation of y
    sdy::Float64

    # Mean X
    meanx::Vector{Float64}

    # Standard deviations of X
    sdx::Vector{Float64}

    # Variable names
    vnames::Vector{String}

    # The levels of categorical variables.  The elements of levels are
    # vectors containing the unique levels of each categorical variable,
    # or the empty vector [] if a variable is not categorical.
    levels::Vector
end

# Returns the orders of all terms in the model.
function order(E::EarthModel)
    return order.(E.Terms)
end

# Returns the degrees of all terms in the model.
function degree(E::EarthModel)
    return degree.(E.Terms)
end

function response(E::EarthModel)
    (; y, meany, sdy) = E
    return meany .+ sdy * y
end

function residuals(E::EarthModel)
    (; resid, sdy) = E
    return sdy * resid
end

function predict(E::EarthModel)
    (; resid, meany, sdy) = E
    return response(E) - residuals(E)
end

function coefnames(E::EarthModel)
    return E.vnames
end

function coef(E::EarthModel)
    return E.coef
end

function predict(E::EarthModel, X)

    (; levels, vnames, meany, sdy, meanx, sdx) = E

    cols, _, _ = handle_covars(X)
    _, X = build_design(cols, levels, vnames)

    # Standardize using the mean/sd parameters
    # derived from the training data.
    for j in 1:size(X, 2)
        X[:, j] = (X[:, j] .- meanx[j]) ./ sdx[j]
    end

    m, q = size(X)
    d = length(E.D)
    Z = ones(m, d)
    for (j,t) in enumerate(E.Terms)
        for h in t.hinges
            update!(h, X, @view(Z[:, j]))
        end
    end

    return meany .+ sdy * (Z * E.coef)
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
function EarthModel(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, config::EarthConfig;
                    vnames::Vector{<:AbstractString}=[], levels::Vector=[])
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

     # Expand num_knots to align with the number of covariates in X.
    if length(config.num_knots) == 1
        config.num_knots = fill(config.num_knots[1], size(X, 2))
    elseif length(config.num_knots) != size(X, 2)
        error("Number of knots ($(length(knots))) must be equal to the number of variables ($(size(X, 2)))")
    end

    # Standardize y, the iterative orthogonalizations used in fitting Earth
    # models exhibit a lot of roundoff error if y is not well-scaled.
    mny = mean(y)
    sdy = std(y)
    y = (y .- mny) ./ sdy

    # Always start with an intercept
    icept = Hinge(-1, 0, true)
    vmask = zeros(Int, p)
    bmask = zeros(Bool, p)
    term = EarthTerm(Hinge[icept,], vmask, bmask)

    # Fitted values based on the intercept-only model
    yhat = mean(y) * ones(n)
    resid = y - yhat

    # Standardize columns of X
    mnx = mean(X; dims=1)[:]
    sdx = std(X; dims=1)[:]
    for j in 1:size(X, 2)
        X[:, j] = (X[:, j] .- mnx[j]) ./ sdx[j]
    end

    # Get the knots for each variable.
    knots = [get_knots(x, knot) for (x, knot) in zip(eachcol(X), config.num_knots)]

    # Setup an initial basis consisting only of the intercept
    z = ones(n)
    D = [z]
    U = [z ./ sqrt(n)]
    edof = Float64[1]
    rss = Float64[sum(abs2, resid)]
    nterms = Int[1]

    return EarthModel(config, EarthTerm[term], D, U, resid, knots, [],
                      rss, edof, nterms, X, y, mny, sdy, mnx, sdx, vnames,
                      levels)
end

# Returns a column iterator, a vector of levels, and a vector of names based on the
# data in `X`, which can be (1) a numeric array, (2) a dataframe, (3) a named tuple.
# (4) a tuple, or (5) a vector.  In cases 3-5 the values/elements are vectors
# (data columns).  In cases 1-2 the columns of the array or dataframe are the
# data vectors.  In case 1 all data must be numeric.  In cases 2-5 data columns can
# either numeric or categorical, in the latter case one-hot encoded indicator
# vectors are created.
function handle_covars(X)

    # Get variable names (creating generic names if needed) and
    # a sequence of data columns.
    nams, cols = if typeof(X) <: AbstractMatrix{<:Real}
        na = ["v$(j)" for j in 1:size(X, 2)]
        na, eachcol(Float64.(X))
    elseif typeof(X) <: DataFrame
        names(X), eachcol(X)
    elseif typeof(X) <: NamedTuple
        String.([keys(X)...]), values(X)
    elseif typeof(X) <: Tuple || typeof(X) <: AbstractVector
        ["v$(j)" for j in 1:length(X)], X
    else
        error("Invalid type $(typeof(X)) for covariates `X`")
    end

    levs = [typeof(c) <: CategoricalArray ? levels(c) : [] for c in cols]

    if length(cols) == 0
        error("No covariates")
    end

    return cols, levs, nams
end

# Given a column iterator, levels vector, and names vector, construct a
# design matrix and a vector of column names.  Categorical columns
# are expanded to indicator arrays, and names for each indicator
# column are constructed.
function build_design(cols, levs, nams)

    # Prepare to insert the expanded names into nams
    nams = Vector{Any}(nams)

    A = []
    n = length(first(cols))
    for (j,c) in enumerate(cols)
        if length(c) != n
            error("Variables 1 and $(j) have different lengths ($(n) and $(length(c))")
        end

        # The variable name, which is also the column name for numeric columns.
        # For non-numeric columns, this becomes the root of the indicator name.
        a = nams[j]

        if eltype(c) <: Real
            push!(A, Float64.(c))
        elseif eltype(c) <: CategoricalValue
            nams[j] = ["$(a)::$(x)" for x in levs[j]]
            push!(A, (levs[j] .== permutedims(c))')
        else
            error("Unknown type `$(eltype(c))` for covariate $(j)")
        end
    end

    # Flatten the names vector
    nams = reduce(vcat, nams)

    return nams, hcat(A...)
end


"""
    fit(EarthModel, X, y; config::EarthConfig=EarthConfig(), prune=true, verbose=false)

Fit a regression model using an approach similar to Friedman's 1991
MARS procedure (also known as Earth for trademark reasons).  The
covariates are in `X` and the vector `y` contains the response values.

Earth/MARS involves two steps: a greedy basis construction followed by
pruning step that aims to eliminate irrelevant terms.  The basis
functions are products of hinge functions.  This implementation uses
the Lasso instead of back-selection to prune the model.

The covariates `X` can be a numeric Matrix, a data frame, a vector or
tuple of vectors, or a named tuple whose values are vectors.  In the
latter two cases, each covariate vector must be of numeric or string
type, or an instance of CategoricalArray. The latter-two types are expanded
into binary indicator vectors.

The `config` argument can be used to specify many aspects of how the model
is fit.  See `EarthConfig` for more specifics.

References:

Friedman (1991) "Multivariate Adaptive Regression Splines".
Ann. Statist. 19(1): 1-67 (March, 1991). DOI: 10.1214/aos/1176347963
https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full
"""
function fit(EarthModel, X, y; config::EarthConfig=EarthConfig(), prune=true, verbose=false)

    cols, levs, nams = handle_covars(X)
    vnames, X = build_design(cols, levs, nams)

    E = EarthModel(X, y, config; vnames=vnames, levels=levs)
    fit!(E; prune=prune, verbose=verbose)
    return E
end

function fit!(E::EarthModel; prune=true, verbose=verbose)

    cfg = E.config

    # Basis construction
    kp = 0
    for k in 1:cfg.maxit
        kp += k
        rr = nextterm!(E)
        if rr < cfg.min_r2
            break
        end
        if verbose
            println(@sprintf("Increase in R^2: %.4f", rr))
        end
    end

    # Pruning
    if prune
        prune!(E; verbose=verbose)
    else
        # Use all terms and estimate the coefficients using OLS
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

# Check the consistency of the internal state of E.  This function is
# used only for debugging.  This function should run without error
# after building the basis functions but before performing the pruning
# step.
function checkstate(E::EarthModel)

    (; y, X, D, resid) = E

    if abs(mean(resid)) > 1e-10
        error("residuals do not have mean zero")
    end

    DD = hcat(D...)
    yhat = DD * (qr(DD) \ y)
    resid1 = y - yhat
    c = norm(resid1 - resid)
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
    newterm.vmask[h.var] += 1
    newterm.bmask[h.var] = true
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

# Expand the model by adding new basis functions corresponding to a
# mirror pair of hinge functions.  The new basis functions are
# constructed by multiplying the current term in position 'iterm' by
# hinge functions derived from variable `ivar`, using the knot in
# position `icut` of the member `K` of the EarthModel.  Two basis
# functions are constructed corresponding to the two orientations of
# the hinge functions.  A hinge function is not added if it is within
# `qtol` of being identically zero, so the effect of this function is
# to add 0, 1, or 2 hinge functions.
function addtermpair!(E::EarthModel, iterm::Int, ivar::Int, icut::Int, qtol::Float64=1e-10)

    (; config) = E
    n = nobs(E)
    z1 = zeros(n)
    z2 = zeros(n)
    h1, h2 = mirrorbasis(E, iterm, ivar, icut, z1, z2)

    fitval = zeros(n)
    t1, t2 = fitreg(E, copy(z1), copy(z2), fitval; qtol=qtol)
    E.resid .-= fitval

    t1 && addterm!(E, iterm, h1, z1)
    t2 && addterm!(E, iterm, h2, z2)

    push!(E.edof, last(E.edof) + t1 + t2 + (t1 || t2) * config.knot_penalty)
    push!(E.rss, sum(abs2, E.resid))
    push!(E.nterms, length(E.D))

    return sum(abs2, fitval) / length(E.y)
end

# Construct a pair of mirror image hinge functions.  The term with
# index 'iterm' is multiplied by left and right hinge functions based
# on the variable with index 'ivar'.  The hinge functions are
# constructed using the knot at position 'icut'.  The newly
# constructed terms are written into the arrays 'z1' and 'z2'.
function mirrorbasis(E::EarthModel, iterm::Int, ivar::Int, icut::Int,
                     z1::Vector{Float64}, z2::Vector{Float64})
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

# Calculate fitted values for the residuals in a model that adds
# variables `z1` and `z2` to the current model.  These variables are
# only included if they have norm exceeding `qtol` when residualized
# against all terms that are already in the model.  This function
# returns updated fitted values (to the residual), and indicators of
# whether each of the two new terms was used to determine the fit.
function fitreg(E::EarthModel, z1, z2, fitval; qtol::Float64=1e-10)

    (; U, resid) = E

    n = length(z1)
    fitval .= 0

    residualize!(E.U, z1)
    residualize!(E.U, z2)

    nz1 = norm(z1)
    t1 = nz1 > qtol
    if t1
        fitval .+= z1 * dot(z1, resid) / nz1^2
        z2 -= z1*dot(z1, z2) / dot(z1, z1)
    end

    nz2 = norm(z2)
    t2 = nz2 > qtol
    if t2
        fitval .+= z2 * dot(z2, resid) / nz2^2
    end

    return t1, t2
end

# Returns the RSS after adding the given term to the model.
function checkvar(E::EarthModel, iterm::Int, ivar::Int, icut::Int, z1::Vector{Float64},
                  z2::Vector{Float64}, fitval::Vector{Float64})

    (; D, U, y, X, K, resid) = E

    n = nobs(E)
    h1, h2 = mirrorbasis(E, iterm, ivar, icut, z1, z2)
    fitreg(E, z1, z2, fitval)

    return sum(abs2, resid - fitval)
end

# Check if the given term can be added to the model.  A term cannot be
# added to the model if its order exceeds `maxorder` or if it contains
# a combination of terms that is not in `constraints`.  If
# `constraints` is empty then the latter condition is not tested.
function isvalid(E::EarthModel, iterm::Int, ivar::Int)
    (; Terms, config) = E
    (; constraints, maxorder, maxdegree) = config
    t = Terms[iterm]
    s = t.bmask[ivar]
    t.bmask[ivar] = true
    t.vmask[ivar] += 1
    f1 = length(constraints) > 0 ? t.bmask in constraints : true
    f2 = sum(t.bmask) <= maxorder
    f3 = maximum(t.vmask) <= maxdegree
    t.bmask[ivar] = s
    t.vmask[ivar] -= 1
    return f1 && f2 && f3
end

# Add the next best-fitting basis term to the model.  Returns the
# improvement in R^2 based on the term addition.
function nextterm!(E::EarthModel)

    (; config, Terms, K, X) = E
    n, p = size(X)
    z1 = zeros(n)
    z2 = zeros(n)
    fitval = zeros(n)

    # The parameters of the best term yet seen
    ii, jj, kk = -1, -1, -1
    rr = Inf

    # Each term can be the parent of the new term.
    for i in eachindex(Terms)

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
                r = checkvar(E, i, j, k, z1, z2, fitval)
                if r < rr
                    ii, jj, kk = i, j, k
                    rr = r
                end
            end
        end
    end

    if ii != -1
        return addtermpair!(E, ii, jj, kk)
    end

    return 0.0
end

# Use the Lasso to drop irrelevant terms from the model.
function prune!(E; verbose::Bool=false)

    (; y, D, config) = E

    if length(D) == 0
        # This should never happen
        verbose && println("Empty model")
        return
    elseif length(D) == 1
        verbose && println("No terms to prune")
        E.coef = Float64[mean(y)]
        E.resid .= y - mean(y)
        return
    end

    verbose && println("Pruning...")

    # Design matrix excluding the intercept
    X = hcat(D[2:end]...)

    m = fit(LassoModel, X, y; select=MinBIC())
    c = Lasso.coef(m)

    # Drop the irrelevant terms, always keep the intercept.
    sdx = std(X; dims=1)[:]
    ii = 1 .+ findall(r->abs(r) > config.min_coef, c[2:end] .* sdx)
    ii = vcat(1, ii)
    if verbose
        b = length(c) - length(ii)
        println("Dropping $b terms using LASSO")
    end
    E.Terms = E.Terms[ii]
    E.D = E.D[ii]
    E.U = []

    # Refit with OLS, including the intercept
    D = hcat(E.D...)
    if config.refit == :lasso
        m = fit(LassoModel, D[:, 2:end], y; select=MinBIC())
        E.coef = Lasso.coef(m)
    elseif config.refit == :ols
        E.coef = qr(D) \ y
    else
        error("Invalid refit method '$(config.refit)' (must be either `:lasso` or `:ols`)")
    end
    E.resid .= y - D * E.coef
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

function Base.show(io::IO, h::Hinge; vnames=String[])
    if h.var == -1
        print(io, "intercept")
        return
    end
    vname = if length(vnames) > 0
        vnames[h.var]
    else
        "v$(string(h.var))"
    end
    if h.dir
        print(io, @sprintf("h(%s - %.3f)", vname, h.cut))
    else
        print(io, @sprintf("h(%.3f - %s)", h.cut, vname))
    end
end

function Base.show(io::IO, t::EarthTerm; vnames=String[])
    for (j,h) in enumerate(t.hinges)
        show(io, h; vnames=vnames)
        if j < length(t.hinges)
            print(io, " * ")
        end
    end
end

function Base.show(io::IO, E::EarthModel)
    (; Terms, vnames, coef, D) = E
    print(io, "     Coef    Std coef    Term\n")
    for (j, trm) in enumerate(Terms)
        print(io, @sprintf("%10.3f  ", coef[j]))
        if j == 1
            print(io, "     --      ")
        else
            print(io, @sprintf("%10.3f   ", coef[j] * std(D[j])))
        end
        show(io, trm; vnames=vnames)
        print(io, "\n")
    end
end
