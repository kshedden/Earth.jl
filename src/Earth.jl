module Earth
    import StatsAPI: fit, nobs, modelmatrix, coefnames, coef
    import Base: show

    export fit, nobs, modelmatrix, response, residuals, predict, coefnames, coef
    export EarthModel, gcv, gr2, show

    include("fitearth.jl")
end
