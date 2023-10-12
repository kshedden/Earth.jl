module Earth
    import StatsAPI: fit, nobs, modelmatrix, coefnames, coef

    export fit, nobs, modelmatrix, response, residuals, predict, coefnames, coef
    export EarthModel, gcv, gr2

    include("fitearth.jl")
end
