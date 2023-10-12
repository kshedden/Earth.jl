module Earth
    import StatsAPI: fit, nobs, modelmatrix, coefnames

    export fit, nobs, modelmatrix, response, residuals, predict, coefnames
    export EarthModel, gcv, gr2

    include("fitearth.jl")
end
