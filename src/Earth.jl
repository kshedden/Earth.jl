module Earth
    import StatsAPI: fit, nobs, modelmatrix

    export fit, nobs, modelmatrix, response, residuals, predict
    export EarthModel, gcv, gr2

    include("earth.jl")
end
