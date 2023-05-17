module Earth
    import StatsAPI: fit, nobs, modelmatrix

    export fit, nobs, modelmatrix
    export EarthModel

    include("earth.jl")
end
