using Literate

if length(ARGS) != 1
    error("usage: julia --project=. build.jl name.jl")
end

Literate.markdown(ARGS[1]; execute=true)
