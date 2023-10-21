using Literate

if length(ARGS) != 1
    error("build.jl requires one argument, the name of the script to be built")
end

Literate.markdown(ARGS[1]; execute=true, flavor=Literate.CommonMarkFlavor())
