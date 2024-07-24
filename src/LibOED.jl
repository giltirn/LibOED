module LibOED

export simulate_inference, replace_param_names!

using Distributed
using .Threads
using Turing
using DynamicPPL
using Random
using NamedArrays

include("inference.jl")

end # module LibOED
