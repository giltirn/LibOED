module LibOED

export simulate_inference

using Distributed
using .Threads
using Turing
using DynamicPPL
using Random
using NamedArrays

include("inference.jl")

end # module LibOED
