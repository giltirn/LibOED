module LibOED

export simulate_inference

using Distributed
using .Threads
using Turing

include("inference.jl")

end # module LibOED
