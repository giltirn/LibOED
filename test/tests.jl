# using BenchmarkTools
# using CSV
# using DataFrames
# using ForwardDiff
# using Plots
# using Plots.PlotMeasures
# using Optimization
# using OptimizationOptimJL
# using Statistics
# using StatsPlots

using .Threads

using Distributed
@everywhere using LinearAlgebra
@everywhere using NamedArrays
@everywhere using Turing
@everywhere using LibOED

function test_divide_work()
    #Divides equally
    N_samp = 6
    nproc = 3
    r = LibOED.divide_work(N_samp, nproc)
    show(stdout,"text/plain", r)
    if r[1] != 2 || r[2] != 2 || r[3] !=2
        error("test_divide_work failed 1")
    end

    #Doesn't divide equally, should fill from end
    N_samp = 8
    nproc = 3
    r = LibOED.divide_work(N_samp, nproc)
    show(stdout,"text/plain", r)
    if r[1] != 2 || r[2] != 3 || r[3] !=3
        error("test_divide_work failed 2")
    end
end

@everywhere @model function test_inference_model_dist(y, d)
    σ ~ LogNormal(log(1), log(2)/2) #data noise
    a ~ Normal(1.2,2) #parameter with prior
    m = a * d    
    y ~ MvNormal(m, σ*I) #y=2d + eps
    return mean(m) # outputs mean of model
end

function test_inference()
    obs_sz = 10
    N_samp = 5
    d=Array{Float64}(undef, obs_sz)
    y=Array{Float64,2}(undef, obs_sz, N_samp)
    for i in 1:obs_sz
        d[i] = i
        for s in 1:N_samp
            y[i,s] = 2*i + randn()
        end
    end

    println("Y-samples:")
    for s in 1:N_samp
        println(y[:,s])
    end

    try
        inf=LibOED.simulate_inference(test_inference_model_dist,d, y, chain_length=100, dist_properties=nothing, param_dist_properties=nothing)
    catch e
        println("Got expected error: ",e)
    end
        
    #Test with dist_properties
    inf=LibOED.simulate_inference(test_inference_model_dist,d, y, chain_length=100, dist_properties=[var,mean,std], param_dist_properties=nothing)
    println("Test with dist_properties")
    show(stdout,"text/plain",inf)

    #Test with param_dist_properties
    inf=LibOED.simulate_inference(test_inference_model_dist,d, y, chain_length=100, dist_properties=nothing, param_dist_properties=[var])
    println("Test with param_dist_properties")
    show(stdout,"text/plain",inf)

    #Test with both
    inf=LibOED.simulate_inference(test_inference_model_dist,d, y, chain_length=100, dist_properties=[mean], param_dist_properties=[var])
    println("Test with dist_properties and param_dist_properties")
    show(stdout,"text/plain",inf)
   
end

@everywhere @model function test_inference_extra_params_model_dist(y, d, smu::Float64, ssig::Float64)
    σ ~ LogNormal(smu, ssig)
    y ~ MvNormal(2*d, σ*I) #y=2d + eps    
    return mean(y) # outputs mean of model
end

function test_inference_extra_params()
    obs_sz = 10
    N_samp = 4
    d=Array{Float64}(undef, obs_sz)
    y=Array{Float64,2}(undef, obs_sz, N_samp)
    for i in 1:obs_sz
        d[i] = i
        for s in 1:N_samp
            y[i,s] = 2*i + randn()
        end
    end

    smu = log(1)
    ssig = log(2)/2

    println("Y-samples:")
    for s in 1:N_samp
        println(y[:,s])
    end

    mwrap(y,d) = test_inference_extra_params_model_dist(y,d,smu,ssig)
    
    inf=LibOED.simulate_inference(mwrap,d, y;  chain_length=5)
    show(stdout,"text/plain",inf)
end


@everywhere @model function test_inference_extra_params2_model_dist(y, d, s::Array{Float64})
    σ ~ LogNormal(s[1], s[2])
    y ~ MvNormal(2*d, σ*I) #y=2d + eps    
    return mean(y) # outputs mean of model
end

function test_inference_extra_params2()
    obs_sz = 10
    N_samp = 4
    d=Array{Float64}(undef, obs_sz)
    y=Array{Float64,2}(undef, obs_sz, N_samp)
    for i in 1:obs_sz
        d[i] = i
        for s in 1:N_samp
            y[i,s] = 2*i + randn()
        end
    end

    println("Y-samples:")
    for s in 1:N_samp
        println(y[:,s])
    end

    svals::Array{Float64} = [ log(1), log(2)/2 ]
    mwrap(y,d) = test_inference_extra_params2_model_dist(y,d,svals)
    
    inf=LibOED.simulate_inference(mwrap,d, y;  chain_length=5)
    show(stdout,"text/plain",inf)
end




test_divide_work()
test_inference()
#test_inference_extra_params2()


#TODO:
#Test inference works for models that have no return type
#Test inference works for models that have tuple return types
