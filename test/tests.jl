using .Threads

using Distributed
@everywhere using LinearAlgebra
@everywhere using NamedArrays
@everywhere using Turing
@everywhere using LibOED

using Random

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

function inference_repro_test(model,driver,obs; chain_length=100,dist_properties=nothing, param_dist_properties=nothing, output_chains=false, base_seed=1234)
    println("Running inference using LibOED")
    inf=LibOED.simulate_inference(model, driver, obs, chain_length=chain_length,
                                  dist_properties=dist_properties, param_dist_properties=param_dist_properties, output_chains=output_chains, base_seed=base_seed)
    println("Running chains separately for reproduction")
    nsamp = size(obs,2)   

    dmean = Dict{String,Float64}()
    pmean = Dict{String,Dict{String,Float64}}()
    
    for s in 1:nsamp
        sobs = obs[:,s]

        rng=Xoshiro(base_seed + s)
        chain = sample(rng, model(sobs,driver), NUTS(0.65), chain_length)

        if(dist_properties != nothing)
            Q = generated_quantities(model(sobs,driver), Turing.MCMCChains.get_sections(chain, :parameters))
            for f in dist_properties
                fname=String(nameof(f))
                fexpect = f(Q)                
                fgot=inf.dist_properties[fname][s]
                println("Sample ",s, " dist_properties ", fname, " got ", fgot, " expect ", fexpect)
                if(abs(fexpect-fgot) > 1e-8); error("Failed reproduction test"); end

                if(s==1)
                    dmean[fname] = fexpect
                else
                    dmean[fname] += fexpect
                end
            end

        end
        if(param_dist_properties != nothing)
            S = summarize(chain, param_dist_properties...)
            for f in param_dist_properties
                fname=String(nameof(f))
                if(s==1); pmean[fname] = Dict{String,Float64}(); end
                
                for p in chain.name_map.parameters
                    fexpect = getindex(S,p,nameof(f))
                    fgot=inf.param_dist_properties[fname][String(p)][s]
                    println("Sample ",s, " param_dist_properties ", fname, " ", String(p), " got ", fgot, " expect ", fexpect)
                    if(abs(fexpect-fgot) > 1e-8); error("Failed reproduction test"); end

                    if(s==1)
                        pmean[fname][String(p)] = fexpect
                    else
                        pmean[fname][String(p)] += fexpect
                    end
                end
            end
        end

        if(output_chains)
            got = summarize(inf.chains[s], mean, std)
            expect = summarize(chain, mean, std)
            
            for p in chain.name_map.parameters
                for f in [:mean, :std]                
                    fexpect = getindex(expect,p,f)
                    fgot=getindex(got,p,f)
                    println("Sample ",s, " chain ", f, " ", p, " got ", fgot, " expect ", fexpect)
                    if(abs(fexpect-fgot) > 1e-8); error("Failed reproduction test"); end
                end
            end
        end
    end

    if(dist_properties != nothing)
        println("Checking dist_properties averages")
        for f in dist_properties
            fname=String(nameof(f))
            fexpect = dmean[fname] / nsamp
            fgot = inf.avg_dist_properties[fname]
            println(fname," got ",fgot," expect ",fexpect)
            if(abs(fexpect-fgot) > 1e-8); error("Failed reproduction test"); end
        end
    end

    if(param_dist_properties != nothing)
        println("Checking param_dist_properties averages")
        for f in param_dist_properties
            fname=String(nameof(f))
            for p in keys(pmean[fname])                         
                fexpect = pmean[fname][p] / nsamp
                fgot = inf.avg_param_dist_properties[fname][p]
                println(fname," ",p," got ",fgot," expect ",fexpect)
                if(abs(fexpect-fgot) > 1e-8); error("Failed reproduction test"); end
            end
        end
    end

    
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

    #Check error if no functions provided and not outputing chains
    try
        inf=LibOED.simulate_inference(test_inference_model_dist,d, y, chain_length=100, dist_properties=nothing, param_dist_properties=nothing)
    catch e
        println("Got expected error: ",e)
    end

    println("Test with dist_properties")
    inference_repro_test(test_inference_model_dist,d, y, chain_length=100, dist_properties=[var,mean,std], param_dist_properties=nothing, base_seed=1234)
    println("Test with param_dist_properties")
    inference_repro_test(test_inference_model_dist,d, y, chain_length=100, dist_properties=nothing, param_dist_properties=[var], base_seed=1234)
    println("Test with dist_properties and param_dist_properties")    
    inference_repro_test(test_inference_model_dist,d, y, chain_length=100, dist_properties=[mean], param_dist_properties=[var], base_seed=1234)
    println("Test with just chains output")    
    inference_repro_test(test_inference_model_dist,d, y, chain_length=100, dist_properties=nothing, param_dist_properties=nothing, base_seed=1234, output_chains=true)
end

function test_base_seed_increment()
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

    base_seed_1 = copy(LibOED.global_base_seed[])
    println("Initial base seed ", base_seed_1)
    
    inf=simulate_inference(test_inference_model_dist,d, y, chain_length=10)   

    if(inf.base_seed != base_seed_1); error("Test failed"); end

    base_seed_2 = copy(LibOED.global_base_seed[])
    println("Base seed after first inference ", base_seed_2)
    
    if(base_seed_2 != base_seed_1 + N_samp); error("Test failed"); end

    inf=simulate_inference(test_inference_model_dist,d, y, chain_length=10)   

    if(inf.base_seed != base_seed_2); error("Test failed"); end
    
    base_seed_3 = copy(LibOED.global_base_seed[])
    println("Base seed after second inference ", base_seed_3)
    
    if(base_seed_3 != base_seed_2 + N_samp); error("Test failed"); end

    println("test_base_seed_increment passed")
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




#test_divide_work()
#test_inference()
test_base_seed_increment()
#test_inference_extra_params2()


#TODO:
#Test inference works for models that have no return type
#Test inference works for models that have tuple return types
